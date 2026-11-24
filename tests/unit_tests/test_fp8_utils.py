# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from megatron.core import fp8_utils
from megatron.core.transformer import TransformerConfig
from megatron.training.utils import get_device_arch_version
from tests.unit_tests.test_utilities import Utils

try:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    HAVE_MXFP8_TENSOR = True
except ImportError:
    HAVE_MXFP8_TENSOR = False

try:
    from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer

    HAVE_BLOCKWISE_FP8_TENSOR = True
except ImportError:
    HAVE_BLOCKWISE_FP8_TENSOR = False

# MXFP8 needs Blackwell or newer.
mxfp8_available = HAVE_MXFP8_TENSOR and get_device_arch_version() >= 10
reason_for_no_mxfp8 = "MXFP8 requires Transformer Engine and device arch >= 10"

blockwise_fp8_available = (
    HAVE_BLOCKWISE_FP8_TENSOR
    and fp8_utils.te_post_all_gather_processing is not None
    and torch.cuda.is_available()
    and get_device_arch_version() >= 9
)
reason_for_no_blockwise_fp8 = (
    "Blockwise FP8 post_all_gather_processing requires Transformer Engine and device arch >= 9"
)


@pytest.mark.skipif(not fp8_utils.HAVE_TE, reason="Transformer Engine is not installed")
@pytest.mark.parametrize(
    ("is_init", "config_values", "te_helper"),
    [
        (
            False,
            {"fp8": "hybrid", "fp4": None, "fp8_param": False, "fp4_param": False},
            "fp8_autocast",
        ),
        (True, {"fp8": None, "fp4": None, "fp8_param": True, "fp4_param": False}, "fp8_model_init"),
    ],
)
def test_get_fp8_disabled_context_uses_disabled_te_context(is_init, config_values, te_helper):
    config = Mock(**config_values)
    disabled_context = Mock()

    with patch.object(
        fp8_utils.transformer_engine.pytorch, te_helper, return_value=disabled_context
    ) as te_context:
        result = fp8_utils.get_fp8_disabled_context(config, is_init=is_init)

    assert result is disabled_context
    te_context.assert_called_once_with(enabled=False)


def _dummy_quantizer_factory(*args, **kwargs):
    return None


@pytest.mark.skipif(not fp8_utils.HAVE_TE, reason="Transformer Engine required")
def test_fp8_model_init_uses_custom_recipe_attrs(monkeypatch):
    if not hasattr(fp8_utils.transformer_engine.common.recipe, "CustomRecipe"):
        pytest.skip("CustomRecipe requires newer Transformer Engine")

    captured = {}

    def fake_fp8_model_init(enabled=False, recipe=None, **kwargs):
        captured["enabled"] = enabled
        captured["recipe"] = recipe
        captured["kwargs"] = kwargs
        return nullcontext()

    monkeypatch.setattr(fp8_utils.transformer_engine.pytorch, "fp8_model_init", fake_fp8_model_init)

    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        fp8="e4m3",
        fp8_param=True,
        fp8_recipe="custom",
        fp8_quantizer_factory="tests.unit_tests.test_fp8_utils._dummy_quantizer_factory",
        fp8_recipe_attrs={"_test_recipe_flag": "set"},
    )

    with fp8_utils.get_fp8_context(config, is_init=True):
        pass

    assert captured["enabled"] is True
    assert isinstance(captured["recipe"], fp8_utils.transformer_engine.common.recipe.CustomRecipe)
    assert captured["recipe"]._test_recipe_flag == "set"


class MockTELinear(nn.Module):
    """Mock TE Linear module for testing."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x):
        return x @ self.weight.t()


class TestFP8Padding:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        # Clear the wrapped modules set before each test
        fp8_utils._fp8_inference_wrapped_modules.clear()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        fp8_utils._fp8_inference_wrapped_modules.clear()

    def test_prepare_model_for_fp8_inference_basic(self):
        """Test prepare_model_for_fp8_inference wraps TE modules."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.te_layer = MockTELinear(128, 128)
                self.regular_layer = nn.Linear(128, 128)

        with (
            patch.object(fp8_utils, 'HAVE_TE', True),
            patch.object(fp8_utils, 'Fp8Padding'),
            patch.object(fp8_utils, 'Fp8Unpadding'),
            patch.object(fp8_utils, 'TE_LINEAR_TYPES', (MockTELinear,)),
        ):

            model = SimpleModel()
            original_te_forward = model.te_layer.forward
            original_regular_forward = model.regular_layer.forward

            # Prepare model
            prepared_model = fp8_utils.prepare_model_for_fp8_inference(model)

            # Check same model returned
            assert prepared_model is model

            # Check TE layer was wrapped
            assert model.te_layer.forward != original_te_forward
            assert model.te_layer in fp8_utils._fp8_inference_wrapped_modules

            # Check regular layer was not wrapped
            assert model.regular_layer.forward == original_regular_forward

    def test_padding_mechanism_works(self):
        """Test that the padding mechanism actually pads and unpads correctly."""

        with (
            patch.object(fp8_utils, 'HAVE_TE', True),
            patch.object(fp8_utils, 'Fp8Padding') as mock_pad_class,
            patch.object(fp8_utils, 'Fp8Unpadding') as mock_unpad_class,
        ):

            # Setup padding mock to pad from 6 to 16
            mock_pad_instance = Mock()
            mock_pad_instance.return_value = (torch.zeros(16, 8192), [16])
            mock_pad_class.return_value = mock_pad_instance

            # Setup unpadding mock to unpad from 16 to 6
            mock_unpad_instance = Mock()
            mock_unpad_instance.return_value = torch.zeros(6, 8192)
            mock_unpad_class.return_value = mock_unpad_instance

            # Create module and get access to padded_forward directly
            module = MockTELinear(4096, 4096)
            module.cuda()

            # Store original forward to track what it receives
            original_forward_input = None

            def track_forward(x):
                nonlocal original_forward_input
                original_forward_input = x
                return torch.randn(x.shape[0], x.shape[1], 4096).cuda()

            module.forward = track_forward

            # Manually create the wrapped forward function
            fp8_utils._wrap_te_linear_for_padding(module)
            padded_forward = module.forward

            # Mock FP8GlobalStateManager.is_fp8_enabled to return True
            with patch(
                'transformer_engine.pytorch.fp8.FP8GlobalStateManager.is_fp8_enabled',
                return_value=True,
            ):
                # Create input: (seq_len=6, batch=2, hidden=4096)
                input_tensor = torch.randn(6, 2, 4096).cuda()

                # Call padded_forward directly
                output = padded_forward(input_tensor)

            # Verify padding was called with correct reshaped input
            mock_pad_instance.assert_called_once()
            call_args = mock_pad_instance.call_args[0]
            assert call_args[0].shape == (6, 8192)  # Reshaped to 2D
            assert call_args[1] == [6]  # Split info

            # Verify the original forward received padded input with correct shape
            assert original_forward_input.shape == (16, 2, 4096)  # Padded to 16

            # Verify unpadding was called
            mock_unpad_instance.assert_called_once()
            unpad_args = mock_unpad_instance.call_args[0]
            assert unpad_args[0].shape == (16, 8192)  # Padded 2D tensor
            assert unpad_args[1] == [6]  # Original split

            # Verify output has original shape
            assert output.shape == (6, 2, 4096)  # Back to original seq_len


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
class TestCopyTensorsToQuantizedParams:
    """Cover the batched MXFP8 param copy-back used by _post_param_sync.

    ``copy_tensors_to_quantized_params`` bypasses ``copy_`` and calls the destination quantizer
    directly, so the contract to protect is that it still writes exactly what the per-param
    ``copy_tensor_to_quantized_param`` would have written.
    """

    SHAPES = [(1024, 512), (2048, 256), (512, 1024)]

    def _make_param(self, shape):
        quantizer = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True)
        tensor = quantizer.make_empty(shape, dtype=torch.bfloat16, device="cuda")
        return torch.nn.Parameter(tensor, requires_grad=False)

    def _raw_buffers(self, param):
        """The four buffers MXFP8 storage is made of, i.e. everything a cast writes."""
        data = param.data
        return (
            data._rowwise_data,
            data._rowwise_scale_inv,
            data._columnwise_data,
            data._columnwise_scale_inv,
        )

    def test_matches_per_param_copy(self):
        """Batched copy-back is bitwise identical to copying one param at a time."""
        torch.manual_seed(0)
        reference_params = [self._make_param(shape) for shape in self.SHAPES]
        batched_params = [self._make_param(shape) for shape in self.SHAPES]
        # Sources are flat slices, matching how _post_param_sync views the param buffer.
        srcs = [
            torch.randn(shape, dtype=torch.bfloat16, device="cuda").view(-1)
            for shape in self.SHAPES
        ]

        for param, src in zip(reference_params, srcs):
            fp8_utils.copy_tensor_to_quantized_param(param, src)
        fp8_utils.copy_tensors_to_quantized_params(batched_params, srcs)
        torch.cuda.synchronize()

        for reference, batched in zip(reference_params, batched_params):
            for expected, actual in zip(self._raw_buffers(reference), self._raw_buffers(batched)):
                assert torch.equal(expected, actual)

    def test_falls_back_without_quantizer(self):
        """A destination with no quantizer of its own still gets written."""
        param = self._make_param(self.SHAPES[0])
        param.data._quantizer = None
        src = torch.randn(self.SHAPES[0], dtype=torch.bfloat16, device="cuda").view(-1)

        fp8_utils.copy_tensors_to_quantized_params([param], [src])
        torch.cuda.synchronize()

        # A quantized copy of a non-zero source cannot be all zeros.
        assert param.data._rowwise_data.any()

    def test_empty_input(self):
        """No params is a no-op rather than an error."""
        fp8_utils.copy_tensors_to_quantized_params([], [])


class _FakeCudaBuffer:
    """Minimal stand-in for a CUDA tensor supporting record_stream."""

    def __init__(self, device="cuda:0"):
        self.is_cuda = True
        self.device = device
        self.streams = []

    def record_stream(self, stream):
        self.streams.append(stream)


class _FakeBlockwiseWeight:
    """Minimal stand-in for a TE Float8BlockwiseQTensorStorage weight."""

    def __init__(self, *, is_2d=True, has_rowwise=True, has_col=True, has_col_scale=True):
        self._is_2D_scaled = is_2d
        self._rowwise_data = _FakeCudaBuffer() if has_rowwise else None
        self._rowwise_scale_inv = _FakeCudaBuffer() if has_rowwise else None
        self._columnwise_data = _FakeCudaBuffer() if has_col else None
        self._columnwise_scale_inv = _FakeCudaBuffer() if has_col_scale else None


class _FakeTensorwiseWeight:
    """Minimal stand-in for a TE tensorwise/delayed Float8Tensor weight."""

    def __init__(self, *, has_data=True, has_transpose=True):
        self._data = _FakeCudaBuffer() if has_data else None
        self._transpose = _FakeCudaBuffer() if has_transpose else None
        self._transpose_invalid = not has_transpose


class _FakeMXFP8Weight:
    """Minimal stand-in for a TE MXFP8Tensor weight."""

    def __init__(self):
        self._rowwise_data = _FakeCudaBuffer()
        self._rowwise_scale_inv = _FakeCudaBuffer()
        self._columnwise_data = _FakeCudaBuffer()
        self._columnwise_scale_inv = _FakeCudaBuffer()


class _FakeGroupedWeight:
    """Minimal stand-in for a TE quantized GroupedTensor weight."""

    def __init__(self, members=None, recipe=None, *, has_rowwise=True):
        self.rowwise_data = SimpleNamespace(dtype=torch.uint8) if has_rowwise else None
        self.columnwise_data = _FakeCudaBuffer()
        self.columnwise_scale_inv = _FakeCudaBuffer()
        self.quantized_tensors = members
        self.quantizer = (
            SimpleNamespace(_get_compatible_recipe=lambda: recipe) if recipe is not None else None
        )


class TestFP8RowwiseOnlyPostAllGather:
    """Unit tests for rowwise-only FP8 parameter metadata and post-all-gather filtering."""

    def _patch_fake_te_types(self, monkeypatch):
        monkeypatch.setattr(fp8_utils, "HAVE_TE_BLOCKWISE_FP8TENSOR", True)
        monkeypatch.setattr(
            fp8_utils, "Float8BlockwiseQTensorStorage", _FakeBlockwiseWeight, raising=False
        )
        monkeypatch.setattr(fp8_utils, "HAVE_TE_MXFP8TENSOR", True)
        monkeypatch.setattr(fp8_utils, "MXFP8Tensor", _FakeMXFP8Weight, raising=False)
        monkeypatch.setattr(fp8_utils, "HAVE_TE_FP8_TENSOR_CLASS", True)
        monkeypatch.setattr(fp8_utils, "FP8_TENSOR_CLASS", _FakeTensorwiseWeight, raising=False)
        monkeypatch.setattr(fp8_utils, "HAVE_TE_GROUPED_TENSOR_CLASS", True)
        monkeypatch.setattr(fp8_utils, "GroupedTensor", _FakeGroupedWeight, raising=False)

    def test_post_all_gather_skips_rowwise_only_params(self, monkeypatch):
        """Rowwise-only parameters are excluded from te_post_all_gather_processing."""
        tagged_a, tagged_b, normal = (
            _FakeBlockwiseWeight(),
            _FakeBlockwiseWeight(),
            _FakeBlockwiseWeight(),
        )
        fp8_utils.set_fp8_param_rowwise_only(tagged_a)
        fp8_utils.set_fp8_param_rowwise_only(tagged_b)

        captured = []
        monkeypatch.setattr(fp8_utils, "te_post_all_gather_processing", captured.append)

        fp8_utils.post_all_gather_processing([tagged_a, normal])
        assert captured == [[normal]]

        fp8_utils.post_all_gather_processing((tagged_a, normal))
        assert captured[-1] == [normal]

        # When every parameter in the call is rowwise-only, TE is not invoked at all.
        fp8_utils.post_all_gather_processing([tagged_a, tagged_b])
        fp8_utils.post_all_gather_processing(tagged_a)
        assert len(captured) == 2

    def test_post_all_gather_grouped_container_and_member_filtering(self, monkeypatch):
        """Both whole GroupedTensor and individual grouped members can opt out."""
        self._patch_fake_te_types(monkeypatch)
        m0, m1 = _FakeBlockwiseWeight(), _FakeBlockwiseWeight()
        grouped = _FakeGroupedWeight(members=[m0, m1])
        normal = _FakeBlockwiseWeight()

        captured = []
        monkeypatch.setattr(fp8_utils, "te_post_all_gather_processing", captured.append)

        # Member-level opt-out skips only the tagged member.
        fp8_utils.set_fp8_param_rowwise_only(m0)
        fp8_utils.post_all_gather_processing([grouped, normal])
        assert captured == [[m1, normal]]

        # Container-level opt-out skips the whole GroupedTensor without touching members.
        fp8_utils.set_fp8_param_rowwise_only(grouped, True)
        assert fp8_utils.is_fp8_param_rowwise_only(grouped)
        assert fp8_utils.is_fp8_param_rowwise_only(m0)
        assert fp8_utils.is_fp8_param_rowwise_only(m1)
        monkeypatch.setattr(
            fp8_utils,
            "get_grouped_quantized_members",
            Mock(side_effect=AssertionError("should not expand rowwise-only grouped container")),
        )
        fp8_utils.post_all_gather_processing([grouped, normal])
        assert captured[-1] == [normal]

        # Unmarking clears the container and cached members so normal processing resumes.
        fp8_utils.set_fp8_param_rowwise_only(grouped, False)
        assert not fp8_utils.is_fp8_param_rowwise_only(grouped)
        assert not fp8_utils.is_fp8_param_rowwise_only(m0)
        assert not fp8_utils.is_fp8_param_rowwise_only(m1)

    def test_set_rowwise_only_validation_and_toggle(self):
        """Invalid arguments raise ValueError and toggling False overrides unwrapped attrs."""
        param = nn.Parameter(torch.zeros(4, 4))
        with pytest.raises(ValueError, match="release_columnwise=True requires rowwise_only=True"):
            fp8_utils.set_fp8_param_rowwise_only(param, False, release_columnwise=True)

        assert not fp8_utils.is_fp8_param_rowwise_only(param)
        setattr(param.data, fp8_utils.FP8_PARAM_ROWWISE_ONLY_ATTR, True)
        assert fp8_utils.is_fp8_param_rowwise_only(param)

        fp8_utils.set_fp8_param_rowwise_only(param, False)
        assert not fp8_utils.is_fp8_param_rowwise_only(param)

        fp8_utils.set_fp8_param_rowwise_only(param, True)
        assert fp8_utils.is_fp8_param_rowwise_only(param)
        assert getattr(param, fp8_utils.FP8_PARAM_ROWWISE_ONLY_ATTR) is True

    def test_release_blockwise_2d_records_stream_and_is_idempotent(self, monkeypatch):
        """Releasing 2-D blockwise columns records the current CUDA stream and is idempotent."""
        self._patch_fake_te_types(monkeypatch)
        stream = object()
        stream_calls = []
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        monkeypatch.setattr(
            torch.cuda, "current_stream", lambda device=None: stream_calls.append(device) or stream
        )

        weight = _FakeBlockwiseWeight()
        col_data, col_scale = weight._columnwise_data, weight._columnwise_scale_inv
        row_data, row_scale = weight._rowwise_data, weight._rowwise_scale_inv

        fp8_utils.set_fp8_param_rowwise_only(weight, True, release_columnwise=True)
        assert fp8_utils.is_fp8_param_rowwise_only(weight)
        assert weight._columnwise_data is None
        assert weight._columnwise_scale_inv is None
        assert weight._rowwise_data is row_data
        assert weight._rowwise_scale_inv is row_scale
        assert col_data.streams == [stream]
        assert col_scale.streams == [stream]
        assert len(stream_calls) == 2

        # Second release is a no-op and does not query current_stream again.
        assert fp8_utils.release_fp8_columnwise_storage(weight) is False
        assert len(stream_calls) == 2

    def test_release_guards_non_derivable_and_invalid_tensors(self, monkeypatch):
        """MXFP8, NVFP4, 1-D blockwise, missing-rowwise, and plain tensors are never released."""
        self._patch_fake_te_types(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: object())

        one_dim_blockwise = _FakeBlockwiseWeight(is_2d=False)
        missing_rowwise = _FakeBlockwiseWeight(has_rowwise=False)
        mxfp8 = _FakeMXFP8Weight()
        grouped_mxfp8 = _FakeGroupedWeight(
            members=[_FakeMXFP8Weight()],
            recipe=SimpleNamespace(mxfp8=lambda: True, nvfp4=lambda: False),
        )
        grouped_nvfp4 = _FakeGroupedWeight(
            members=[_FakeBlockwiseWeight()],
            recipe=SimpleNamespace(mxfp8=lambda: False, nvfp4=lambda: True),
        )
        plain = nn.Parameter(torch.zeros(4, 4))

        for tensor in (
            one_dim_blockwise,
            missing_rowwise,
            mxfp8,
            grouped_mxfp8,
            grouped_nvfp4,
            plain,
        ):
            assert fp8_utils.release_fp8_columnwise_storage(tensor) is False

        assert one_dim_blockwise._columnwise_data is not None
        assert missing_rowwise._columnwise_data is not None
        assert mxfp8._columnwise_data is not None
        assert grouped_mxfp8.columnwise_data is not None
        assert grouped_mxfp8.quantized_tensors[0]._columnwise_data is not None
        assert grouped_nvfp4.columnwise_data is not None
        assert grouped_nvfp4.quantized_tensors[0]._columnwise_data is not None

    def test_release_tensorwise_transpose_cache(self, monkeypatch):
        """Tensorwise Float8Tensor transpose cache is cleared and marked invalid."""
        self._patch_fake_te_types(monkeypatch)
        stream = object()
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream)

        weight = _FakeTensorwiseWeight()
        old_transpose = weight._transpose

        assert fp8_utils.release_fp8_columnwise_storage(weight) is True
        assert weight._transpose is None
        assert weight._transpose_invalid is True
        assert old_transpose.streams == [stream]
        assert fp8_utils.release_fp8_columnwise_storage(weight) is False

    def test_release_grouped_blockwise_releases_container_and_members(self, monkeypatch):
        """Grouped 2-D blockwise release drops both member and container columnwise storage."""
        self._patch_fake_te_types(monkeypatch)
        stream = object()
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream)

        m0, m1 = _FakeBlockwiseWeight(), _FakeBlockwiseWeight()
        grouped = _FakeGroupedWeight(
            members=[m0, m1],
            recipe=SimpleNamespace(
                mxfp8=lambda: False, nvfp4=lambda: False, float8_block_scaling=lambda: True
            ),
        )
        container_col, container_scale = grouped.columnwise_data, grouped.columnwise_scale_inv

        assert fp8_utils.release_fp8_columnwise_storage(grouped) is True
        assert grouped.columnwise_data is None
        assert grouped.columnwise_scale_inv is None
        assert container_col.streams == [stream]
        assert container_scale.streams == [stream]
        for member in (m0, m1):
            assert member._columnwise_data is None
            assert member._columnwise_scale_inv is None

    def test_release_during_cuda_graph_capture_raises(self, monkeypatch):
        """Releasing CUDA columnwise storage inside CUDA graph capture is rejected."""
        self._patch_fake_te_types(monkeypatch)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

        weight = _FakeBlockwiseWeight()
        with pytest.raises(RuntimeError, match="CUDA graph capture"):
            fp8_utils.release_fp8_columnwise_storage(weight)
        assert weight._columnwise_data is not None

    @pytest.mark.skipif(not blockwise_fp8_available, reason=reason_for_no_blockwise_fp8)
    def test_blockwise_real_te_rowwise_only_skips_columnwise_materialization(self):
        """End-to-end TE check: rowwise-only blockwise weight stays column-free after gather."""
        torch.manual_seed(0)
        quantizer = Float8BlockQuantizer(
            fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=True, block_scaling_dim=2
        )
        src_a = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
        src_b = torch.randn(256, 256, dtype=torch.bfloat16, device="cuda")
        rowwise_param = nn.Parameter(quantizer(src_a), requires_grad=True)
        normal_param = nn.Parameter(quantizer(src_b), requires_grad=True)
        expected_dequant = rowwise_param.dequantize().clone()

        fp8_utils.set_fp8_param_rowwise_only(rowwise_param, True, release_columnwise=True)
        assert rowwise_param._columnwise_data is None
        assert rowwise_param._columnwise_scale_inv is None
        assert rowwise_param._rowwise_data is not None

        # Drop normal_param's columnwise buffers without marking it rowwise-only,
        # then verify post_all_gather_processing rebuilds only normal_param.
        assert fp8_utils.release_fp8_columnwise_storage(normal_param) is True
        assert normal_param._columnwise_data is None

        fp8_utils.post_all_gather_processing([rowwise_param, normal_param])
        torch.cuda.synchronize()

        assert rowwise_param._columnwise_data is None
        assert rowwise_param._columnwise_scale_inv is None
        assert torch.equal(rowwise_param.dequantize(), expected_dequant)
        assert normal_param._columnwise_data is not None
        assert normal_param._columnwise_scale_inv is not None
