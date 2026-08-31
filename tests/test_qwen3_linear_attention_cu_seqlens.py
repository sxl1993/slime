from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

NUM_GPUS = 0


def install_megatron_stubs() -> None:
    required_modules = {
        "megatron.core",
        "megatron.core.models",
        "megatron.core.models.gpt",
        "megatron.core.models.gpt.gpt_layer_specs",
        "megatron.core.inference",
        "megatron.core.inference.contexts",
        "megatron.core.packed_seq_params",
        "megatron.core.transformer",
        "megatron.core.transformer.module",
        "megatron.core.transformer.spec_utils",
        "megatron.core.transformer.transformer_block",
        "megatron.core.transformer.transformer_layer",
    }
    if required_modules.issubset(sys.modules) and hasattr(sys.modules["megatron.core"], "tensor_parallel"):
        return

    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    models_mod = types.ModuleType("megatron.core.models")
    gpt_mod = types.ModuleType("megatron.core.models.gpt")
    gpt_layer_specs_mod = types.ModuleType("megatron.core.models.gpt.gpt_layer_specs")
    inference_mod = types.ModuleType("megatron.core.inference")
    inference_contexts_mod = types.ModuleType("megatron.core.inference.contexts")
    packed_seq_mod = types.ModuleType("megatron.core.packed_seq_params")
    transformer_mod = types.ModuleType("megatron.core.transformer")
    transformer_module_mod = types.ModuleType("megatron.core.transformer.module")
    spec_utils_mod = types.ModuleType("megatron.core.transformer.spec_utils")
    transformer_block_mod = types.ModuleType("megatron.core.transformer.transformer_block")
    transformer_layer_mod = types.ModuleType("megatron.core.transformer.transformer_layer")

    class PackedSeqParams:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class MegatronModule(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config

    class ModuleSpec:
        def __init__(self, module=None, params=None):
            self.module = module
            self.params = params or {}

    mpu_stub = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_group=lambda: None,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
    )
    tensor_parallel_stub = types.SimpleNamespace(
        gather_from_sequence_parallel_region=lambda x, group=None: x,
        scatter_to_sequence_parallel_region=lambda x, group=None: x,
    )

    gpt_layer_specs_mod.get_gpt_decoder_block_spec = lambda *args, **kwargs: None
    inference_contexts_mod.BaseInferenceContext = type("BaseInferenceContext", (), {})
    packed_seq_mod.PackedSeqParams = PackedSeqParams
    transformer_module_mod.MegatronModule = MegatronModule
    spec_utils_mod.ModuleSpec = ModuleSpec
    transformer_block_mod.get_num_layers_to_build = lambda *args, **kwargs: 0
    transformer_layer_mod.get_transformer_layer_offset = lambda *args, **kwargs: 0

    core_mod.mpu = mpu_stub
    core_mod.tensor_parallel = tensor_parallel_stub

    sys.modules["megatron"] = megatron_mod
    sys.modules["megatron.core"] = core_mod
    sys.modules["megatron.core.models"] = models_mod
    sys.modules["megatron.core.models.gpt"] = gpt_mod
    sys.modules["megatron.core.models.gpt.gpt_layer_specs"] = gpt_layer_specs_mod
    sys.modules["megatron.core.inference"] = inference_mod
    sys.modules["megatron.core.inference.contexts"] = inference_contexts_mod
    sys.modules["megatron.core.packed_seq_params"] = packed_seq_mod
    sys.modules["megatron.core.transformer"] = transformer_mod
    sys.modules["megatron.core.transformer.module"] = transformer_module_mod
    sys.modules["megatron.core.transformer.spec_utils"] = spec_utils_mod
    sys.modules["megatron.core.transformer.transformer_block"] = transformer_block_mod
    sys.modules["megatron.core.transformer.transformer_layer"] = transformer_layer_mod


class FakeShortConvolution(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, cu_seqlens=None, **kwargs):
        return x, None


class FakeFusedRMSNormGated(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, z):
        return x


def make_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=32,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        dtype=torch.float32,
    )


def load_module(module_name: str):
    install_megatron_stubs()
    sys.modules.pop("slime_plugins.models.hf_attention", None)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("module_name", "class_name", "args", "expected_backend"),
    [
        ("slime_plugins.models.qwen3_5", "Qwen3_5GatedDeltaNet", None, "fla"),
        (
            "slime_plugins.models.qwen3_5",
            "Qwen3_5GatedDeltaNet",
            SimpleNamespace(qwen_gdn_backend="flashqla"),
            "flashqla",
        ),
        ("slime_plugins.models.qwen3_next", "Qwen3NextGatedDeltaNet", None, "fla"),
        (
            "slime_plugins.models.qwen3_next",
            "Qwen3NextGatedDeltaNet",
            SimpleNamespace(qwen_gdn_backend="flashqla"),
            "flashqla",
        ),
    ],
)
def test_linear_attention_forwards_cu_seqlens_to_chunk_kernel(
    monkeypatch,
    module_name: str,
    class_name: str,
    args,
    expected_backend: str,
):
    module = load_module(module_name)

    monkeypatch.setattr(module.accelerator, "current_device", lambda: "cpu")
    monkeypatch.setattr(module, "ShortConvolution", FakeShortConvolution, raising=False)
    monkeypatch.setattr(module, "FusedRMSNormGated", FakeFusedRMSNormGated, raising=False)

    chunk_calls = []
    selected_backends = []

    def fake_chunk_gated_delta_rule(
        q,
        k,
        v,
        *,
        g,
        beta,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens=None,
        **kwargs,
    ):
        chunk_calls.append(cu_seqlens.clone() if cu_seqlens is not None else None)
        assert q.shape[0] == 1
        assert cu_seqlens is not None
        return torch.zeros_like(v), None

    def fake_get_chunk_gated_delta_rule(backend):
        selected_backends.append(backend)
        return fake_chunk_gated_delta_rule

    monkeypatch.setattr(module, "get_chunk_gated_delta_rule", fake_get_chunk_gated_delta_rule)

    layer = getattr(module, class_name)(make_config(), layer_idx=0, args=args)
    hidden_states = torch.randn(1, 7, 32)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)

    output = layer(hidden_states, cu_seqlens=cu_seqlens)

    assert selected_backends == [expected_backend]
    assert layer.gdn_backend == expected_backend
    assert output.shape == hidden_states.shape
    assert len(chunk_calls) == 1
    assert torch.equal(chunk_calls[0], cu_seqlens)


@pytest.mark.unit
def test_hf_attention_uses_cached_cp_slices_without_reading_cu_seqlens(monkeypatch):
    module = load_module("slime_plugins.models.hf_attention")

    class IdentityAttention(module.HuggingfaceAttention):
        def __init__(self):
            nn.Module.__init__(self)
            self.args = SimpleNamespace(sequence_parallel=False)

        def hf_forward(self, hidden_states, packed_seq_params):
            del packed_seq_params
            return hidden_states

    attention = IdentityAttention()
    rank_inputs = [torch.arange(12, dtype=torch.float32).reshape(12, 1, 1) + 1000 * rank for rank in range(4)]

    class GuardedCuSeqlens:
        def __len__(self):
            raise AssertionError("cached CP slices should avoid reading cu_seqlens")

        def __getitem__(self, index):
            del index
            raise AssertionError("cached CP slices should avoid indexing cu_seqlens")

    packed_seq_params = module.PackedSeqParams(cu_seqlens_q=GuardedCuSeqlens())
    packed_seq_params.hf_attention_cp_slices = (
        (
            (0, 0, 2),
            (1, 0, 2),
            (2, 0, 2),
            (3, 0, 2),
            (3, 2, 4),
            (2, 2, 4),
            (1, 2, 4),
            (0, 2, 4),
            (0, 4, 8),
            (1, 4, 8),
            (2, 4, 8),
            (3, 4, 8),
            (3, 8, 12),
            (2, 8, 12),
            (1, 8, 12),
            (0, 8, 12),
        ),
        (
            ((0, 2), (14, 16), (16, 20), (44, 48)),
            ((2, 4), (12, 14), (20, 24), (40, 44)),
            ((4, 6), (10, 12), (24, 28), (36, 40)),
            ((6, 8), (8, 10), (28, 32), (32, 36)),
        ),
    )

    monkeypatch.setattr(module.mpu, "get_context_parallel_world_size", lambda: 4)
    monkeypatch.setattr(module.mpu, "get_context_parallel_group", lambda: None)
    monkeypatch.setattr(
        module._AllGatherForDuplicatedComputation,
        "apply",
        staticmethod(lambda hidden_states, group: tuple(rank_inputs)),
    )

    for rank in range(4):
        monkeypatch.setattr(module.mpu, "get_context_parallel_rank", lambda r=rank: r)
        output, _ = module.HuggingfaceAttention.forward(
            attention,
            rank_inputs[rank],
            attention_mask=None,
            packed_seq_params=packed_seq_params,
        )
        assert torch.equal(output, rank_inputs[rank])


@pytest.mark.unit
def test_get_batch_attaches_hf_attention_cp_slices(monkeypatch):
    install_megatron_stubs()
    sys.modules.pop("slime.backends.megatron_utils.data", None)
    data_module = importlib.import_module("slime.backends.megatron_utils.data")
    cp_module = importlib.import_module("slime.backends.megatron_utils.cp_utils")

    for module in (data_module, cp_module):
        monkeypatch.setattr(module.mpu, "get_context_parallel_world_size", lambda: 4)
        monkeypatch.setattr(module.mpu, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(data_module.accelerator, "device", lambda: "cpu")
    monkeypatch.setattr(data_module.accelerator, "current_device", lambda: "cpu")

    iterator = data_module.DataIterator(
        {
            "tokens": [torch.arange(16), torch.arange(32)],
            "loss_masks": [torch.ones(8), torch.ones(16)],
            "total_lengths": [16, 32],
            "response_lengths": [8, 16],
        },
        [[0, 1]],
    )
    batch = data_module.get_batch(
        iterator,
        ["tokens", "loss_masks", "total_lengths", "response_lengths"],
        pad_multiplier=1,
    )

    assert batch["packed_seq_params"].hf_attention_cp_slices == (
        (
            (0, 0, 2),
            (1, 0, 2),
            (2, 0, 2),
            (3, 0, 2),
            (3, 2, 4),
            (2, 2, 4),
            (1, 2, 4),
            (0, 2, 4),
            (0, 4, 8),
            (1, 4, 8),
            (2, 4, 8),
            (3, 4, 8),
            (3, 8, 12),
            (2, 8, 12),
            (1, 8, 12),
            (0, 8, 12),
        ),
        (
            ((0, 2), (14, 16), (16, 20), (44, 48)),
            ((2, 4), (12, 14), (20, 24), (40, 44)),
            ((4, 6), (10, 12), (24, 28), (36, 40)),
            ((6, 8), (8, 10), (28, 32), (32, 36)),
        ),
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
