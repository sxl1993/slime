import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

NUM_GPUS = 0


def _load_model_provider(monkeypatch):
    modules = {
        "megatron": types.ModuleType("megatron"),
        "megatron.core": types.ModuleType("megatron.core"),
        "megatron.core.models": types.ModuleType("megatron.core.models"),
        "megatron.core.models.gpt": types.ModuleType("megatron.core.models.gpt"),
        "megatron.core.models.gpt.gpt_layer_specs": types.ModuleType("megatron.core.models.gpt.gpt_layer_specs"),
        "megatron.core.transformer": types.ModuleType("megatron.core.transformer"),
        "megatron.core.transformer.spec_utils": types.ModuleType("megatron.core.transformer.spec_utils"),
        "megatron.core.transformer.transformer_config": types.ModuleType(
            "megatron.core.transformer.transformer_config"
        ),
        "megatron.training": types.ModuleType("megatron.training"),
        "megatron.training.arguments": types.ModuleType("megatron.training.arguments"),
        "slime.utils.misc": types.ModuleType("slime.utils.misc"),
    }
    modules["megatron.core"].tensor_parallel = types.SimpleNamespace()
    modules["megatron.core.models.gpt"].GPTModel = torch.nn.Module
    layer_specs = modules["megatron.core.models.gpt.gpt_layer_specs"]
    layer_specs.get_gpt_decoder_block_spec = lambda *args, **kwargs: None
    layer_specs.get_gpt_layer_local_spec = lambda *args, **kwargs: None
    layer_specs.get_gpt_layer_with_transformer_engine_spec = lambda *args, **kwargs: None
    modules["megatron.core.transformer.spec_utils"].import_module = lambda value: value
    modules["megatron.core.transformer.transformer_config"].TransformerConfig = object
    modules["megatron.training.arguments"].core_transformer_config_from_args = lambda args: object()
    modules["slime.utils.misc"].load_function = lambda value: value
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "backends" / "megatron_utils" / "model_provider.py"
    module_name = "test_model_provider_freeze_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _GLMIndexerAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.wq_b = torch.nn.Linear(2, 2, bias=False)
        self.wk = torch.nn.Linear(2, 2, bias=False)
        self.k_norm = torch.nn.LayerNorm(2)
        self.weights_proj = torch.nn.Linear(2, 2, bias=False)
        self.linear_q_down_proj = torch.nn.Linear(2, 2, bias=False)


class _UpstreamDSAAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.core_attention = torch.nn.Module()
        self.core_attention.indexer = torch.nn.Module()
        self.core_attention.indexer.linear_wq_b = torch.nn.Linear(2, 2, bias=False)
        self.core_attention.indexer.linear_wk = torch.nn.Linear(2, 2, bias=False)
        self.core_attention.indexer.k_norm = torch.nn.LayerNorm(2)
        self.core_attention.indexer.linear_weights_proj = torch.nn.Linear(2, 2, bias=False)
        self.core_attention.regular_projection = torch.nn.Linear(2, 2, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.self_attention = attention
        self.mlp = torch.nn.Linear(2, 2, bias=False)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(_GLMIndexerAttention()), _Layer(_UpstreamDSAAttention())])
        self.output_layer = torch.nn.Linear(2, 2, bias=False)


class _FullAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_qkv = torch.nn.Linear(2, 2, bias=False)
        self.linear_proj = torch.nn.Linear(2, 2, bias=False)
        self.q_layernorm = torch.nn.LayerNorm(2)
        self.k_layernorm = torch.nn.LayerNorm(2)


class _LinearAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(2)
        self.linear_attn = torch.nn.Module()
        self.linear_attn.conv1d = torch.nn.Conv1d(2, 2, 1, bias=False)
        self.linear_attn.in_proj_qkv = torch.nn.Linear(2, 2, bias=False)
        self.linear_attn.in_proj_z = torch.nn.Linear(2, 2, bias=False)
        self.linear_attn.in_proj_b = torch.nn.Linear(2, 2, bias=False)
        self.linear_attn.in_proj_a = torch.nn.Linear(2, 2, bias=False)
        self.linear_attn.dt_bias = torch.nn.Parameter(torch.ones(2))
        self.linear_attn.A_log = torch.nn.Parameter(torch.ones(2))
        self.linear_attn.norm = torch.nn.LayerNorm(2)
        self.linear_attn.out_proj = torch.nn.Linear(2, 2, bias=False)


class _AttentionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_Layer(_FullAttention()), _Layer(_LinearAttention())])
        self.output_layer = torch.nn.Linear(2, 1, bias=False)


@pytest.mark.unit
def test_freeze_indexer_covers_glm_and_upstream_dsa_names(monkeypatch):
    model_provider = _load_model_provider(monkeypatch)
    model = _Model()
    args = types.SimpleNamespace(
        only_train_params_name_list=None,
        freeze_params_name_list=None,
        freeze_indexer=True,
    )

    model_provider.freeze_model_params(model, args)

    frozen = set(model._slime_frozen_indexer_param_names)
    assert frozen
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert not parameter.requires_grad, name
        else:
            assert parameter.requires_grad, name
    assert "layers.0.self_attention.linear_q_down_proj.weight" not in frozen
    assert "layers.1.self_attention.core_attention.regular_projection.weight" not in frozen
    assert "layers.0.mlp.weight" not in frozen
    assert "output_layer.weight" not in frozen


@pytest.mark.unit
def test_freeze_indexer_rejects_unrecognized_attention(monkeypatch):
    model_provider = _load_model_provider(monkeypatch)
    model = _Layer(torch.nn.MultiheadAttention(2, 1))
    args = types.SimpleNamespace(
        only_train_params_name_list=None,
        freeze_params_name_list=None,
        freeze_indexer=True,
    )

    with pytest.raises(RuntimeError, match="no recognized DSA indexer"):
        model_provider.freeze_model_params(model, args)


@pytest.mark.unit
def test_sao_critic_attention_freeze_preserves_norms_and_non_attention_params(monkeypatch):
    model_provider = _load_model_provider(monkeypatch)
    model = _AttentionModel()
    args = types.SimpleNamespace(
        only_train_params_name_list=None,
        freeze_params_name_list=None,
        freeze_indexer=False,
        sao_critic_freeze_attention=True,
    )

    model_provider.wrap_model_provider_with_freeze(
        lambda pre_process=True, post_process=True: model,
        args,
        role="critic",
    )()

    frozen = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    assert frozen == {
        "layers.0.self_attention.linear_qkv.weight",
        "layers.0.self_attention.linear_proj.weight",
        "layers.1.self_attention.linear_attn.conv1d.weight",
        "layers.1.self_attention.linear_attn.in_proj_qkv.weight",
        "layers.1.self_attention.linear_attn.in_proj_z.weight",
        "layers.1.self_attention.linear_attn.in_proj_b.weight",
        "layers.1.self_attention.linear_attn.in_proj_a.weight",
        "layers.1.self_attention.linear_attn.dt_bias",
        "layers.1.self_attention.linear_attn.A_log",
        "layers.1.self_attention.linear_attn.out_proj.weight",
    }

    for name, parameter in model.named_parameters():
        if name not in frozen:
            assert parameter.requires_grad, name


@pytest.mark.unit
def test_sao_critic_attention_freeze_is_opt_in_and_critic_only(monkeypatch):
    model_provider = _load_model_provider(monkeypatch)
    args = types.SimpleNamespace(
        only_train_params_name_list=None,
        freeze_params_name_list=None,
        freeze_indexer=False,
        sao_critic_freeze_attention=True,
    )

    disabled_model = _AttentionModel()
    disabled_args = types.SimpleNamespace(**vars(args))
    disabled_args.sao_critic_freeze_attention = False
    model_provider.wrap_model_provider_with_freeze(
        lambda pre_process=True, post_process=True: disabled_model,
        disabled_args,
        role="critic",
    )()
    assert all(parameter.requires_grad for parameter in disabled_model.parameters())

    actor_model = _AttentionModel()
    model_provider.wrap_model_provider_with_freeze(
        lambda pre_process=True, post_process=True: actor_model,
        args,
        role="actor",
    )()
    assert all(parameter.requires_grad for parameter in actor_model.parameters())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
