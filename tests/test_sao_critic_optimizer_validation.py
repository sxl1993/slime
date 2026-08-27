import dataclasses
import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest
import torch

from test_model_provider_freeze import _AttentionModel, _load_model_provider


def _load_model_module(monkeypatch):
    modules = {
        "megatron": types.ModuleType("megatron"),
        "megatron.core": types.ModuleType("megatron.core"),
        "megatron.core.distributed": types.ModuleType("megatron.core.distributed"),
        "megatron.core.enums": types.ModuleType("megatron.core.enums"),
        "megatron.core.models": types.ModuleType("megatron.core.models"),
        "megatron.core.models.gpt": types.ModuleType("megatron.core.models.gpt"),
        "megatron.core.optimizer": types.ModuleType("megatron.core.optimizer"),
        "megatron.core.optimizer.optimizer": types.ModuleType("megatron.core.optimizer.optimizer"),
        "megatron.core.optimizer_param_scheduler": types.ModuleType("megatron.core.optimizer_param_scheduler"),
        "megatron.core.pipeline_parallel": types.ModuleType("megatron.core.pipeline_parallel"),
        "megatron.core.utils": types.ModuleType("megatron.core.utils"),
        "megatron.core.mpu": types.ModuleType("megatron.core.mpu"),
        "megatron.training": types.ModuleType("megatron.training"),
        "megatron.training.global_vars": types.ModuleType("megatron.training.global_vars"),
        "megatron.training.training": types.ModuleType("megatron.training.training"),
        "tqdm": types.ModuleType("tqdm"),
        "slime.backends.megatron_utils.checkpoint": types.ModuleType("slime.backends.megatron_utils.checkpoint"),
        "slime.backends.megatron_utils.data": types.ModuleType("slime.backends.megatron_utils.data"),
        "slime.backends.megatron_utils.loss": types.ModuleType("slime.backends.megatron_utils.loss"),
        "slime.backends.megatron_utils.model_provider": types.ModuleType(
            "slime.backends.megatron_utils.model_provider"
        ),
        "slime.backends.megatron_utils.stateless_adam": types.ModuleType(
            "slime.backends.megatron_utils.stateless_adam"
        ),
        "slime.observability": types.ModuleType("slime.observability"),
        "slime.observability.logging_utils": types.ModuleType("slime.observability.logging_utils"),
        "slime.observability.train_metric_utils": types.ModuleType("slime.observability.train_metric_utils"),
        "slime.utils.memory_utils": types.ModuleType("slime.utils.memory_utils"),
    }

    modules["megatron.core"].mpu = modules["megatron.core.mpu"]
    modules["megatron.core.distributed"].DistributedDataParallel = torch.nn.Module
    modules["megatron.core.distributed"].finalize_model_grads = lambda *args, **kwargs: None
    modules["megatron.core.enums"].ModelType = types.SimpleNamespace(encoder_or_decoder="encoder_or_decoder")
    modules["megatron.core.models.gpt"].GPTModel = torch.nn.Module

    @dataclasses.dataclass
    class FakeOptimizerConfig:
        optimizer: str = "adam"

    modules["megatron.core.optimizer"].OptimizerConfig = FakeOptimizerConfig
    modules["megatron.core.optimizer"].get_megatron_optimizer = lambda *args, **kwargs: None
    modules["megatron.core.optimizer.optimizer"].MegatronOptimizer = object
    modules["megatron.core.optimizer_param_scheduler"].OptimizerParamScheduler = object
    modules["megatron.core.pipeline_parallel"].get_forward_backward_func = lambda: None
    modules["megatron.core.utils"].get_model_config = lambda model: None
    modules["megatron.core.utils"].unwrap_model = lambda model: model
    modules["megatron.training.global_vars"].get_args = lambda: None
    modules["megatron.training.training"].get_model = lambda *args, **kwargs: None
    modules["tqdm"].tqdm = lambda values, *args, **kwargs: values
    modules["slime.backends.megatron_utils.checkpoint"].load_checkpoint = lambda *args, **kwargs: None
    modules["slime.backends.megatron_utils.checkpoint"].save_checkpoint = lambda *args, **kwargs: None
    modules["slime.backends.megatron_utils.data"].DataIterator = object
    modules["slime.backends.megatron_utils.data"].get_batch = lambda *args, **kwargs: None
    modules["slime.backends.megatron_utils.loss"].ROLLOUT_TOP_P_TOKEN_KEYS = ()
    modules["slime.backends.megatron_utils.loss"].get_rollout_top_p_logprob_kwargs = lambda *args, **kwargs: {}
    modules["slime.backends.megatron_utils.loss"].loss_function = lambda *args, **kwargs: None
    modules["slime.backends.megatron_utils.model_provider"].get_model_provider_func = lambda *args, **kwargs: None
    modules["slime.backends.megatron_utils.stateless_adam"].StatelessAdam = object
    modules["slime.observability"].logging_utils = modules["slime.observability.logging_utils"]
    modules["slime.observability"].train_metric_utils = modules["slime.observability.train_metric_utils"]
    modules["slime.utils.memory_utils"].clear_memory = lambda: None

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_path = Path(__file__).resolve().parents[1] / "slime" / "backends" / "megatron_utils" / "model.py"
    module_name = "slime.backends.megatron_utils.test_megatron_model_sao_validation_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _freeze_critic_attention(monkeypatch):
    model_provider = _load_model_provider(monkeypatch)
    model = _AttentionModel()
    args = types.SimpleNamespace(
        only_train_params_name_list=None,
        freeze_params_name_list=None,
        freeze_indexer=False,
        sao_critic_freeze_attention=True,
    )
    model_provider.freeze_model_params(model, args, role="critic")
    return model


def _trainable_optimizer(model):
    return types.SimpleNamespace(
        param_groups=[{"params": [parameter for parameter in model.parameters() if parameter.requires_grad]}]
    )


@pytest.mark.unit
def test_sao_critic_optimizer_diagnostics_report_local_counts_and_exclude_frozen(monkeypatch, caplog):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    model_module = _load_model_module(monkeypatch)
    model = _freeze_critic_attention(monkeypatch)
    optimizer = _trainable_optimizer(model)

    expected_frozen_numel = sum(parameter.numel() for parameter in model._slime_sao_frozen_attention_params)
    expected_optimizer_numel = sum(parameter.numel() for parameter in optimizer.param_groups[0]["params"])

    with caplog.at_level(logging.INFO, logger=model_module.logger.name):
        frozen_numel, optimizer_numel = model_module.validate_sao_critic_optimizer_parameters(
            [model], optimizer, role="critic"
        )

    assert (frozen_numel, optimizer_numel) == (expected_frozen_numel, expected_optimizer_numel)
    assert "role=critic" in caplog.text
    assert "local_rank=0" in caplog.text
    assert f"frozen_attention_numel={expected_frozen_numel}" in caplog.text
    assert f"optimizer_numel={expected_optimizer_numel}" in caplog.text


@pytest.mark.unit
def test_sao_critic_optimizer_validation_rejects_frozen_parameter_membership(monkeypatch):
    model_module = _load_model_module(monkeypatch)
    model = _freeze_critic_attention(monkeypatch)
    optimizer = types.SimpleNamespace(param_groups=[{"params": list(model.parameters())}])

    with pytest.raises(RuntimeError, match="SAO-frozen attention parameter"):
        model_module.validate_sao_critic_optimizer_parameters([model], optimizer, role="critic")


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_name",
    [
        "layers.0.self_attention.q_layernorm.weight",
        "layers.0.mlp.weight",
        "output_layer.weight",
    ],
)
def test_sao_critic_optimizer_validation_rejects_unexpected_sao_freeze(monkeypatch, bad_name):
    model_module = _load_model_module(monkeypatch)
    model = _freeze_critic_attention(monkeypatch)
    bad_parameter = dict(model.named_parameters())[bad_name]
    model._slime_sao_frozen_attention_param_names = (
        *model._slime_sao_frozen_attention_param_names,
        bad_name,
    )
    model._slime_sao_frozen_attention_params = (*model._slime_sao_frozen_attention_params, bad_parameter)

    with pytest.raises(RuntimeError, match="normalization|MLP|value-output"):
        model_module.validate_sao_critic_optimizer_parameters([model], _trainable_optimizer(model), role="critic")


@pytest.mark.unit
def test_existing_global_freeze_is_not_reported_as_sao_violation(monkeypatch):
    model_module = _load_model_module(monkeypatch)
    model_provider = _load_model_provider(monkeypatch)
    model = _AttentionModel()
    args = types.SimpleNamespace(
        only_train_params_name_list=None,
        freeze_params_name_list=[r"\.mlp\."],
        freeze_indexer=False,
        sao_critic_freeze_attention=True,
    )
    model_provider.freeze_model_params(model, args, role="critic")

    model_module.validate_sao_critic_optimizer_parameters([model], _trainable_optimizer(model), role="critic")


@pytest.mark.unit
def test_setup_model_and_optimizer_runs_sao_validation_after_optimizer_creation(monkeypatch, caplog):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    model_module = _load_model_module(monkeypatch)
    model = _freeze_critic_attention(monkeypatch)
    optimizer = _trainable_optimizer(model)
    monkeypatch.setattr(model_module, "get_model", lambda *args, **kwargs: [model])
    monkeypatch.setattr(model_module, "get_megatron_optimizer", lambda **kwargs: optimizer)
    monkeypatch.setattr(model_module, "get_optimizer_param_scheduler", lambda args, optimizer: "scheduler")
    args = types.SimpleNamespace(
        moe_use_upcycling=False,
        load="/tmp/checkpoint",
        pretrained_checkpoint=None,
        num_rollout=1,
        use_stateless_adam=False,
        enable_gloo_process_groups=False,
        sao_critic_freeze_attention=True,
    )

    with caplog.at_level(logging.INFO, logger=model_module.logger.name):
        returned_model, returned_optimizer, scheduler = model_module.setup_model_and_optimizer(args, role="critic")

    assert returned_model == [model]
    assert returned_optimizer is optimizer
    assert scheduler == "scheduler"
    assert "SAO critic parameter diagnostics" in caplog.text
