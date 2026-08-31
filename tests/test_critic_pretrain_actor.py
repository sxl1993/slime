from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import examples.coding_agent_rl.critic_pretrain.actor as actor_module  # noqa: E402
from examples.coding_agent_rl.critic_pretrain.actor import (  # noqa: E402
    CriticPretrainRayActor,
    initialize_prior_centered_value_head,
    should_initialize_value_head,
)


def test_prior_centered_value_head_initialization():
    model = [type("Chunk", (), {"output_layer": torch.nn.Linear(4, 1)})()]
    initialize_prior_centered_value_head(model, optimizer=None)
    output_layer = model[0].output_layer
    torch.testing.assert_close(output_layer.weight, torch.zeros_like(output_layer.weight))
    torch.testing.assert_close(output_layer.bias, torch.full_like(output_layer.bias, 0.5))


def test_head_initialization_only_applies_to_hf_directory(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}")
    assert should_initialize_value_head(tmp_path)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("200\n")
    assert not should_initialize_value_head(tmp_path)


def test_critic_actor_rejects_non_critic_role():
    actor = object.__new__(CriticPretrainRayActor)
    try:
        actor.init(object(), role="actor")
    except ValueError as exc:
        assert "role=critic" in str(exc)
    else:
        raise AssertionError("critic-only actor accepted actor role")


def test_critic_evaluation_restores_context_parallel_values(monkeypatch):
    cp_module = types.ModuleType("slime.backends.megatron_utils.cp_utils")
    cp_module.all_gather_with_cp = lambda value, _total_length, response_length: torch.full(
        (response_length,), 0.5, dtype=value.dtype
    )
    monkeypatch.setitem(sys.modules, "slime.backends.megatron_utils.cp_utils", cp_module)
    monkeypatch.setattr(
        actor_module,
        "mpu",
        types.SimpleNamespace(
            is_pipeline_last_stage=lambda: True,
            get_tensor_model_parallel_rank=lambda: 0,
            get_context_parallel_rank=lambda: 0,
            get_context_parallel_world_size=lambda: 2,
        ),
    )
    monkeypatch.setattr(actor_module, "get_data_iterator", lambda rollout_data: rollout_data, raising=False)
    monkeypatch.setattr(
        actor_module,
        "forward_only",
        lambda *_args, **_kwargs: {"values": [torch.zeros(6380)]},
        raising=False,
    )
    monkeypatch.setattr(actor_module, "get_values", object(), raising=False)

    actor = object.__new__(CriticPretrainRayActor)
    actor.args = object()
    actor.model = object()
    actor._get_rollout_data = lambda _ref: {
        "num_microbatches": [1],
        "returns": [torch.ones(11178)],
        "loss_masks": [torch.ones(11178)],
        "total_lengths": [12000],
        "response_lengths": [11178],
    }

    metrics = actor.evaluate_critic(object())

    assert metrics["trajectory_count"] == 1
    assert metrics["squared_error_sum"] == pytest.approx(0.25)
