from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
