from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.coding_agent_rl.critic_pretrain.train import (  # noqa: E402
    add_critic_pretrain_arguments,
    build_selection_payload,
    configure_critic_pretrain_schedule,
    initial_training_step,
    iter_record_batches,
    selected_record_count,
    should_save_checkpoint,
    should_save_training_checkpoint,
    total_optimizer_steps,
    validate_artifact_context,
    validate_canary_metrics,
    validate_gradient_states,
    write_selection_json,
)


def test_checkpoint_selection_saves_only_dev_improvements(tmp_path: Path):
    best = float("inf")
    saved = []
    for iteration, value in ((100, 0.24), (200, 0.19), (300, 0.21)):
        if should_save_checkpoint(value, best):
            saved.append(iteration)
            best = value
    assert saved == [100, 200]
    payload = build_selection_payload(
        best_iteration=200,
        best_value=0.19,
        global_batch_size=128,
        train_limit=4096,
    )
    assert payload == {
        "schema_version": 1,
        "selection_metric": "dev/trajectory_equal_mse",
        "best_iteration": 200,
        "best_value": 0.19,
        "global_batch_size": 128,
        "train_limit": 4096,
    }
    path = tmp_path / "selection.json"
    write_selection_json(path, payload)
    assert json.loads(path.read_text()) == payload


def test_final_partial_batch_is_discarded():
    batches = list(iter_record_batches(list(range(5)), batch_size=2))
    assert batches == [[0, 1], [2, 3]]
    assert total_optimizer_steps(4096, 128) == 32
    assert total_optimizer_steps(4097, 128) == 32


def test_critic_eval_limit_defaults_to_balanced_512_records():
    parser = add_critic_pretrain_arguments(argparse.ArgumentParser())
    args = parser.parse_args(["--critic-pretrain-data", "data", "--critic-pretrain-selection-json", "selection.json"])
    assert args.critic_pretrain_eval_limit == 512


def test_resume_training_starts_after_loaded_iteration():
    assert initial_training_step([1, 1]) == 0
    assert initial_training_step([21, 21]) == 20
    with pytest.raises(ValueError, match="different checkpoint iterations"):
        initial_training_step([21, 22])


def test_training_checkpoint_saves_on_interval_and_final_step():
    assert should_save_training_checkpoint(50, 409, 50)
    assert not should_save_training_checkpoint(51, 409, 50)
    assert should_save_training_checkpoint(409, 409, 50)
    with pytest.raises(ValueError, match="positive"):
        should_save_training_checkpoint(1, 2, 0)


def test_selected_record_count_includes_both_balanced_outcomes():
    manifest = {"splits": {"train": {"selected_resolved": 4096, "selected_unresolved": 4096}}}
    assert selected_record_count(manifest, "train") == 8192


def test_selected_record_count_rejects_unbalanced_artifact():
    manifest = {"splits": {"train": {"selected_resolved": 2, "selected_unresolved": 1}}}
    with pytest.raises(ValueError, match="outcome-balanced"):
        selected_record_count(manifest, "train")


def test_critic_pretrain_schedule_uses_optimizer_step_count():
    args = SimpleNamespace(num_rollout=0, rollout_batch_size=128, n_samples_per_prompt=1, global_batch_size=128)
    configure_critic_pretrain_schedule(args, train_limit=4096)
    assert args.num_rollout == 32


def test_artifact_context_must_match_model_context():
    validate_artifact_context({"max_seq_length": 98_304}, 98_304)
    with pytest.raises(ValueError, match="does not match"):
        validate_artifact_context({"max_seq_length": 98_304}, 4096)


def test_canary_rejects_missing_gradient_or_value_separation():
    with pytest.raises(ValueError, match="finite gradient"):
        validate_gradient_states([{"grad_norm": float("nan")}])
    with pytest.raises(ValueError, match="value separation"):
        validate_canary_metrics({"trajectory_equal_mse": 0.2, "resolved_mean": 0.4, "unresolved_mean": 0.5})


def test_canary_accepts_finite_gradient_and_separated_values():
    validate_gradient_states([{"grad_norm": 1.0}, {"grad_norm": 2.0}])
    validate_canary_metrics({"trajectory_equal_mse": 0.2, "resolved_mean": 0.6, "unresolved_mean": 0.4})
