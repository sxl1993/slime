from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.coding_agent_rl.critic_pretrain.train import (  # noqa: E402
    build_selection_payload,
    iter_record_batches,
    should_save_checkpoint,
    total_optimizer_steps,
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


def test_final_partial_batch_is_not_discarded():
    batches = list(iter_record_batches(list(range(5)), batch_size=2))
    assert batches == [[0, 1], [2, 3], [4]]
    assert total_optimizer_steps(4096, 128) == 32
    assert total_optimizer_steps(4097, 128) == 33
