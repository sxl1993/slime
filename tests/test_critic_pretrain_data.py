from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.coding_agent_rl.critic_pretrain.data import (
    CriticCandidate,
    assign_instance_split,
    iter_critic_records,
    normalize_orchard_row,
    select_instance_candidates,
    write_critic_artifact,
)


class FakeMaskGenerator:
    def __init__(self, tokens, mask):
        self.tokens = tokens
        self.mask = mask

    def get_loss_mask(self, messages, tools=None):
        return list(self.tokens), list(self.mask)


def orchard_row(*, outcome="resolved", instance_id="django__django-123", messages=None):
    return {
        "messages": messages
        or [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "issue"},
            {"role": "assistant", "content": "tool call"},
            {"role": "tool", "content": "observation"},
            {"role": "assistant", "content": "final"},
        ],
        "tools": [],
        "metadata": {
            "instance_id": instance_id,
            "sample_idx": 3,
            "verify_status": outcome,
            "repo": "django/django",
        },
    }


def test_instance_split_is_stable_and_instance_isolated():
    assert assign_instance_split("django__django-123", seed=17) == assign_instance_split("django__django-123", seed=17)


def test_instance_cap_prefers_two_examples_per_outcome():
    candidates = [
        CriticCandidate(row_index=i, record_id=f"r{i}", instance_id="task", outcome=outcome)
        for i, outcome in enumerate(["resolved"] * 4 + ["unresolved"] * 4)
    ]
    selected = select_instance_candidates(candidates, max_per_instance=4, seed=17)
    assert len(selected) == 4
    assert [item.outcome for item in selected].count("resolved") == 2
    assert [item.outcome for item in selected].count("unresolved") == 2


def test_normalize_orchard_row_preserves_first_action_suffix():
    full_tokens = [10, 11, 12, 13, 14, 15]
    full_mask = [0, 0, 0, 1, 1, 1]
    record, skip_reason = normalize_orchard_row(
        orchard_row(),
        mask_generator=FakeMaskGenerator(full_tokens, full_mask),
        max_seq_length=32,
    )

    assert skip_reason is None
    assert record is not None
    assert record.reward == 1.0
    assert len(record.tokens) - record.response_length == 3
    assert record.loss_mask == full_mask[3:]


@pytest.mark.parametrize(
    ("row", "mask", "max_seq_length", "reason"),
    [
        (orchard_row(outcome="unknown"), ([1], [1]), 32, "unknown_outcome"),
        (orchard_row(), ([1, 2], [0, 0]), 32, "no_action_tokens"),
        (orchard_row(), ([1, 2, 3], [0, 1, 1]), 2, "overlength"),
    ],
)
def test_normalize_orchard_row_reports_expected_skip_reasons(row, mask, max_seq_length, reason):
    record, skip_reason = normalize_orchard_row(
        row,
        mask_generator=FakeMaskGenerator(*mask),
        max_seq_length=max_seq_length,
    )
    assert record is None
    assert skip_reason == reason


def test_normalize_orchard_row_reports_template_error():
    class BrokenMaskGenerator:
        def get_loss_mask(self, messages, tools=None):
            raise ValueError("bad template")

    record, skip_reason = normalize_orchard_row(orchard_row(), mask_generator=BrokenMaskGenerator(), max_seq_length=32)
    assert record is None
    assert skip_reason == "template_error"


def test_write_and_iterate_balanced_artifact(tmp_path: Path):
    pytest.importorskip("pyarrow")
    records = []
    for instance_idx, split in enumerate(("train", "dev", "test")):
        del split
        for outcome in ("resolved", "unresolved"):
            row = orchard_row(outcome=outcome, instance_id=f"instance-{instance_idx}")
            record, reason = normalize_orchard_row(
                row,
                mask_generator=FakeMaskGenerator([1, 2, 3], [0, 1, 1]),
                max_seq_length=32,
            )
            assert reason is None
            records.append(record)

    class TinyDataset:
        def __iter__(self):
            return iter(
                [
                    {
                        "messages": orchard_row(outcome=record.outcome, instance_id=record.instance_id)["messages"],
                        "tools": [],
                        "metadata": {
                            "instance_id": record.instance_id,
                            "sample_idx": i,
                            "verify_status": record.outcome,
                        },
                    }
                    for i, record in enumerate(records)
                ]
            )

    artifact_dir = tmp_path / "artifact"
    manifest = write_critic_artifact(
        TinyDataset(),
        artifact_dir,
        tokenizer=object(),
        mask_generator=FakeMaskGenerator([1, 2, 3], [0, 1, 1]),
        dataset_revision="revision",
        shard_size=2,
        max_seq_length=32,
        canary_count=4,
    )
    assert manifest.schema_version == 1
    assert manifest.gamma == 1.0
    assert manifest.lambd == 1.0
    assert manifest.max_per_instance == 4
    assert manifest.canary_count == 4
    assert manifest.max_seq_length == 32
    assert (artifact_dir / "manifest.json").is_file()
    loaded = json.loads((artifact_dir / "manifest.json").read_text())
    assert loaded["schema_version"] == 1
    assert loaded["max_seq_length"] == 32
    rows = list(iter_critic_records(artifact_dir, "train", limit=4))
    assert len(rows) == 4
    assert rows[0].returns == [rows[0].reward] * rows[0].response_length
