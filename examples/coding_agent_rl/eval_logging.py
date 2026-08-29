"""Persist per-task and aggregate records for standalone agent evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _sample_record(dataset: str, rollout_id: int, sample: Any, checkpoint: str) -> dict[str, Any]:
    metadata = sample.metadata or {}
    reward = float(sample.reward)
    invalid_reason = metadata.get("invalid_reason")
    applied_cleanly = metadata.get("applied_cleanly")
    indeterminate = invalid_reason not in {None, "evaluation_only"} or applied_cleanly is False
    return {
        "dataset": dataset,
        "rollout_id": rollout_id,
        "checkpoint": checkpoint,
        "instance_id": metadata.get("instance_id") or sample.label,
        "reward": reward,
        "outcome": "indeterminate" if indeterminate else ("pass" if reward == 1.0 else "fail"),
        "status": sample.status.value,
        "applied_cleanly": applied_cleanly,
        "agent_exit_code": metadata.get("agent_exit_code"),
        "invalid_reason": invalid_reason,
        "evaluation": {"protocol": "swebench", "metric": "pass@1", "n_samples_per_prompt": 1},
    }


def _aggregate(dataset: str, rollout_id: int, records: list[dict[str, Any]], checkpoint: str) -> dict[str, Any]:
    passed = sum(record["outcome"] == "pass" for record in records)
    failed = sum(record["outcome"] == "fail" for record in records)
    indeterminate = sum(record["outcome"] == "indeterminate" for record in records)
    determinate = passed + failed
    return {
        "dataset": dataset,
        "rollout_id": rollout_id,
        "checkpoint": checkpoint,
        "count": len(records),
        "determinate": determinate,
        "indeterminate": indeterminate,
        "passed": passed,
        "failed": failed,
        "pass_at_1": passed / determinate if determinate else None,
    }


def log_swebench_eval(rollout_id: int, _args: Any, data: dict[str, Any], _extra_metrics: dict[str, Any]) -> bool:
    """Write one JSONL and one aggregate JSON file per evaluation dataset."""
    output_dir = os.environ.get("SWE_EVAL_OUTPUT_DIR", "").strip()
    if not output_dir:
        return False

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = os.environ.get("SWE_EVAL_CHECKPOINT", "").strip()
    for dataset, payload in data.items():
        records = [_sample_record(dataset, rollout_id, sample, checkpoint) for sample in payload.get("samples", [])]
        (root / f"{dataset}.jsonl").write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
            encoding="utf-8",
        )
        (root / f"{dataset}.aggregate.json").write_text(
            json.dumps(_aggregate(dataset, rollout_id, records, checkpoint), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return True
