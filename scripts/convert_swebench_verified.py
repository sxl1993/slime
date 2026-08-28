"""Convert a pinned SWE-bench Verified parquet to Slime swebench JSONL.

The parquet is the pinned, enriched SWE-bench Verified v5 artifact. Image tags
are deliberately kept out of the generated dataset: the runtime receives a
stable ``local/<instance_id>`` key and resolves it in code via
``ArcaImageResolver`` (slime/agent/sandbox.py), driven by
SLIME_AGENT_ARCA_IMAGE_REGISTRY / SLIME_AGENT_ARCA_IMAGE_TAG_SUFFIX.

Usage:
    python scripts/convert_swebench_verified.py \
        --input /path/to/test-00000-of-00001.parquet \
        --output /path/to/swe_verified_v5.jsonl \
        [--image-map datasets/arca-images.json]   # optional coverage check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from io import StringIO
from pathlib import Path
from typing import Any, TextIO

SWEBENCH_VERIFIED_SHA256 = "bb5b123d29ce70107cc0951cf444894241c570a11d76aec452332c65b01e06d8"
SWEBENCH_V5_FIELDS = (
    "instance_id",
    "repo",
    "version",
    "base_commit",
    "problem_statement",
    "hints_text",
    "test_patch",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "environment_setup_commit",
    "image",
    "eval_script",
    "log_parser",
    "eval_type",
)


def load_image_map(path: str | Path) -> dict[str, str]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load image map {path}: {type(exc).__name__}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"image map {path} must contain a JSON object")
    for instance_id, image in payload.items():
        if not isinstance(instance_id, str) or not isinstance(image, str) or not image.strip():
            raise ValueError(f"image map {path} has an invalid entry for {instance_id!r}")
    return payload


def _row_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, Mapping):
        return dict(row)
    if hasattr(row, "to_dict"):
        return dict(row.to_dict())
    raise TypeError(f"unsupported parquet row type: {type(row).__name__}")


def _validate_rows(rows: list[dict[str, Any]], image_map: Mapping[str, str] | None) -> None:
    row_ids = [row.get("instance_id") for row in rows]
    if any(not isinstance(instance_id, str) or not instance_id for instance_id in row_ids):
        raise ValueError("SWE-bench v5 rows must have a non-empty string instance_id")
    duplicates = sorted({instance_id for instance_id in row_ids if row_ids.count(instance_id) > 1})
    if duplicates:
        raise ValueError(f"duplicate instance_id values: {duplicates[:5]}")

    if image_map is not None:
        row_id_set = set(row_ids)
        image_id_set = set(image_map)
        missing_images = sorted(row_id_set - image_id_set)
        extra_images = sorted(image_id_set - row_id_set)
        if missing_images or extra_images:
            raise ValueError(
                "SWE-bench/image-map IDs do not match: "
                f"missing_images={missing_images[:5]}, extra_images={extra_images[:5]}"
            )

    for row in rows:
        missing_fields = [field for field in SWEBENCH_V5_FIELDS if field not in row]
        if missing_fields:
            raise ValueError(f"{row['instance_id']} is missing SWE-bench v5 fields: {missing_fields}")


def convert_rows(rows: Iterable[Any], image_map: Mapping[str, str] | None, output: TextIO) -> int:
    normalized_rows = [_row_dict(row) for row in rows]
    _validate_rows(normalized_rows, image_map)
    for row in normalized_rows:
        instance_id = row["instance_id"]
        remote_env_info = {field: row[field] for field in SWEBENCH_V5_FIELDS}
        remote_env_info["dataset_image"] = remote_env_info["image"]
        remote_env_info["image"] = f"local/{instance_id}"
        remote_env_info["workdir"] = "/testbed"
        obj = {
            "prompt": row["problem_statement"],
            "label": instance_id,
            "metadata": {"remote_env_info": remote_env_info},
        }
        output.write(json.dumps(obj, ensure_ascii=False) + "\n")
    return len(normalized_rows)


def convert(input_path: str | Path, output_path: str | Path, image_map: dict[str, str]) -> tuple[int, int]:
    import pandas as pd

    df = pd.read_parquet(input_path)
    buffer = StringIO()
    written = convert_rows((row for _, row in df.iterrows()), image_map, buffer)
    Path(output_path).write_text(buffer.getvalue(), encoding="utf-8")
    return written, 0


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert pinned SWE-bench Verified v5 parquet to Slime JSONL")
    parser.add_argument("--input", required=True, help="Path to the parquet file")
    parser.add_argument("--output", required=True, help="Path to the output JSONL file")
    parser.add_argument("--image-map", required=True, help="JSON map from instance ID to current ARCA image tag")
    parser.add_argument("--sha256", default=SWEBENCH_VERIFIED_SHA256, help="Expected input parquet SHA-256")
    args = parser.parse_args()

    actual_sha256 = sha256_file(args.input)
    if actual_sha256 != args.sha256:
        raise SystemExit(f"input parquet SHA-256 mismatch: expected {args.sha256}, got {actual_sha256}")
    image_map = load_image_map(args.image_map)
    written, skipped = convert(args.input, args.output, image_map)
    assert skipped == 0
    print(f"Converted {written} SWE-bench v5 instances → {args.output}")


if __name__ == "__main__":
    main()
