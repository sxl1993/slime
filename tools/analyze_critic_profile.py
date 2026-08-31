#!/usr/bin/env python3
"""Analyze PyTorch profiler traces with the fast orjson parser.

Usage:
    python tools/analyze_critic_profile.py path/to/train_critic_rank_0.trace.json.gz
    python tools/analyze_critic_profile.py path/to/trace.json.gz --top-n 30 --json
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:
    import orjson as _orjson
except ModuleNotFoundError:  # Keep unit-test collection usable on minimal hosts.
    _orjson = None

GPU_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}
HOST_SYNC_NAMES = {
    "aten::item",
    "aten::_local_scalar_dense",
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
}


def _open_trace(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rb")
    return path.open("rb")


def _load_trace(path: Path) -> dict[str, Any]:
    if _orjson is None:
        raise RuntimeError("orjson is required to analyze profiler traces")
    with _open_trace(path) as stream:
        trace = _orjson.loads(stream.read())
    if not isinstance(trace, dict):
        raise ValueError(f"Trace root must be an object: {path}")
    if not isinstance(trace.get("traceEvents"), list):
        raise ValueError(f'Trace does not contain a "traceEvents" array: {path}')
    return trace


def _metadata(trace: dict[str, Any]) -> dict[str, Any]:
    distributed = trace.get("distributedInfo") or {}
    devices = trace.get("deviceProperties") or []
    first_device = devices[0] if devices and isinstance(devices[0], dict) else {}
    return {
        "schema_version": trace.get("schemaVersion"),
        "rank": distributed.get("rank"),
        "world_size": distributed.get("world_size"),
        "backend": distributed.get("backend"),
        "nccl_version": distributed.get("nccl_version"),
        "cuda_runtime_version": trace.get("cuda_runtime_version"),
        "device_name": first_device.get("name"),
        "visible_device_count": len(devices),
    }


def iter_trace_events(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield events from a gzip-compressed or plain trace parsed by orjson."""

    yield from (event for event in _load_trace(Path(path))["traceEvents"] if isinstance(event, dict))


def classify_kernel(name: str) -> str:
    lowered = name.lower()
    if "nccl" in lowered:
        return "NCCL communication"
    if "flash_attn" in lowered or "flashattn" in lowered:
        return "Flash attention"
    if any(
        token in lowered
        for token in (
            "gatedelta",
            "gated_delta",
            "chunk_gated",
            "causal_conv1d",
            "recompute_w_u",
            "prepare_wy_repr",
            "chunk_fwd",
            "chunk_bwd",
        )
    ):
        return "GDN/linear attention"
    if any(token in lowered for token in ("nvjet", "gemm", "cutlass", "matmul", "cublas")):
        return "GEMM/MatMul"
    if "norm" in lowered:
        return "Normalization"
    if any(token in lowered for token in ("memcpy", "memset", "copy")):
        return "Memory/copy"
    if any(token in lowered for token in ("softmax", "reduce")):
        return "Reduce/softmax"
    return "Other"


def _new_name_stat() -> dict[str, float | int]:
    return {"count": 0, "total_us": 0.0, "max_us": 0.0}


def _record_name_stat(
    stats: dict[tuple[str, str], dict[str, float | int]], category: str, name: str, dur: float
) -> None:
    item = stats.setdefault((category, name), _new_name_stat())
    item["count"] += 1
    item["total_us"] += dur
    item["max_us"] = max(item["max_us"], dur)


def _sorted_name_stats(
    stats: dict[tuple[str, str], dict[str, float | int]], category: str, top_n: int
) -> list[dict[str, Any]]:
    rows = []
    for (item_category, name), item in stats.items():
        if item_category != category:
            continue
        total_us = float(item["total_us"])
        count = int(item["count"])
        rows.append(
            {
                "name": name,
                "count": count,
                "total_us": total_us,
                "avg_us": total_us / count,
                "max_us": float(item["max_us"]),
            }
        )
    rows.sort(key=lambda row: row["total_us"], reverse=True)
    return rows[:top_n]


def analyze_trace_file(path: str | Path, top_n: int = 20) -> dict[str, Any]:
    """Return a JSON-serializable critic trace summary."""

    path = Path(path)
    if top_n < 1:
        raise ValueError("top_n must be positive")

    trace = _load_trace(path)
    name_stats: dict[tuple[str, str], dict[str, float | int]] = {}
    category_counts: Counter[str] = Counter()
    category_durations: defaultdict[str, float] = defaultdict(float)
    kernel_category_counts: Counter[str] = Counter()
    kernel_category_durations: defaultdict[str, float] = defaultdict(float)
    phase_counts: Counter[str] = Counter()
    phase_durations: defaultdict[str, float] = defaultdict(float)
    total_events = 0
    kernel_events = 0
    kernel_duration_us = 0.0
    trace_start_us: float | None = None
    trace_end_us: float | None = None

    for event in trace["traceEvents"]:
        if not isinstance(event, dict):
            continue
        total_events += 1
        category = str(event.get("cat", ""))
        phase = str(event.get("ph", ""))
        category_counts[category] += 1
        phase_counts[phase] += 1

        duration = event.get("dur")
        timestamp = event.get("ts")
        if not isinstance(duration, (int, float)) or not isinstance(timestamp, (int, float)):
            continue

        duration_us = float(duration)
        timestamp_us = float(timestamp)
        if phase == "X":
            category_durations[category] += duration_us
            phase_durations[phase] += duration_us
            trace_start_us = timestamp_us if trace_start_us is None else min(trace_start_us, timestamp_us)
            trace_end_us = (
                timestamp_us + duration_us if trace_end_us is None else max(trace_end_us, timestamp_us + duration_us)
            )
            name = str(event.get("name", ""))
            _record_name_stat(name_stats, category, name, duration_us)

            if category == "kernel":
                kernel_events += 1
                kernel_duration_us += duration_us
                kernel_category = classify_kernel(name)
                kernel_category_counts[kernel_category] += 1
                kernel_category_durations[kernel_category] += duration_us

    trace_wall_time_us = 0.0 if trace_start_us is None or trace_end_us is None else trace_end_us - trace_start_us
    return {
        "path": str(path),
        "metadata": _metadata(trace),
        "total_events": total_events,
        "phase_counts": dict(phase_counts),
        "category_counts": dict(category_counts),
        "category_durations_us": dict(category_durations),
        "trace_wall_time_us": trace_wall_time_us,
        "kernel_events": kernel_events,
        "kernel_duration_us": kernel_duration_us,
        "kernel_categories_us": dict(kernel_category_durations),
        "kernel_category_counts": dict(kernel_category_counts),
        "top_kernels": _sorted_name_stats(name_stats, "kernel", top_n),
        "top_cpu_ops": _sorted_name_stats(name_stats, "cpu_op", top_n),
        "top_cuda_runtime": _sorted_name_stats(name_stats, "cuda_runtime", top_n),
        "top_user_annotations": _sorted_name_stats(name_stats, "user_annotation", top_n),
        "host_sync": {
            row["name"]: {key: row[key] for key in ("count", "total_us", "avg_us", "max_us")}
            for category in ("cpu_op", "cuda_runtime")
            for row in _sorted_name_stats(name_stats, category, len(name_stats))
            if row["name"] in HOST_SYNC_NAMES
        },
    }


def _format_duration_us(duration_us: float) -> str:
    if duration_us >= 60 * 60 * 1_000_000:
        return f"{duration_us / (60 * 60 * 1_000_000):.2f} h"
    if duration_us >= 60 * 1_000_000:
        return f"{duration_us / (60 * 1_000_000):.2f} min"
    if duration_us >= 1_000_000:
        return f"{duration_us / 1_000_000:.2f} s"
    if duration_us >= 1_000:
        return f"{duration_us / 1_000:.2f} ms"
    return f"{duration_us:.2f} us"


def _print_rows(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{title}")
    if not rows:
        print("  (none)")
        return
    for row in rows:
        name = row["name"]
        if len(name) > 120:
            name = name[:117] + "..."
        print(
            f"  {_format_duration_us(row['total_us']):>12}  n={row['count']:>7}  "
            f"avg={_format_duration_us(row['avg_us']):>10}  max={_format_duration_us(row['max_us']):>10}  {name}"
        )


def render_text(summary: dict[str, Any]) -> str:
    metadata = summary["metadata"]
    lines = [
        f"Trace: {summary['path']}",
        f"Device: {metadata.get('device_name') or '?'} × {metadata.get('visible_device_count') or '?'}",
        f"Distributed: rank={metadata.get('rank')}, world_size={metadata.get('world_size')}, "
        f"backend={metadata.get('backend') or '?'}, NCCL={metadata.get('nccl_version') or '?' }",
        f"Events: {summary['total_events']:,}; wall span: {_format_duration_us(summary['trace_wall_time_us'])}",
        f"Kernel events: {summary['kernel_events']:,}; summed kernel duration: "
        f"{_format_duration_us(summary['kernel_duration_us'])}",
        "\nKernel categories (summed event duration; overlapping events are not deduplicated):",
    ]
    category_rows = sorted(summary["kernel_categories_us"].items(), key=lambda item: item[1], reverse=True)
    kernel_total = summary["kernel_duration_us"]
    for category, duration_us in category_rows:
        percentage = 0.0 if kernel_total == 0 else duration_us / kernel_total * 100
        count = summary["kernel_category_counts"].get(category, 0)
        lines.append(f"  {percentage:5.1f}%  {_format_duration_us(duration_us):>12}  n={count:>7}  {category}")

    sync = summary["host_sync"]
    lines.append("\nHost synchronization indicators (summed event duration; CPU events may be nested):")
    if sync:
        for name, item in sorted(sync.items(), key=lambda entry: entry[1]["total_us"], reverse=True):
            lines.append(f"  {_format_duration_us(item['total_us']):>12}  n={item['count']:>7}  {name}")
    else:
        lines.append("  (none)")

    for title, key in (
        ("Top GPU kernels by summed duration:", "top_kernels"),
        ("Top CPU ops by summed duration:", "top_cpu_ops"),
        ("Top CUDA runtime calls by summed duration:", "top_cuda_runtime"),
    ):
        lines.append(f"\n{title}")
        rows = summary[key]
        if not rows:
            lines.append("  (none)")
        else:
            for row in rows:
                name = row["name"]
                if len(name) > 120:
                    name = name[:117] + "..."
                lines.append(
                    f"  {_format_duration_us(row['total_us']):>12}  n={row['count']:>7}  "
                    f"avg={_format_duration_us(row['avg_us']):>10}  {name}"
                )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze a PyTorch critic profiler trace")
    parser.add_argument("trace", type=Path, help="Plain or gzip-compressed trace JSON file")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top names to report")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of the text report")
    args = parser.parse_args()

    try:
        summary = analyze_trace_file(args.trace, args.top_n)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(render_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
