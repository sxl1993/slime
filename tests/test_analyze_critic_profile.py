import gzip
import importlib.util
import json
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "analyze_critic_profile.py"
_SPEC = importlib.util.spec_from_file_location("analyze_critic_profile", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

analyze_trace_file = _MODULE.analyze_trace_file
iter_trace_events = _MODULE.iter_trace_events


def _write_trace(path):
    trace = {
        "schemaVersion": 1,
        "distributedInfo": {"backend": "nccl", "rank": 0, "world_size": 64},
        "traceEvents": [
            {"ph": "X", "cat": "user_annotation", "name": "ProfilerStep#3", "ts": 0, "dur": 5_000},
            {"ph": "X", "cat": "kernel", "name": "ncclDevKernel_SendRecv", "ts": 100, "dur": 1_000},
            {
                "ph": "X",
                "cat": "kernel",
                "name": "flash_attn_fwd_kernel",
                "ts": 1_200,
                "dur": 200,
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "chunk_gated_delta_rule_fwd_kernel",
                "ts": 1_500,
                "dur": 300,
            },
            {"ph": "X", "cat": "cpu_op", "name": "aten::item", "ts": 1_000, "dur": 50},
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaStreamSynchronize", "ts": 1_100, "dur": 75},
        ],
    }
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(trace, stream)


def _patch_orjson(monkeypatch):
    class FakeOrjson:
        calls = 0

        @staticmethod
        def loads(payload):
            FakeOrjson.calls += 1
            return json.loads(payload)

    monkeypatch.setattr(_MODULE, "_orjson", FakeOrjson, raising=False)
    return FakeOrjson


def test_iter_trace_events_reads_gzip_trace(tmp_path, monkeypatch):
    trace_path = tmp_path / "critic.trace.json.gz"
    _write_trace(trace_path)
    _patch_orjson(monkeypatch)

    events = list(iter_trace_events(trace_path))

    assert len(events) == 6
    assert events[1]["name"] == "ncclDevKernel_SendRecv"


def test_iter_trace_events_uses_orjson(tmp_path, monkeypatch):
    trace_path = tmp_path / "critic.trace.json.gz"
    _write_trace(trace_path)
    FakeOrjson = _patch_orjson(monkeypatch)

    events = list(iter_trace_events(trace_path))

    assert len(events) == 6
    assert events[-1]["name"] == "cudaStreamSynchronize"
    assert FakeOrjson.calls == 1


def test_analyze_trace_reports_critic_kernel_and_host_sync_breakdown(tmp_path, monkeypatch):
    trace_path = tmp_path / "critic.trace.json.gz"
    _write_trace(trace_path)
    _patch_orjson(monkeypatch)

    summary = analyze_trace_file(trace_path, top_n=5)

    assert summary["total_events"] == 6
    assert summary["metadata"]["rank"] == 0
    assert summary["metadata"]["world_size"] == 64
    assert summary["trace_wall_time_us"] == 5_000
    assert summary["kernel_categories_us"]["NCCL communication"] == 1_000
    assert summary["kernel_categories_us"]["Flash attention"] == 200
    assert summary["kernel_categories_us"]["GDN/linear attention"] == 300
    assert summary["host_sync"]["aten::item"]["count"] == 1
    assert summary["host_sync"]["aten::item"]["total_us"] == 50
    assert summary["host_sync"]["cudaStreamSynchronize"]["total_us"] == 75
