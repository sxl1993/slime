from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[2] / "slime/agent/adapters/anthropic_stream.py"
_SPEC = importlib.util.spec_from_file_location("anthropic_stream_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

CumulativeTextTracker = _MODULE.CumulativeTextTracker
DeltaKind = _MODULE.DeltaKind
QwenAnthropicStreamParser = _MODULE.QwenAnthropicStreamParser


def _feed(parser: QwenAnthropicStreamParser, chunks: list[str], final_text: str):
    deltas = []
    for chunk in chunks:
        deltas.extend(parser.push(chunk))
    tail, snapshot = parser.finish(final_text)
    return deltas + tail, snapshot


def _two_chunk_cumulative_views(text: str):
    for cut in range(len(text) + 1):
        yield [text[:cut], text]


def test_cumulative_text_tracker_releases_only_confirmed_prefix():
    tracker = CumulativeTextTracker()
    assert tracker.push("caf�") == ""
    assert tracker.push("café") == "caf"
    assert tracker.finish("café") == "é"


@pytest.mark.parametrize(
    "raw,reasoning,text",
    [
        ("plain answer", "", "plain answer"),
        ("<think>inspect logs</think>fixed", "inspect logs", "fixed"),
        ("<think>inspect logs<tool_call>", "inspect logs", ""),
        ("<think>unfinished", "unfinished", ""),
    ],
)
def test_reasoning_and_text_are_independent_of_chunk_boundary(raw, reasoning, text):
    for chunks in _two_chunk_cumulative_views(raw):
        parser = QwenAnthropicStreamParser(tools_schema=None)
        deltas, snapshot = _feed(parser, chunks, raw)
        assert snapshot.reasoning == reasoning
        assert snapshot.text == text
        assert "".join(d.text for d in deltas if d.kind is DeltaKind.REASONING) == reasoning
        assert "".join(d.text for d in deltas if d.kind is DeltaKind.TEXT) == text
