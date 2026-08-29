from __future__ import annotations

import importlib.util
import json
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
StreamProtocolError = _MODULE.StreamProtocolError


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Write",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                    "overwrite": {"type": "boolean"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Read",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
            },
        },
    },
]


def _feed(parser: QwenAnthropicStreamParser, chunks: list[str], final_text: str):
    deltas = []
    for chunk in chunks:
        deltas.extend(parser.push(chunk))
    tail, snapshot = parser.finish(final_text)
    return deltas + tail, snapshot


def _two_chunk_cumulative_views(text: str):
    for cut in range(len(text) + 1):
        yield [text[:cut], text]


def _tool_input_json(deltas, tool_index):
    return "".join(
        delta.text for delta in deltas if delta.kind is DeltaKind.TOOL_INPUT and delta.tool_index == tool_index
    )


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


def test_qwen3_coder_tool_call_is_independent_of_chunk_boundary():
    raw = (
        "<tool_call><function=Write>"
        "<parameter=file_path>/tmp/a.py</parameter>"
        '<parameter=content>print("ok")\n</parameter>'
        "<parameter=overwrite>true</parameter>"
        "</function></tool_call>"
    )
    expected = {
        "file_path": "/tmp/a.py",
        "content": 'print("ok")\n',
        "overwrite": True,
    }
    for chunks in _two_chunk_cumulative_views(raw):
        parser = QwenAnthropicStreamParser(tools_schema=TOOLS)
        deltas, snapshot = _feed(parser, chunks, raw)
        assert snapshot.tool_uses == ({"name": "Write", "input": expected},)
        assert json.loads(_tool_input_json(deltas, 0)) == expected


def test_long_string_argument_emits_before_parameter_closes():
    prefix = "<tool_call><function=Write><parameter=content>"
    parser = QwenAnthropicStreamParser(tools_schema=TOOLS)
    parser.push(prefix)
    first = parser.push(prefix + "a" * 128)
    second = parser.push(prefix + "a" * 256)
    assert any(delta.kind is DeltaKind.TOOL_INPUT and delta.text for delta in first + second)


def test_consecutive_tools_do_not_emit_separator_whitespace_as_text():
    raw = (
        "<tool_call><function=Read><parameter=file_path>a</parameter></function></tool_call>\n"
        "<tool_call><function=Read><parameter=file_path>b</parameter></function></tool_call>"
    )
    parser = QwenAnthropicStreamParser(tools_schema=TOOLS)
    deltas, snapshot = _feed(parser, [raw[: len(raw) // 2], raw], raw)
    assert snapshot.text == ""
    assert [tool["input"]["file_path"] for tool in snapshot.tool_uses] == ["a", "b"]
    assert [delta.kind for delta in deltas].count(DeltaKind.TOOL_STOP) == 2


def test_unknown_tool_fails_before_tool_start_is_emitted():
    raw = "<tool_call><function=Unknown></function></tool_call>"
    parser = QwenAnthropicStreamParser(tools_schema=TOOLS)
    with pytest.raises(StreamProtocolError, match="unknown tool"):
        _feed(parser, [raw], raw)


def test_truncated_parameter_fails_at_finish():
    raw = "<tool_call><function=Write><parameter=content>unfinished"
    parser = QwenAnthropicStreamParser(tools_schema=TOOLS)
    parser.push(raw)
    with pytest.raises(StreamProtocolError, match="parameter"):
        parser.finish(raw)


def test_deferred_tool_emits_only_its_canonical_form():
    grep_tool = {
        "type": "function",
        "function": {
            "name": "Grep",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
            },
        },
    }
    raw = "<tool_call><function=Grep><parameter=pattern>needle</parameter></function></tool_call>"
    parser = QwenAnthropicStreamParser(
        tools_schema=[grep_tool],
        canonicalize_tool=lambda _: {"name": "Bash", "input": {"command": "grep needle"}},
        deferred_tool_names=frozenset({"Grep"}),
    )
    deltas, snapshot = _feed(parser, [raw], raw)
    assert snapshot.tool_uses == ({"name": "Bash", "input": {"command": "grep needle"}},)
    assert [delta.tool_name for delta in deltas if delta.kind is DeltaKind.TOOL_START] == ["Bash"]
    assert json.loads(_tool_input_json(deltas, 0)) == {"command": "grep needle"}


def test_streamed_blocks_match_canonical_blocks_semantically():
    streamed = [
        {"type": "thinking", "thinking": "inspect"},
        {"type": "text", "text": "done"},
        {"type": "tool_use", "id": "toolu_stream", "name": "Write", "input": {"b": 2, "a": 1}},
    ]
    canonical = [
        {"type": "thinking", "thinking": "inspect"},
        {"type": "text", "text": "done"},
        {"type": "tool_use", "id": "toolu_final", "name": "Write", "input": {"a": 1, "b": 2}},
    ]
    _MODULE.validate_block_parity(streamed, canonical)


@pytest.mark.parametrize(
    "streamed,canonical",
    [
        (
            [{"type": "text", "text": "streamed"}],
            [{"type": "text", "text": "canonical"}],
        ),
        (
            [{"type": "tool_use", "name": "Read", "input": {}}],
            [{"type": "tool_use", "name": "Write", "input": {}}],
        ),
        (
            [
                {"type": "tool_use", "name": "Read", "input": {"path": "a"}},
                {"type": "tool_use", "name": "Write", "input": {"path": "b"}},
            ],
            [
                {"type": "tool_use", "name": "Write", "input": {"path": "b"}},
                {"type": "tool_use", "name": "Read", "input": {"path": "a"}},
            ],
        ),
    ],
)
def test_streamed_block_semantic_mismatch_fails(streamed, canonical):
    with pytest.raises(_MODULE.StreamParityError, match="streamed blocks differ"):
        _MODULE.validate_block_parity(streamed, canonical)
