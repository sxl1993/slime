from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

NUM_GPUS = 0

_MODULE_PATH = Path(__file__).resolve().parents[2] / "slime/agent/adapters/anthropic_stream.py"
_SPEC = importlib.util.spec_from_file_location("anthropic_stream_under_test", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

CumulativeTextTracker = _MODULE.CumulativeTextTracker
DeltaKind = _MODULE.DeltaKind
AnthropicStreamResponse = _MODULE.AnthropicStreamResponse
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


class _RecordingResponse:
    def __init__(self):
        self.body = bytearray()
        self.eof = False

    async def write(self, data: bytes) -> None:
        self.body.extend(data)

    async def write_eof(self) -> None:
        self.eof = True


def _stream_controller(parser_factory, response=None, logger=None):
    return AnthropicStreamResponse(
        response=response or _RecordingResponse(),
        parser_factory=parser_factory,
        input_tokens=1,
        model="model",
        logger=logger or logging.getLogger("test_anthropic_stream"),
        log_prefix="test",
        sid="sid",
    )


def test_cumulative_text_tracker_releases_only_confirmed_prefix():
    tracker = CumulativeTextTracker()
    assert tracker.push("caf�") == ""
    assert tracker.push("café") == "caf"
    assert tracker.finish("café") == "é"


def test_cumulative_text_tracker_rejects_revision_of_emitted_prefix():
    tracker = CumulativeTextTracker()
    tracker.push("abc")
    assert tracker.push("abcd") == "abc"
    with pytest.raises(StreamProtocolError, match="already emitted"):
        tracker.push("abXde")


def test_cumulative_text_tracker_scans_only_uncommitted_tail(monkeypatch):
    scan_starts = []
    original = CumulativeTextTracker._common_prefix_length

    def record_scan_start(left, right, start=0):
        scan_starts.append(start)
        return original(left, right, start)

    monkeypatch.setattr(CumulativeTextTracker, "_common_prefix_length", staticmethod(record_scan_start))
    tracker = CumulativeTextTracker()
    tracker.push("a" * 1024)
    tracker.push("a" * 1024 + "b")
    tracker.push("a" * 1024 + "bc")
    assert scan_starts[-1] == 1024


def test_implicit_qwen_reasoning_is_independent_of_chunk_boundary():
    raw = "inspect logs</think>fixed"
    for chunks in _two_chunk_cumulative_views(raw):
        parser = QwenAnthropicStreamParser(tools_schema=None, starts_in_reasoning=True)
        deltas, snapshot = _feed(parser, chunks, raw)
        assert snapshot.reasoning == "inspect logs"
        assert snapshot.text == "fixed"
        assert "".join(d.text for d in deltas if d.kind is DeltaKind.REASONING) == "inspect logs"
        assert "".join(d.text for d in deltas if d.kind is DeltaKind.TEXT) == "fixed"


def test_prompt_prefilled_reasoning_uses_explicit_initial_state():
    raw = "unfinished reasoning"
    parser = QwenAnthropicStreamParser(
        tools_schema=None,
        starts_in_reasoning=True,
    )

    deltas, snapshot = _feed(parser, [raw], raw)

    assert snapshot.reasoning == raw
    assert snapshot.text == ""
    assert "".join(delta.text for delta in deltas if delta.kind is DeltaKind.REASONING) == raw


@pytest.mark.parametrize(
    "raw,reasoning,text",
    [
        ("plain answer", "", "plain answer"),
        ("\n<think>\ninspect</think>fixed", "\ninspect", "fixed"),
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


def test_incremental_parser_consumes_text_deltas_without_cumulative_views():
    raw = "<think>inspect logs</think>fixed"
    parser = QwenAnthropicStreamParser(tools_schema=None)
    deltas = []
    for character in raw:
        deltas.extend(parser.push_delta(character))
    tail, snapshot = parser.finish_delta()
    deltas.extend(tail)

    assert snapshot.reasoning == "inspect logs"
    assert snapshot.text == "fixed"
    assert "".join(delta.text for delta in deltas if delta.kind is DeltaKind.REASONING) == "inspect logs"
    assert "".join(delta.text for delta in deltas if delta.kind is DeltaKind.TEXT) == "fixed"


@pytest.mark.parametrize(
    "raw",
    [
        "reasoning only",
        "reasoning</think>answer",
        ("reasoning<tool_call><function=Read>" "<parameter=file_path>a</parameter></function></tool_call>"),
    ],
)
def test_incremental_and_complete_parse_share_one_semantics(raw):
    def make_parser():
        return QwenAnthropicStreamParser(
            tools_schema=TOOLS,
            starts_in_reasoning=True,
        )

    canonical = make_parser()
    _, canonical_snapshot = canonical.finish(raw)
    for cut in range(len(raw) + 1):
        incremental = make_parser()
        incremental.push_delta(raw[:cut])
        incremental.push_delta(raw[cut:])
        _, incremental_snapshot = incremental.finish_delta()
        _MODULE.validate_snapshot_parity(incremental_snapshot, canonical_snapshot)


def test_snapshot_parity_rejects_semantic_drift_without_logging_content():
    incremental = _MODULE.StreamSnapshot(
        reasoning="DISTINCTIVE_REASONING_CONTENT",
        text="",
        tool_uses=(),
    )
    canonical = _MODULE.StreamSnapshot(
        reasoning="",
        text="DISTINCTIVE_TEXT_CONTENT",
        tool_uses=(),
    )

    with pytest.raises(_MODULE.StreamParityError) as error:
        _MODULE.validate_snapshot_parity(incremental, canonical)

    message = str(error.value)
    assert "incremental snapshot differs from complete snapshot" in message
    assert "DISTINCTIVE_REASONING_CONTENT" not in message
    assert "DISTINCTIVE_TEXT_CONTENT" not in message


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


@pytest.mark.parametrize("terminal_marker", ["<|im_end|>", "<|endoftext|>"])
def test_terminal_model_marker_after_tool_ends_the_stream(terminal_marker):
    raw = "<tool_call><function=Read><parameter=file_path>a</parameter></function></tool_call>" + terminal_marker
    for chunks in _two_chunk_cumulative_views(raw):
        parser = QwenAnthropicStreamParser(tools_schema=TOOLS)
        deltas, snapshot = _feed(parser, chunks, raw)
        assert snapshot.text == ""
        assert snapshot.tool_uses == ({"name": "Read", "input": {"file_path": "a"}},)
        assert [delta.kind for delta in deltas].count(DeltaKind.TOOL_STOP) == 1


def test_real_text_after_terminal_model_marker_still_fails():
    raw = "<tool_call><function=Read><parameter=file_path>a</parameter></function></tool_call>" "<|im_end|>unexpected"
    parser = QwenAnthropicStreamParser(tools_schema=TOOLS)
    with pytest.raises(StreamProtocolError, match="text after a completed tool"):
        _feed(parser, [raw], raw)


def test_terminal_model_marker_after_text_is_preserved():
    raw = "answer" + "<|im_end|>"
    for chunks in _two_chunk_cumulative_views(raw):
        parser = QwenAnthropicStreamParser(tools_schema=None)
        _, snapshot = _feed(parser, chunks, raw)
        assert snapshot.text == raw


def test_invalid_boolean_parameter_fails_instead_of_becoming_false():
    raw = "<tool_call><function=Write>" "<parameter=overwrite>definitely</parameter>" "</function></tool_call>"
    parser = QwenAnthropicStreamParser(tools_schema=TOOLS)
    with pytest.raises(StreamProtocolError, match="invalid boolean"):
        _feed(parser, [raw], raw)


def test_parser_can_record_a_terminal_failure_state():
    parser = QwenAnthropicStreamParser(tools_schema=None)
    parser.push("partial")
    parser.mark_failed()
    assert parser.state == "failed"


def test_unknown_tool_fails_before_tool_start_is_emitted():
    raw = "<tool_call><function=Unknown></function></tool_call>"
    parser = QwenAnthropicStreamParser(tools_schema=TOOLS)
    with pytest.raises(StreamProtocolError, match="unknown tool"):
        _feed(parser, [raw], raw)


def test_truncated_parameter_fails_at_finish():
    raw = "<tool_call><function=Write><parameter=content>unfinished"
    parser = QwenAnthropicStreamParser(tools_schema=TOOLS)
    parser.push(raw)
    with pytest.raises(StreamProtocolError, match="parameter") as caught:
        parser.finish(raw)
    assert caught.value.code == "incomplete_tool_parameter"


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


def test_deferred_tool_can_be_canonicalized_when_missing_from_request_schema():
    raw = "<tool_call><function=Grep><parameter=pattern>needle</parameter></function></tool_call>"
    parser = QwenAnthropicStreamParser(
        tools_schema=None,
        canonicalize_tool=lambda tool: {"name": "Bash", "input": {"command": f"grep {tool['input']['pattern']}"}},
        deferred_tool_names=frozenset({"Grep"}),
    )
    deltas, snapshot = _feed(parser, [raw], raw)
    assert snapshot.tool_uses == (({"name": "Bash", "input": {"command": "grep needle"}}),)
    assert [delta.tool_name for delta in deltas if delta.kind is DeltaKind.TOOL_START] == ["Bash"]


def test_streamed_blocks_match_canonical_blocks_semantically():
    streamed = [
        {"type": "thinking", "thinking": "inspect"},
        {"type": "text", "text": "done"},
        {"type": "tool_use", "id": "toolu_same", "name": "Write", "input": {"b": 2, "a": 1}},
    ]
    canonical = [
        {"type": "thinking", "thinking": "inspect"},
        {"type": "text", "text": "done"},
        {"type": "tool_use", "id": "toolu_same", "name": "Write", "input": {"a": 1, "b": 2}},
    ]
    _MODULE.validate_block_parity(streamed, canonical)


def test_stream_response_reuses_emitted_tool_id_for_canonical_blocks(monkeypatch):
    raw = "<tool_call><function=Read><parameter=file_path>a</parameter></function></tool_call>"
    monkeypatch.setattr(_MODULE.secrets, "token_hex", lambda _: "stream")

    async def run_case():
        response = _RecordingResponse()
        controller = _stream_controller(
            lambda: QwenAnthropicStreamParser(tools_schema=TOOLS),
            response=response,
        )
        await controller.on_progress(
            SimpleNamespace(
                rid="rid",
                chunk_index=1,
                text=raw,
                output_token_logprobs=((-0.1, 1),),
                finish_reason="length",
                incremental=False,
                output_token_count=1,
            )
        )
        reply = SimpleNamespace(
            wire=(
                [{"type": "tool_use", "id": "toolu_canonical", "name": "Read", "input": {"file_path": "a"}}],
                "tool_use",
            )
        )
        await controller.finish(
            raw,
            1,
            1,
            lambda reasoning, text, tool_uses: (SimpleNamespace(), reply),
        )
        return response

    response = asyncio.run(run_case())
    assert response.eof
    assert b'"id":"toolu_stream"' in response.body


@pytest.mark.parametrize("incremental", [True, False])
def test_stream_response_drops_incomplete_tool_on_length(incremental):
    raw = "<tool_call><function=Write><parameter=content>unfinished"

    async def run_case():
        response = _RecordingResponse()
        controller = _stream_controller(
            lambda: QwenAnthropicStreamParser(tools_schema=TOOLS),
            response=response,
        )
        await controller.on_progress(
            SimpleNamespace(
                rid="rid",
                chunk_index=1,
                text=raw,
                text_delta=raw,
                output_token_logprobs=((-0.1, 1),),
                finish_reason="length",
                incremental=incremental,
                output_token_count=1,
            )
        )
        reply = SimpleNamespace(wire=([{"type": "text", "text": ""}], "max_tokens"))
        await controller.finish(
            raw,
            1,
            1,
            lambda reasoning, text, tool_uses: (SimpleNamespace(), reply),
        )
        return response

    response = asyncio.run(run_case())

    assert response.eof
    assert b'"type":"tool_use"' not in response.body
    assert b'"stop_reason":"max_tokens"' in response.body


def test_stream_response_drops_all_tools_when_later_tool_is_incomplete():
    raw = (
        "<tool_call><function=Read><parameter=file_path>a</parameter></function></tool_call>"
        "<tool_call><function=Write><parameter=content>unfinished"
    )

    async def run_case():
        response = _RecordingResponse()
        controller = _stream_controller(
            lambda: QwenAnthropicStreamParser(tools_schema=TOOLS),
            response=response,
        )
        await controller.on_progress(
            SimpleNamespace(
                rid="rid",
                chunk_index=1,
                text_delta=raw,
                output_token_logprobs=((-0.1, 1),),
                finish_reason="length",
                incremental=True,
                output_token_count=1,
            )
        )

        def result_factory(reasoning, text, tool_uses):
            if tool_uses:
                blocks = [{"type": "tool_use", "name": tool["name"], "input": tool["input"]} for tool in tool_uses]
                stop_reason = "tool_use"
            else:
                blocks = [{"type": "text", "text": ""}]
                stop_reason = "max_tokens"
            return SimpleNamespace(), SimpleNamespace(wire=(blocks, stop_reason))

        await controller.finish(raw, 1, 1, result_factory)
        return response

    response = asyncio.run(run_case())

    assert response.eof
    assert b'"type":"tool_use"' not in response.body
    assert b'"stop_reason":"max_tokens"' in response.body


def test_stream_response_reparses_full_output_with_fresh_parser():
    parser_count = 0

    def make_parser():
        nonlocal parser_count
        parser_count += 1
        return QwenAnthropicStreamParser(tools_schema=None)

    async def run_case():
        response = _RecordingResponse()
        controller = AnthropicStreamResponse(
            response=response,
            parser_factory=make_parser,
            input_tokens=1,
            model="model",
            logger=logging.getLogger("test_anthropic_stream"),
            log_prefix="test",
            sid="sid",
        )
        await controller.on_progress(
            SimpleNamespace(
                rid="rid",
                chunk_index=1,
                text="done",
                output_token_logprobs=((-0.1, 1),),
                incremental=False,
                output_token_count=1,
            )
        )
        reply = SimpleNamespace(wire=([{"type": "text", "text": "done"}], "end_turn"))
        await controller.finish(
            "done",
            1,
            1,
            lambda reasoning, text, tool_uses: (SimpleNamespace(), reply),
        )

    asyncio.run(run_case())
    assert parser_count == 2


def test_stream_failure_summary_is_sanitized_and_reports_progress(monkeypatch, caplog):
    class ContextWindowExceeded(Exception):
        pass

    monkeypatch.setitem(
        sys.modules,
        "slime.agent.adapters.common",
        SimpleNamespace(ContextWindowExceeded=ContextWindowExceeded),
    )
    logger = logging.getLogger("test_anthropic_stream.failure")
    caplog.set_level(logging.DEBUG, logger=logger.name)

    async def run_case():
        controller = _stream_controller(
            lambda: QwenAnthropicStreamParser(tools_schema=None),
            logger=logger,
        )
        await controller.on_progress(
            SimpleNamespace(
                rid="rid",
                chunk_index=1,
                text="partial",
                output_token_logprobs=((-0.1, 1), (-0.2, 2)),
                incremental=False,
                output_token_count=2,
            )
        )
        await controller.fail(RuntimeError("DISTINCTIVE_SECRET_OUTPUT"))

    asyncio.run(run_case())
    records = [record for record in caplog.records if "event=anthropic_stream_summary" in record.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    assert "output_tokens=2" in records[0].getMessage()
    assert "parser_state=preamble" in records[0].getMessage()
    assert "DISTINCTIVE_SECRET_OUTPUT" not in records[0].getMessage()


def test_incomplete_tool_failure_reports_safe_structured_cause(monkeypatch, caplog):
    class ContextWindowExceeded(Exception):
        pass

    monkeypatch.setitem(
        sys.modules,
        "slime.agent.adapters.common",
        SimpleNamespace(ContextWindowExceeded=ContextWindowExceeded),
    )
    logger = logging.getLogger("test_anthropic_stream.structured_failure")
    caplog.set_level(logging.DEBUG, logger=logger.name)
    failures = []
    raw = "<tool_call><function=Write><parameter=content>DISTINCTIVE_SECRET_OUTPUT<|im_end|>"

    async def run_case():
        response = _RecordingResponse()
        controller = AnthropicStreamResponse(
            response=response,
            parser_factory=lambda: QwenAnthropicStreamParser(tools_schema=TOOLS),
            input_tokens=1,
            model="model",
            logger=logger,
            log_prefix="test",
            sid="sid",
            failure_reporter=lambda family, code: failures.append((family, code)),
        )
        await controller.on_progress(
            SimpleNamespace(
                rid="rid",
                chunk_index=1,
                text=raw,
                text_delta=raw,
                output_token_logprobs=((-0.1, 248046),),
                finish_reason="stop",
                finish_matched_token_id=248046,
                incremental=True,
                output_token_count=1,
            )
        )
        try:
            await controller.finish(raw, 1, 1, lambda *_: None)
        except StreamProtocolError as error:
            await controller.fail(error)

    asyncio.run(run_case())

    assert failures == [("stream_protocol_error", "incomplete_tool_parameter")]
    record = next(record for record in caplog.records if "event=stream_parse_failed" in record.getMessage())
    message = record.getMessage()
    assert "failure_family=stream_protocol_error" in message
    assert "failure_code=incomplete_tool_parameter" in message
    assert "finish_matched_token_id=248046" in message
    assert "DISTINCTIVE_SECRET_OUTPUT" not in message


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
            [{"type": "tool_use", "id": "toolu_stream", "name": "Read", "input": {}}],
            [{"type": "tool_use", "id": "toolu_final", "name": "Read", "input": {}}],
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
