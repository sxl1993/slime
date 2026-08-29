"""Incremental Qwen output parsing for Anthropic streaming responses."""

from __future__ import annotations

import dataclasses
import enum
import json
import logging
import secrets
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from aiohttp import web

if TYPE_CHECKING:
    from slime.agent.adapters.common import GenerationProgress, Reply


class StreamProtocolError(RuntimeError):
    pass


class StreamParityError(RuntimeError):
    pass


def normalize_blocks(blocks: list[dict]) -> list[dict]:
    normalized = []
    for block in blocks:
        block_type = block["type"]
        if block_type == "thinking":
            normalized.append({"type": "thinking", "thinking": block.get("thinking", "")})
        elif block_type == "text":
            normalized.append({"type": "text", "text": block.get("text", "")})
        elif block_type == "tool_use":
            normalized.append({"type": "tool_use", "name": block.get("name"), "input": block.get("input") or {}})
        else:
            raise StreamParityError(f"unsupported Anthropic block type: {block_type}")
    return normalized


def validate_block_parity(streamed: list[dict], canonical: list[dict]) -> None:
    if normalize_blocks(streamed) != normalize_blocks(canonical):
        raise StreamParityError(
            f"streamed blocks differ from canonical blocks: "
            f"streamed={_block_summary(streamed)} canonical={_block_summary(canonical)}"
        )


def _block_summary(blocks: list[dict]) -> list[dict]:
    summary = []
    for block in blocks:
        block_type = block.get("type")
        item = {"type": block_type}
        if block_type == "thinking":
            item["length"] = len(block.get("thinking", ""))
        elif block_type == "text":
            item["length"] = len(block.get("text", ""))
        elif block_type == "tool_use":
            item["name"] = block.get("name")
        summary.append(item)
    return summary


class DeltaKind(enum.Enum):
    REASONING = "reasoning"
    TEXT = "text"
    TOOL_START = "tool_start"
    TOOL_INPUT = "tool_input"
    TOOL_STOP = "tool_stop"


@dataclasses.dataclass(frozen=True)
class SemanticDelta:
    kind: DeltaKind
    text: str = ""
    tool_index: int | None = None
    tool_name: str | None = None


@dataclasses.dataclass(frozen=True)
class StreamSnapshot:
    reasoning: str
    text: str
    tool_uses: tuple[dict[str, Any], ...]


class CumulativeTextTracker:
    """Release only text confirmed by two consecutive cumulative chunks."""

    def __init__(self) -> None:
        self._previous = ""
        self._committed = 0

    @staticmethod
    def _common_prefix_length(left: str, right: str) -> int:
        size = min(len(left), len(right))
        index = 0
        while index < size and left[index] == right[index]:
            index += 1
        return index

    def push(self, cumulative_text: str) -> str:
        common = self._common_prefix_length(self._previous, cumulative_text)
        if common < self._committed:
            raise StreamProtocolError("sglang revised text that was already emitted")
        stable = self._previous[self._committed : common]
        self._committed = common
        self._previous = cumulative_text
        return stable

    def finish(self, final_text: str) -> str:
        if not final_text.startswith(self._previous[: self._committed]):
            raise StreamProtocolError("terminal text changed an emitted prefix")
        self._previous = final_text
        stable = final_text[self._committed :]
        self._committed = len(final_text)
        return stable


class _TopState(enum.Enum):
    PREAMBLE = "preamble"
    REASONING = "reasoning"
    TEXT = "text"
    TOOL = "tool"
    POST_TOOL = "post_tool"
    FINISHED = "finished"
    FAILED = "failed"


class _ToolState(enum.Enum):
    EXPECT_FUNCTION = "expect_function"
    FUNCTION_NAME = "function_name"
    EXPECT_PARAMETER_OR_FUNCTION_END = "expect_parameter_or_function_end"
    PARAMETER_NAME = "parameter_name"
    PARAMETER_VALUE = "parameter_value"
    EXPECT_TOOL_END = "expect_tool_end"


class QwenAnthropicStreamParser:
    _THINK_START = "<think>"
    _THINK_END = "</think>"
    _TOOL_START = "<tool_call>"

    _FUNCTION_START = "<function="
    _FUNCTION_END = "</function>"
    _PARAMETER_START = "<parameter="
    _PARAMETER_END = "</parameter>"
    _TOOL_END = "</tool_call>"

    def __init__(
        self,
        *,
        tools_schema: list[dict] | None,
        canonicalize_tool: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        deferred_tool_names: frozenset[str] = frozenset(),
    ) -> None:
        self._tracker = CumulativeTextTracker()
        self._state = _TopState.PREAMBLE
        self._buffer = ""
        self._reasoning_parts: list[str] = []
        self._text_parts: list[str] = []
        self._text_started = False
        self._pending_text_whitespace = ""
        self._tool_uses: list[dict[str, Any]] = []
        self._tools_schema = list(tools_schema or [])
        self._canonicalize_tool = canonicalize_tool
        self._deferred_tool_names = deferred_tool_names
        self._tool_state = _ToolState.EXPECT_FUNCTION
        self._tool_index = 0
        self._tool_name: str | None = None
        self._parameter_name: str | None = None
        self._parameter_type: dict[str, Any] = {}
        self._parameter_buffer: list[str] = []
        self._pending_parameter_newline = ""
        self._parameter_had_leading_newline = False
        self._parameter_leading_checked = False
        self._json_started = False
        self._parameter_count = 0
        self._current_input: dict[str, Any] = {}

    def push(self, cumulative_text: str) -> list[SemanticDelta]:
        stable = self._tracker.push(cumulative_text)
        return self._consume(stable, final=False)

    @property
    def state(self) -> str:
        if self._state is _TopState.TOOL:
            return f"{self._state.value}.{self._tool_state.value}"
        return self._state.value

    def finish(self, final_text: str) -> tuple[list[SemanticDelta], StreamSnapshot]:
        deltas = self._consume(self._tracker.finish(final_text), final=True)
        if self._state is _TopState.TOOL and not self._tools_schema and not self._buffer:
            self._state = _TopState.POST_TOOL
        elif self._state is _TopState.TOOL:
            detail = "parameter" if self._tool_state is _ToolState.PARAMETER_VALUE else self._tool_state.value
            raise StreamProtocolError(f"stream ended in incomplete tool {detail}")
        if self._state is _TopState.POST_TOOL and self._buffer.strip():
            raise StreamProtocolError("stream ended after a tool with unexpected text")
        if self._state is _TopState.POST_TOOL:
            self._buffer = ""
        if self._state not in {
            _TopState.TEXT,
            _TopState.POST_TOOL,
            _TopState.PREAMBLE,
            _TopState.REASONING,
        }:
            raise StreamProtocolError(f"stream ended in parser state {self._state.value}")
        self._pending_text_whitespace = ""
        self._state = _TopState.FINISHED
        return deltas, StreamSnapshot(
            reasoning="".join(self._reasoning_parts),
            text="".join(self._text_parts),
            tool_uses=tuple(self._tool_uses),
        )

    @staticmethod
    def _proper_marker_suffix(text: str, markers: tuple[str, ...]) -> int:
        max_size = min(len(text), max(map(len, markers)) - 1)
        for size in range(max_size, 0, -1):
            suffix = text[-size:]
            if any(marker.startswith(suffix) for marker in markers):
                return size
        return 0

    def _consume(self, text: str, *, final: bool) -> list[SemanticDelta]:
        self._buffer += text
        deltas: list[SemanticDelta] = []
        while self._buffer:
            before = (self._state, self._tool_state, len(self._buffer))
            if self._state is _TopState.PREAMBLE:
                if self._buffer.startswith(self._THINK_START):
                    self._buffer = self._buffer[len(self._THINK_START) :]
                    self._state = _TopState.REASONING
                elif self._buffer.startswith(self._TOOL_START):
                    self._buffer = self._buffer[len(self._TOOL_START) :]
                    self._state = _TopState.TOOL
                elif not final and self._proper_marker_suffix(
                    self._buffer, (self._THINK_START, self._TOOL_START)
                ) == len(self._buffer):
                    break
                else:
                    self._state = _TopState.TEXT
            elif self._state is _TopState.REASONING:
                markers = ((self._THINK_END, _TopState.TEXT), (self._TOOL_START, _TopState.TOOL))
                found = [(self._buffer.find(marker), marker, state) for marker, state in markers]
                found = [item for item in found if item[0] >= 0]
                if found:
                    index, marker, state = min(found, key=lambda item: item[0])
                    deltas.extend(self._emit_reasoning(self._buffer[:index]))
                    self._buffer = self._buffer[index + len(marker) :]
                    self._state = state
                else:
                    retained = (
                        0 if final else self._proper_marker_suffix(self._buffer, (self._THINK_END, self._TOOL_START))
                    )
                    emit_to = len(self._buffer) - retained
                    deltas.extend(self._emit_reasoning(self._buffer[:emit_to]))
                    self._buffer = self._buffer[emit_to:]
                    if retained:
                        break
            elif self._state is _TopState.TEXT:
                index = self._buffer.find(self._TOOL_START)
                if index >= 0:
                    deltas.extend(self._emit_text(self._buffer[:index]))
                    self._buffer = self._buffer[index + len(self._TOOL_START) :]
                    self._state = _TopState.TOOL
                else:
                    retained = 0 if final else self._proper_marker_suffix(self._buffer, (self._TOOL_START,))
                    emit_to = len(self._buffer) - retained
                    deltas.extend(self._emit_text(self._buffer[:emit_to]))
                    self._buffer = self._buffer[emit_to:]
                    if retained:
                        break
            elif self._state is _TopState.TOOL:
                if not self._consume_tool(deltas, final=final):
                    break
            elif self._state is _TopState.POST_TOOL:
                if not self._consume_post_tool(final=final):
                    break
            else:
                raise StreamProtocolError(f"cannot consume text in parser state {self._state.value}")

            if before == (self._state, self._tool_state, len(self._buffer)):
                raise StreamProtocolError(f"parser made no progress in state {self._state.value}")
        return deltas

    def _consume_tool(self, deltas: list[SemanticDelta], *, final: bool) -> bool:
        if self._tool_state is _ToolState.EXPECT_FUNCTION:
            stripped = self._buffer.lstrip()
            self._buffer = stripped
            if self._buffer.startswith(self._FUNCTION_START):
                self._buffer = self._buffer[len(self._FUNCTION_START) :]
                self._tool_state = _ToolState.FUNCTION_NAME
                return True
            if not final and self._FUNCTION_START.startswith(self._buffer):
                return False
            raise StreamProtocolError("expected <function= after <tool_call>")

        if self._tool_state is _ToolState.FUNCTION_NAME:
            end = self._buffer.find(">")
            if end < 0:
                if final:
                    raise StreamProtocolError("stream ended in incomplete tool function name")
                return False
            name = self._buffer[:end]
            if name not in self._known_tool_names():
                raise StreamProtocolError(f"unknown tool: {name}")
            self._buffer = self._buffer[end + 1 :]
            self._tool_name = name
            self._tool_index = len(self._tool_uses)
            self._tool_state = _ToolState.EXPECT_PARAMETER_OR_FUNCTION_END
            if name not in self._deferred_tool_names:
                deltas.append(
                    SemanticDelta(
                        DeltaKind.TOOL_START,
                        tool_index=self._tool_index,
                        tool_name=name,
                    )
                )
            return True

        if self._tool_state is _ToolState.EXPECT_PARAMETER_OR_FUNCTION_END:
            self._buffer = self._buffer.lstrip()
            if self._buffer.startswith(self._PARAMETER_START):
                self._buffer = self._buffer[len(self._PARAMETER_START) :]
                self._tool_state = _ToolState.PARAMETER_NAME
                return True
            if self._buffer.startswith(self._FUNCTION_END):
                self._buffer = self._buffer[len(self._FUNCTION_END) :]
                if self._tool_name not in self._deferred_tool_names:
                    self._emit_tool_input(deltas, "}" if self._json_started else "{}")
                self._tool_state = _ToolState.EXPECT_TOOL_END
                return True
            markers = (self._PARAMETER_START, self._FUNCTION_END)
            if not final and any(marker.startswith(self._buffer) for marker in markers):
                return False
            raise StreamProtocolError("expected tool parameter or </function>")

        if self._tool_state is _ToolState.PARAMETER_NAME:
            end = self._buffer.find(">")
            if end < 0:
                if final:
                    raise StreamProtocolError("stream ended in incomplete parameter name")
                return False
            name = self._buffer[:end]
            if not name:
                raise StreamProtocolError("tool parameter name is empty")
            self._buffer = self._buffer[end + 1 :]
            self._parameter_name = name
            self._parameter_type = self._parameter_schema(self._tool_name or "", name)
            self._parameter_buffer = []
            self._pending_parameter_newline = ""
            self._parameter_had_leading_newline = False
            self._parameter_leading_checked = False
            prefix = ("{" if self._parameter_count == 0 else ",") + json.dumps(name, ensure_ascii=False) + ":"
            if self._tool_name not in self._deferred_tool_names:
                self._emit_tool_input(deltas, prefix)
                if self._is_string_parameter():
                    self._emit_tool_input(deltas, '"')
            self._json_started = True
            self._tool_state = _ToolState.PARAMETER_VALUE
            return True

        if self._tool_state is _ToolState.PARAMETER_VALUE:
            end = self._buffer.find(self._PARAMETER_END)
            if end >= 0:
                self._consume_parameter_fragment(self._buffer[:end], deltas)
                self._buffer = self._buffer[end + len(self._PARAMETER_END) :]
                self._finish_parameter(deltas)
                self._tool_state = _ToolState.EXPECT_PARAMETER_OR_FUNCTION_END
                return True
            if final:
                raise StreamProtocolError("stream ended in incomplete tool parameter")
            retained = self._proper_marker_suffix(self._buffer, (self._PARAMETER_END,))
            emit_to = len(self._buffer) - retained
            self._consume_parameter_fragment(self._buffer[:emit_to], deltas)
            self._buffer = self._buffer[emit_to:]
            return emit_to > 0

        if self._tool_state is _ToolState.EXPECT_TOOL_END:
            self._buffer = self._buffer.lstrip()
            if self._buffer.startswith(self._TOOL_END):
                self._buffer = self._buffer[len(self._TOOL_END) :]
                self._finish_tool(deltas)
                self._state = _TopState.POST_TOOL
                return True
            if not final and self._TOOL_END.startswith(self._buffer):
                return False
            raise StreamProtocolError("expected </tool_call> after </function>")

        raise StreamProtocolError(f"unsupported tool parser state {self._tool_state.value}")

    def _consume_post_tool(self, *, final: bool) -> bool:
        rest = self._buffer.lstrip()
        if not rest:
            if final:
                self._buffer = ""
                return True
            return False
        if rest.startswith(self._TOOL_START):
            self._buffer = rest[len(self._TOOL_START) :]
            self._reset_tool()
            self._state = _TopState.TOOL
            return True
        if not final and self._TOOL_START.startswith(rest):
            return False
        raise StreamProtocolError("text after a completed tool cannot preserve Anthropic block order")

    def _known_tool_names(self) -> set[str]:
        return {
            str(tool.get("function", {}).get("name"))
            for tool in self._tools_schema
            if tool.get("function", {}).get("name")
        }

    def _parameter_schema(self, tool_name: str, parameter_name: str) -> dict[str, Any]:
        for tool in self._tools_schema:
            function = tool.get("function", {})
            if function.get("name") == tool_name:
                properties = function.get("parameters", {}).get("properties", {})
                schema = properties.get(parameter_name, {})
                return schema if isinstance(schema, dict) else {}
        return {}

    def _is_string_parameter(self) -> bool:
        value_type = str(self._parameter_type.get("type", "string")).lower()
        return value_type in {"string", "str", "text", "varchar", "char", "enum"}

    def _consume_parameter_fragment(self, fragment: str, deltas: list[SemanticDelta]) -> None:
        if not fragment:
            return
        if not self._parameter_leading_checked:
            self._parameter_leading_checked = True
            if fragment.startswith("\n"):
                fragment = fragment[1:]
                self._parameter_had_leading_newline = True

        candidate = self._pending_parameter_newline + fragment
        self._pending_parameter_newline = ""
        if self._parameter_had_leading_newline and candidate.endswith("\n"):
            candidate, self._pending_parameter_newline = candidate[:-1], "\n"
        if not candidate:
            return
        self._parameter_buffer.append(candidate)
        if self._is_string_parameter() and self._tool_name not in self._deferred_tool_names:
            self._emit_tool_input(deltas, json.dumps(candidate, ensure_ascii=False)[1:-1])

    def _finish_parameter(self, deltas: list[SemanticDelta]) -> None:
        name = self._parameter_name
        if name is None:
            raise StreamProtocolError("parameter closed without a name")
        raw = "".join(self._parameter_buffer)
        if self._is_string_parameter():
            value: Any = raw
            if self._tool_name not in self._deferred_tool_names:
                self._emit_tool_input(deltas, '"')
        else:
            value = self._convert_complete_value(raw, self._parameter_type)
            if self._tool_name not in self._deferred_tool_names:
                self._emit_tool_input(deltas, json.dumps(value, ensure_ascii=False, separators=(",", ":")))
        self._current_input[name] = value
        self._parameter_count += 1
        self._parameter_name = None
        self._pending_parameter_newline = ""

    def _finish_tool(self, deltas: list[SemanticDelta]) -> None:
        name = self._tool_name
        if name is None:
            raise StreamProtocolError("tool closed without a function name")
        tool_use = {"name": name, "input": dict(self._current_input)}
        if name in self._deferred_tool_names:
            if self._canonicalize_tool is not None:
                tool_use = self._canonicalize_tool(tool_use)
            deltas.append(
                SemanticDelta(
                    DeltaKind.TOOL_START,
                    tool_index=self._tool_index,
                    tool_name=str(tool_use.get("name") or "tool"),
                )
            )
            self._emit_tool_input(
                deltas,
                json.dumps(tool_use.get("input") or {}, ensure_ascii=False, separators=(",", ":")),
            )
        deltas.append(SemanticDelta(DeltaKind.TOOL_STOP, tool_index=self._tool_index))
        self._tool_uses.append(tool_use)

    def _reset_tool(self) -> None:
        self._tool_state = _ToolState.EXPECT_FUNCTION
        self._tool_name = None
        self._parameter_name = None
        self._parameter_type = {}
        self._parameter_buffer = []
        self._pending_parameter_newline = ""
        self._parameter_had_leading_newline = False
        self._parameter_leading_checked = False
        self._json_started = False
        self._parameter_count = 0
        self._current_input = {}

    def _emit_tool_input(self, deltas: list[SemanticDelta], text: str) -> None:
        if text:
            deltas.append(SemanticDelta(DeltaKind.TOOL_INPUT, text=text, tool_index=self._tool_index))

    @staticmethod
    def _convert_complete_value(raw: str, schema: dict[str, Any]) -> Any:
        value_type = str(schema.get("type", "string")).lower()
        if raw.lower() == "null":
            return None
        if value_type in {"string", "str", "text", "varchar", "char", "enum"}:
            return raw
        if value_type in {"boolean", "bool", "binary"}:
            return raw.lower() == "true"
        if value_type.startswith(("int", "uint", "long", "short", "unsigned")):
            try:
                return int(raw)
            except ValueError:
                return raw
        if value_type.startswith(("num", "float")):
            try:
                number = float(raw)
            except ValueError:
                return raw
            return int(number) if number.is_integer() and "." not in raw and "e" not in raw.lower() else number
        if value_type in {"object", "array", "arr"} or value_type.startswith(("dict", "list")):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw

    def _emit_reasoning(self, text: str) -> list[SemanticDelta]:
        if not text:
            return []
        self._reasoning_parts.append(text)
        return [SemanticDelta(DeltaKind.REASONING, text=text)]

    def _emit_text(self, text: str) -> list[SemanticDelta]:
        candidate = self._pending_text_whitespace + text
        self._pending_text_whitespace = ""
        if not self._text_started:
            candidate = candidate.lstrip()
            if not candidate:
                return []
            self._text_started = True

        emitted = candidate.rstrip()
        self._pending_text_whitespace = candidate[len(emitted) :]
        if not emitted:
            return []
        self._text_parts.append(emitted)
        return [SemanticDelta(DeltaKind.TEXT, text=emitted)]


async def _write_event(response: web.StreamResponse, name: str, payload: dict) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    await response.write(f"event: {name}\ndata: {encoded}\n\n".encode())


class AnthropicStreamResponse:
    """Translate semantic model deltas into one Anthropic SSE response."""

    def __init__(
        self,
        *,
        response: web.StreamResponse,
        parser: QwenAnthropicStreamParser,
        input_tokens: int,
        model: str,
        logger: logging.Logger,
        log_prefix: str,
        sid: str,
    ) -> None:
        self.response = response
        self.parser = parser
        self.input_tokens = input_tokens
        self.model = model
        self.logger = logger
        self.log_prefix = log_prefix
        self.sid = sid
        self.started_at = time.monotonic()
        self.first_chunk_ms: float | None = None
        self.first_content_delta_ms: float | None = None
        self.last_content_delta_ms: float | None = None
        self.stream_chunks = 0
        self.content_delta_count = 0
        self.latest_text = ""
        self.rid = "-"
        self._summary_logged = False
        self._next_block_index = 0
        self._open_block_index: int | None = None
        self._open_block_kind: DeltaKind | None = None
        self._tool_ids: dict[int, str] = {}

    async def on_progress(self, progress: GenerationProgress) -> None:
        now = time.monotonic()
        if self.first_chunk_ms is None:
            self.first_chunk_ms = (now - self.started_at) * 1000
        self.stream_chunks = progress.chunk_index
        self.rid = progress.rid
        self.latest_text = progress.text
        await self._write_deltas(self._coalesce(self.parser.push(progress.text)))

    async def finish(
        self,
        reply: Reply,
        raw_output: str,
        input_tokens: int,
        output_tokens: int,
    ) -> web.StreamResponse:
        if self.latest_text != raw_output:
            raise StreamParityError(
                f"terminal SGLang text differs from local decode: "
                f"sglang_length={len(self.latest_text)} decoded_length={len(raw_output)}"
            )
        deltas, snapshot = self.parser.finish(raw_output)
        await self._write_deltas(self._coalesce(deltas))
        await self._close_block()
        canonical_blocks, stop_reason = reply.wire
        streamed_blocks = self._snapshot_blocks(snapshot)
        if not streamed_blocks and canonical_blocks == [{"type": "text", "text": ""}]:
            await self._open_block(DeltaKind.TEXT, {"type": "text", "text": ""})
            await self._close_block()
            streamed_blocks = [{"type": "text", "text": ""}]
        validate_block_parity(streamed_blocks, canonical_blocks)
        await _write_event(
            self.response,
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            },
        )
        await _write_event(self.response, "message_stop", {"type": "message_stop"})
        await self.response.write_eof()
        self._log_summary("success", output_tokens=output_tokens)
        return self.response

    def _snapshot_blocks(self, snapshot: StreamSnapshot) -> list[dict]:
        blocks: list[dict] = []
        if snapshot.reasoning:
            blocks.append({"type": "thinking", "thinking": snapshot.reasoning})
        if snapshot.text:
            blocks.append({"type": "text", "text": snapshot.text})
        for index, tool_use in enumerate(snapshot.tool_uses):
            blocks.append(
                {
                    "type": "tool_use",
                    "id": self._tool_ids.get(index, ""),
                    "name": tool_use.get("name"),
                    "input": tool_use.get("input") or {},
                }
            )
        return blocks

    async def fail(self, error: Exception) -> web.StreamResponse:
        from slime.agent.adapters.common import ContextWindowExceeded

        self._log_summary("failure", output_tokens=0, error=error)
        if isinstance(error, ContextWindowExceeded):
            error_type = "invalid_request_error"
            message = str(error)
        else:
            error_type = "api_error"
            message = "SGLang generation failed"
        await _write_event(
            self.response,
            "error",
            {"type": "error", "error": {"type": error_type, "message": message}},
        )
        await self.response.write_eof()
        return self.response

    def _log_summary(self, outcome: str, *, output_tokens: int, error: Exception | None = None) -> None:
        if self._summary_logged:
            return
        self._summary_logged = True
        self.logger.info(
            "[%s] sid=%s rid=%s event=anthropic_stream_summary outcome=%s "
            "first_chunk_ms=%.1f first_content_delta_ms=%.1f last_content_delta_ms=%.1f "
            "stream_chunks=%d content_delta_count=%d output_tokens=%d parser_state=%s exception_type=%s",
            self.log_prefix,
            self.sid,
            self.rid,
            outcome,
            self.first_chunk_ms if self.first_chunk_ms is not None else -1.0,
            self.first_content_delta_ms if self.first_content_delta_ms is not None else -1.0,
            self.last_content_delta_ms if self.last_content_delta_ms is not None else -1.0,
            self.stream_chunks,
            self.content_delta_count,
            output_tokens,
            self.parser.state,
            type(error).__name__ if error is not None else "-",
        )

    @staticmethod
    def _coalesce(deltas: list[SemanticDelta]) -> list[SemanticDelta]:
        coalesced: list[SemanticDelta] = []
        for delta in deltas:
            if (
                coalesced
                and delta.kind in {DeltaKind.REASONING, DeltaKind.TEXT, DeltaKind.TOOL_INPUT}
                and coalesced[-1].kind is delta.kind
                and coalesced[-1].tool_index == delta.tool_index
            ):
                previous = coalesced[-1]
                coalesced[-1] = dataclasses.replace(previous, text=previous.text + delta.text)
            else:
                coalesced.append(delta)
        return coalesced

    async def _write_deltas(self, deltas: list[SemanticDelta]) -> None:
        for delta in deltas:
            if delta.kind is DeltaKind.REASONING:
                await self._ensure_block(DeltaKind.REASONING, {"type": "thinking", "thinking": ""})
                await self._write_content_delta({"type": "thinking_delta", "thinking": delta.text})
            elif delta.kind is DeltaKind.TEXT:
                await self._ensure_block(DeltaKind.TEXT, {"type": "text", "text": ""})
                await self._write_content_delta({"type": "text_delta", "text": delta.text})
            elif delta.kind is DeltaKind.TOOL_START:
                await self._close_block()
                tool_index = delta.tool_index if delta.tool_index is not None else len(self._tool_ids)
                tool_id = f"toolu_{secrets.token_hex(12)}"
                self._tool_ids[tool_index] = tool_id
                await self._open_block(
                    DeltaKind.TOOL_START,
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": delta.tool_name or "tool",
                        "input": {},
                    },
                )
            elif delta.kind is DeltaKind.TOOL_INPUT:
                if self._open_block_kind is not DeltaKind.TOOL_START:
                    raise StreamProtocolError("tool input arrived without an open tool block")
                await self._write_content_delta({"type": "input_json_delta", "partial_json": delta.text})
            elif delta.kind is DeltaKind.TOOL_STOP:
                if self._open_block_kind is not DeltaKind.TOOL_START:
                    raise StreamProtocolError("tool stop arrived without an open tool block")
                await self._close_block()

    async def _ensure_block(self, kind: DeltaKind, content_block: dict) -> None:
        if self._open_block_kind is kind:
            return
        await self._close_block()
        await self._open_block(kind, content_block)

    async def _open_block(self, kind: DeltaKind, content_block: dict) -> None:
        index = self._next_block_index
        self._next_block_index += 1
        self._open_block_index = index
        self._open_block_kind = kind
        await _write_event(
            self.response,
            "content_block_start",
            {"type": "content_block_start", "index": index, "content_block": content_block},
        )

    async def _write_content_delta(self, delta: dict) -> None:
        if self._open_block_index is None:
            raise StreamProtocolError("content delta arrived without an open block")
        now = time.monotonic()
        elapsed_ms = (now - self.started_at) * 1000
        if self.first_content_delta_ms is None:
            self.first_content_delta_ms = elapsed_ms
        self.last_content_delta_ms = elapsed_ms
        self.content_delta_count += 1
        await _write_event(
            self.response,
            "content_block_delta",
            {"type": "content_block_delta", "index": self._open_block_index, "delta": delta},
        )

    async def _close_block(self) -> None:
        if self._open_block_index is None:
            return
        await _write_event(
            self.response,
            "content_block_stop",
            {"type": "content_block_stop", "index": self._open_block_index},
        )
        self._open_block_index = None
        self._open_block_kind = None
