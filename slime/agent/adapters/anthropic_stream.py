"""Incremental Qwen output parsing for Anthropic streaming responses."""

from __future__ import annotations

import dataclasses
import enum
from typing import Any


class StreamProtocolError(RuntimeError):
    pass


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


class QwenAnthropicStreamParser:
    _THINK_START = "<think>"
    _THINK_END = "</think>"
    _TOOL_START = "<tool_call>"

    def __init__(self, *, tools_schema: list[dict] | None) -> None:
        self._tracker = CumulativeTextTracker()
        self._state = _TopState.PREAMBLE
        self._buffer = ""
        self._reasoning_parts: list[str] = []
        self._text_parts: list[str] = []
        self._text_started = False
        self._pending_text_whitespace = ""
        self._tool_uses: list[dict[str, Any]] = []
        self._tools_schema = list(tools_schema or [])

    def push(self, cumulative_text: str) -> list[SemanticDelta]:
        stable = self._tracker.push(cumulative_text)
        return self._consume(stable, final=False)

    def finish(self, final_text: str) -> tuple[list[SemanticDelta], StreamSnapshot]:
        deltas = self._consume(self._tracker.finish(final_text), final=True)
        if self._state is _TopState.TOOL and not self._buffer:
            self._state = _TopState.POST_TOOL
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
            before = (self._state, len(self._buffer))
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
                break
            else:
                raise StreamProtocolError(f"cannot consume text in parser state {self._state.value}")

            if before == (self._state, len(self._buffer)):
                raise StreamProtocolError(f"parser made no progress in state {self._state.value}")
        return deltas

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
