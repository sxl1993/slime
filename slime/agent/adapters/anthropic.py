"""Anthropic Messages adapter for agent rollouts.

Exposes /v1/messages and /v1/messages/count_tokens. Each Anthropic message
history is rendered with the served model's chat template, sent to sglang
/generate as input_ids, and fed into a shared TrajectoryManager keyed by session
id. finish_session(sid) drains a session's trajectory into a list of Sample.

The per-sid tree inside TrajectoryManager handles sub-agent and compaction
patterns automatically: any divergence in the prompt prefix forks into a new
leaf, so we do not track explicit chains here.

This module mirrors slime.agent.adapters.openai; the section layout (adapter
class -> translation -> reply building -> request framing) is shared between
them. See BaseAdapter for the hooks to fill.
"""

from __future__ import annotations

import json
import logging
import secrets
import shlex
from typing import Any

from aiohttp import web

from slime.agent.adapters.common import (
    BaseAdapter,
    ContextWindowExceeded,
    IncrementalStream,
    Reply,
    flatten_content,
    manager_finish_reason,
    sid_from_bearer,
    tool_call_dict,
)
from slime.agent.adapters.anthropic_stream import AnthropicStreamResponse, QwenAnthropicStreamParser
from slime.agent.parsing import ParsedModelOutput

logger = logging.getLogger(__name__)

_TOOL_RESULT_TRUNCATION_MARKER = "\n...[tool result truncated]...\n"


class AnthropicAdapter(BaseAdapter):
    """Anthropic Messages-compatible HTTP adapter: wire translation and reply
    framing only; the turn machinery is inherited from BaseAdapter."""

    logger = logger
    log_prefix = "anthropic_adapter"
    max_token_keys = ("max_tokens",)
    stop_keys = ("stop_sequences",)

    def __init__(self, *, max_tool_result_chars: int | None = None, **kwargs) -> None:
        if max_tool_result_chars is not None and max_tool_result_chars <= 0:
            raise ValueError("max_tool_result_chars must be positive")
        super().__init__(**kwargs)
        self.max_tool_result_chars = max_tool_result_chars

    def _register_routes(self, app: web.Application) -> None:
        app.router.add_post("/v1/messages", self._run_turn)
        app.router.add_post("/v1/messages/count_tokens", self._count_tokens)

    def _session_id(self, request: web.Request, body: dict) -> str:
        return _request_session_id(request, body)

    def _preprocess_body(self, body: dict) -> None:
        _fold_mid_list_system_into_user(body)

    def _translate(self, body: dict) -> tuple[list[dict], list[dict] | None]:
        translated = _translate_messages(
            body.get("messages") or [],
            body.get("system"),
            max_tool_result_chars=self.max_tool_result_chars,
        )
        tools_schema = _tools_to_chat_tools(body.get("tools"))
        return translated, tools_schema

    def _build_reply(self, parsed, raw_finish, translated, tools_schema) -> Reply:
        blocks, stop_reason, manager_message = _build_reply_parts(parsed, raw_finish)
        return Reply(
            manager_message=manager_message,
            finish_reason=manager_finish_reason(parsed.tool_uses, raw_finish),
            wire=(blocks, stop_reason),
        )

    async def _respond(self, request, body, reply, in_tok, out_tok, stream) -> web.StreamResponse:
        blocks, stop_reason = reply.wire
        if stream:
            return await _render_stream(request, body, blocks, stop_reason, in_tok, out_tok)
        return web.json_response(_render_response(body, blocks, stop_reason, in_tok, out_tok))

    async def _start_incremental_stream(
        self,
        request: web.Request,
        body: dict,
        input_tokens: int,
        tools_schema: list[dict] | None,
    ) -> IncrementalStream | None:
        stream = body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", "")
        if not stream:
            return None
        if self.reasoning_parser not in {None, "qwen3"} or self.tool_parser not in {None, "qwen3_coder"}:
            return None

        response = await _start_stream(request, input_tokens, body.get("model", "slime-actor"))
        parser = QwenAnthropicStreamParser(
            tools_schema=tools_schema,
            canonicalize_tool=_lower_search_tool,
            deferred_tool_names=frozenset({"Grep", "Glob"}),
        )
        return AnthropicStreamResponse(
            response=response,
            parser=parser,
            input_tokens=input_tokens,
            model=body.get("model", "slime-actor"),
        )

    def _context_limit_response(self, error: ContextWindowExceeded) -> web.Response:
        return web.json_response(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": (
                        f"Prompt is too long: {error.prompt_tokens} input tokens plus "
                        f"{error.output_tokens} requested output tokens exceeds the "
                        f"{error.max_context_tokens} token context window"
                    ),
                },
            },
            status=400,
        )

    async def _count_tokens(self, request: web.Request) -> web.Response:
        body = await request.json()
        self._preprocess_body(body)
        _, _, prompt_ids = self._prepare_prompt(body)
        return web.json_response({"input_tokens": len(prompt_ids)})


# --- Translation (Anthropic wire -> chat-template messages) ---


def _truncate_tool_result(content: str, max_chars: int | None) -> str:
    if max_chars is None or len(content) <= max_chars:
        return content
    if max_chars <= len(_TOOL_RESULT_TRUNCATION_MARKER):
        return content[:max_chars]

    kept_chars = max_chars - len(_TOOL_RESULT_TRUNCATION_MARKER)
    tail_chars = (kept_chars + 1) // 2
    head_chars = kept_chars - tail_chars
    tail = content[-tail_chars:] if tail_chars else ""
    return content[:head_chars] + _TOOL_RESULT_TRUNCATION_MARKER + tail


def _translate_messages(
    msgs: list[dict],
    system: Any,
    *,
    max_tool_result_chars: int | None = None,
) -> list[dict]:
    """Anthropic messages + system -> chat-template messages. Pure function."""
    translated: list[dict] = []
    if system:
        translated.append({"role": "system", "content": flatten_content(system)})
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role, content = m.get("role"), m.get("content")
        if role == "user":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": flatten_content(content)}]
            for b in blocks:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    tool_result = _truncate_tool_result(flatten_content(b.get("content")), max_tool_result_chars)
                    translated.append({"role": "tool", "content": tool_result})
                elif isinstance(b, dict) and b.get("type") == "text":
                    translated.append({"role": "user", "content": b.get("text", "")})
                else:
                    translated.append({"role": "user", "content": flatten_content(b)})
        elif role == "assistant":
            texts, thinkings, tcs = [], [], []
            blocks = content if isinstance(content, list) else [{"type": "text", "text": flatten_content(content)}]
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "text":
                    texts.append(b.get("text", ""))
                elif b.get("type") == "thinking":
                    thinkings.append(b.get("thinking", ""))
                elif b.get("type") == "tool_use":
                    # drop the wire-only id; tool_call_dict keeps arguments a dict
                    tcs.append(tool_call_dict(b.get("name", "tool"), b.get("input")))
            mo: dict[str, Any] = {"role": "assistant", "content": "".join(texts)}
            if thinkings:
                mo["reasoning_content"] = "".join(thinkings)
            if tcs:
                mo["tool_calls"] = tcs
            translated.append(mo)
        elif role == "system":
            translated.append({"role": "system", "content": flatten_content(content)})
    return translated


def _tools_to_chat_tools(anth_tools: list[dict] | None) -> list[dict] | None:
    """Convert Anthropic tools to tokenizer chat-template tool schema."""
    if not anth_tools:
        return None
    ts: list[dict] = []
    for t in anth_tools:
        if not isinstance(t, dict) or "name" not in t:
            continue
        ts.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or t.get("parameters") or {"type": "object", "properties": {}},
                },
            }
        )
    return ts or None


# --- Reply building: parsed output -> Anthropic blocks + manager_message ---


def _lower_search_tool(tool_use: dict[str, Any]) -> dict[str, Any]:
    """Lower legacy Grep/Glob calls to Bash tools available in Claude Code."""
    name = tool_use.get("name")
    if name not in {"Grep", "Glob"}:
        return tool_use

    tool_input = tool_use.get("input")
    args = tool_input if isinstance(tool_input, dict) else {}
    pattern = args.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        command = f"printf '%s\\n' {shlex.quote(f'{name} requires a non-empty pattern')} >&2; exit 2"
    elif name == "Glob":
        path = args.get("path") if isinstance(args.get("path"), str) and args.get("path") else "."
        normalized_pattern = pattern.replace("**/", "")
        match_flag = "-path" if "/" in normalized_pattern else "-name"
        match_pattern = f"*/{normalized_pattern}" if match_flag == "-path" else normalized_pattern
        command = f"{shlex.join(['find', path, '-type', 'f', match_flag, match_pattern])} | head -n 100"
    else:
        path = args.get("path") if isinstance(args.get("path"), str) and args.get("path") else "."
        output_mode = args.get("output_mode")
        grep_args = ["grep", "-R"]
        if output_mode == "content":
            if args.get("-n", True) not in (False, "false", "False", "0"):
                grep_args.append("-n")
            for option in ("-B", "-A", "-C"):
                if isinstance(args.get(option), (int, str)):
                    grep_args.extend([option, str(args[option])])
            if isinstance(args.get("context"), (int, str)):
                grep_args.extend(["-C", str(args["context"])])
        elif output_mode == "count":
            grep_args.append("-c")
        else:
            grep_args.append("-l")
        if args.get("-i") in (True, "true", "True", "1"):
            grep_args.append("-i")
        if args.get("-o") in (True, "true", "True", "1"):
            grep_args.append("-o")
        glob = args.get("glob")
        if isinstance(glob, str) and glob:
            grep_args.append(f"--include={glob}")
        elif isinstance(args.get("type"), str) and args["type"]:
            grep_args.append(f"--include=*.{args['type']}")
        grep_args.extend(["--", pattern, path])

        try:
            offset = max(0, int(args.get("offset", 0)))
        except (TypeError, ValueError):
            offset = 0
        try:
            head_limit = max(0, int(args.get("head_limit", 250)))
        except (TypeError, ValueError):
            head_limit = 250
        command = shlex.join(grep_args)
        if head_limit:
            command += f" | sed -n {offset + 1},{offset + head_limit}p"
        elif offset:
            command += f" | tail -n +{offset + 1}"

    return {
        "name": "Bash",
        "input": {
            "command": command,
            "description": "Search file contents" if name == "Grep" else "Match file paths",
        },
    }


def _build_reply_parts(
    parsed: ParsedModelOutput,
    finish: str,
) -> tuple[list[dict], str, dict[str, Any]]:
    """Return (anthropic blocks, wire stop_reason, manager_message).

    The tool_calls inside manager_message use canonical args (tool_call_dict) so
    this assistant turn compares equal (dict equality) to the same turn replayed
    as history on the next request.
    """
    blocks: list[dict] = []
    if parsed.reasoning:
        blocks.append({"type": "thinking", "thinking": parsed.reasoning})
    if parsed.text:
        blocks.append({"type": "text", "text": parsed.text})

    manager_tcs: list[dict] = []
    for tu in parsed.tool_uses:
        tu = _lower_search_tool(tu)
        tu_id = f"toolu_{secrets.token_hex(8)}"
        blocks.append({"type": "tool_use", "id": tu_id, "name": tu["name"], "input": tu["input"]})
        # tu_id is wire-only; tool_call_dict drops it so the leaf matches its echo
        manager_tcs.append(tool_call_dict(tu["name"], tu.get("input")))

    if not blocks:
        blocks.append({"type": "text", "text": ""})

    if parsed.tool_uses:
        stop_reason = "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    manager_message: dict[str, Any] = {"role": "assistant", "content": parsed.text or ""}
    if parsed.reasoning:
        manager_message["reasoning_content"] = parsed.reasoning
    if manager_tcs:
        manager_message["tool_calls"] = manager_tcs

    return blocks, stop_reason, manager_message


# --- Request framing: session id + wire response/stream rendering ---


def _request_session_id(request: web.Request, body: dict) -> str:
    metadata = body.get("metadata")
    if isinstance(metadata, dict) and isinstance(user_id := metadata.get("user_id"), str):
        try:
            user = json.loads(user_id)
        except json.JSONDecodeError:
            user = None
        if isinstance(user, dict) and isinstance(session_id := user.get("session_id"), str):
            if session_id := session_id.strip():
                return session_id

    # Direct clients carry the sid in Anthropic auth. Theta rewrites these
    # headers, so Claude Code metadata takes precedence when it is present.
    return sid_from_bearer(request) or (request.headers.get("X-Api-Key") or "").strip() or "default"


def _render_response(body: dict, blocks: list[dict], stop_reason: str, in_tok: int, out_tok: int) -> dict:
    return {
        "id": f"msg_{secrets.token_hex(12)}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", "slime-actor"),
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }


async def _start_stream(request, in_tok, model="slime-actor") -> web.StreamResponse:
    out = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
    await out.prepare(request)

    ms_data = {
        "type": "message_start",
        "message": {
            "id": f"msg_{secrets.token_hex(12)}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": in_tok, "output_tokens": 0},
        },
    }
    await out.write(f"event: message_start\ndata: {json.dumps(ms_data, ensure_ascii=False)}\n\n".encode())
    return out


async def _render_stream(request, body, blocks, stop_reason, in_tok, out_tok) -> web.StreamResponse:
    response = await _start_stream(request, in_tok, body.get("model", "slime-actor"))
    return await _finish_stream(response, blocks, stop_reason, in_tok, out_tok)


async def _finish_stream(out, blocks, stop_reason, in_tok, out_tok) -> web.StreamResponse:
    """Finish an Anthropic Messages SSE response after generation completes."""

    for idx, block in enumerate(blocks):
        bt = block["type"]
        if bt == "thinking":
            start = {"type": "thinking", "thinking": ""}
            delta = {"type": "thinking_delta", "thinking": block["thinking"]}
        elif bt == "text":
            start = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": block["text"]}
        else:  # tool_use
            start = {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}
            delta = {
                "type": "input_json_delta",
                "partial_json": json.dumps(block["input"], ensure_ascii=False),
            }

        cbs_data = {"type": "content_block_start", "index": idx, "content_block": start}
        await out.write(f"event: content_block_start\ndata: {json.dumps(cbs_data, ensure_ascii=False)}\n\n".encode())

        cbd_data = {"type": "content_block_delta", "index": idx, "delta": delta}
        await out.write(f"event: content_block_delta\ndata: {json.dumps(cbd_data, ensure_ascii=False)}\n\n".encode())

        cbe_data = {"type": "content_block_stop", "index": idx}
        await out.write(f"event: content_block_stop\ndata: {json.dumps(cbe_data, ensure_ascii=False)}\n\n".encode())

    md_data = {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": None},
        "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
    }
    await out.write(f"event: message_delta\ndata: {json.dumps(md_data, ensure_ascii=False)}\n\n".encode())

    mst_data = {"type": "message_stop"}
    await out.write(f"event: message_stop\ndata: {json.dumps(mst_data, ensure_ascii=False)}\n\n".encode())

    return out


# --- Anthropic-specific quirks: mid-list system folding ---


_MID_SYSTEM_WRAP_PREFIX = "<system-reminder>\n"
_MID_SYSTEM_WRAP_SUFFIX = "\n</system-reminder>\n"


def _fold_mid_list_system_into_user(body_obj: dict) -> bool:
    """Fold non-leading role:system messages into a neighbouring user message as
    a <system-reminder> text block. Mutates body_obj in place; returns True iff
    any fold happened.

    Some clients insert a system message in the middle of the message list, but
    many chat templates reject any system message past index 0. Attaching the
    wrapped reminder to the preceding user message (or the next one, if there is
    no prior user message) keeps the history acceptable to the template.
    """
    msgs = body_obj.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return False

    system_idx = [i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "system" and i > 0]
    if not system_idx:
        return False

    def _promote_to_list(msg: dict) -> list:
        c = msg.get("content")
        if isinstance(c, list):
            return c
        msg["content"] = [{"type": "text", "text": c if isinstance(c, str) else ""}]
        return msg["content"]

    def _wrap(text: str) -> dict:
        return {
            "type": "text",
            "text": _MID_SYSTEM_WRAP_PREFIX + text + _MID_SYSTEM_WRAP_SUFFIX,
        }

    changed = False
    TOMBSTONE: dict = {"__folded__": True}
    for i in system_idx:
        sys_msg = msgs[i]
        wrapped = _wrap(flatten_content(sys_msg.get("content")))
        target = None
        for j in range(i - 1, -1, -1):
            cand = msgs[j]
            if isinstance(cand, dict) and cand.get("role") == "user":
                target = cand
                _promote_to_list(target).append(wrapped)
                break
        if target is None:
            for j in range(i + 1, len(msgs)):
                cand = msgs[j]
                if isinstance(cand, dict) and cand.get("role") == "user":
                    target = cand
                    _promote_to_list(target).insert(0, wrapped)
                    break
        if target is None:
            msgs[i] = {"role": "user", "content": [wrapped]}
            changed = True
            continue
        msgs[i] = TOMBSTONE
        changed = True

    if changed:
        body_obj["messages"] = [m for m in msgs if m is not TOMBSTONE]
    return changed
