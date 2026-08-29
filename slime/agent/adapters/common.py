"""Shared adapter primitives for token-capturing agent rollouts.

A protocol adapter (Anthropic / OpenAI) subclasses BaseAdapter and fills in the
wire-specific hooks (_register_routes, _session_id, _translate, _build_reply,
_respond, and optionally _preprocess_body) plus a few class attributes (logger,
log_prefix, max_token_keys, stop_keys). The session lifecycle, per-sid turn cap,
inflight-task bookkeeping and the one-turn _run_turn pipeline are inherited.

flatten_content, tool_call_dict and manager_finish_reason cover the parts both
protocols handle identically.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import aiohttp
from aiohttp import web

from slime.agent.parsing import parse_model_output
from slime.agent.trajectory import TrajectoryManager, TurnRecord

__all__ = ["GenerationProgress", "TurnRecord"]


_COMPACTION_MIN_DROP_TOKENS = 8192
_COMPACTION_MIN_DROP_FRACTION = 0.25
_UPSTREAM_GENERATE_URL_RE = re.compile(r"https?://[^()\s]+/generate")


@dataclasses.dataclass
class Session:
    """Per-sid adapter state: sampling defaults and context budget.

    Trajectory state lives in the shared TrajectoryManager (BaseAdapter.manager),
    not here.
    """

    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    max_context_tokens: int = 0
    last_prompt_tokens: int = 0
    max_prompt_tokens: int = 0
    compaction_count: int = 0
    context_exceeded_count: int = 0


@dataclasses.dataclass
class Reply:
    """Output of an adapter's _build_reply, consumed by _run_turn.

    manager_message and finish_reason feed record_turn and the debug callback;
    wire is opaque to the pipeline and only the adapter's own _respond reads it.
    """

    manager_message: dict
    finish_reason: str
    wire: Any


@dataclasses.dataclass(frozen=True)
class GenerationProgress:
    rid: str
    chunk_index: int
    text: str
    output_token_logprobs: tuple[tuple[float, int], ...]
    finish_reason: str | None


ProgressCallback = Callable[[GenerationProgress], Awaitable[None]]


class ContextWindowExceeded(Exception):
    """The requested turn cannot fit its full output budget in the context."""

    def __init__(self, *, prompt_tokens: int, output_tokens: int, max_context_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.max_context_tokens = max_context_tokens
        super().__init__(f"prompt={prompt_tokens} output_budget={output_tokens} max_context={max_context_tokens}")


def _render_token_ids(
    messages: list[dict],
    tokenizer,
    *,
    tools: list[dict] | None,
    add_generation_prompt: bool = True,
) -> list[int]:
    """Render a chat-message list to token ids with the served chat template."""
    enc = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
    return list(ids)


def flatten_content(c: Any) -> str:
    """Flatten a wire content value into a chat-template string.

    Handles both Anthropic and OpenAI block shapes. A non-list value (str /
    dict / other) is returned via str() unchanged.
    """
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if not isinstance(c, list):
        return str(c)
    parts: list[str] = []
    for b in c:
        if isinstance(b, str):
            parts.append(b)
            continue
        if not isinstance(b, dict):
            parts.append(str(b))
            continue
        t = b.get("type")
        if t in {"text", "input_text", "output_text"}:
            parts.append(b.get("text", ""))
        elif t == "tool_result":
            parts.append(flatten_content(b.get("content")))
        elif t in {"image", "image_url", "input_image"}:
            parts.append("[image omitted]")
        elif "content" in b:
            parts.append(flatten_content(b.get("content")))
        elif "text" in b:
            parts.append(str(b.get("text") or ""))
    return "\n".join(p for p in parts if p)


def tool_call_dict(name: str, arguments: dict | None) -> dict:
    """Canonical OpenAI-shape tool call stored on manager_message.

    arguments stays a dict (not a JSON string): the chat template needs a
    mapping, and the trajectory manager matches history by dict equality, so a
    sampled leaf and its replayed echo compare equal regardless of key order.
    The wire-only tool-call id is dropped for the same reason.
    """
    return {"type": "function", "function": {"name": name, "arguments": arguments or {}}}


def manager_finish_reason(tool_uses: list[dict], raw_finish: str) -> str:
    """Finish reason stored on the manager turn: tool_calls if the turn called a
    tool, else the raw sglang finish."""
    return "tool_calls" if tool_uses else (raw_finish or "stop")


def _is_prompt_compaction(previous_tokens: int, current_tokens: int) -> bool:
    """Distinguish a real history compaction from minor prompt rewrites."""
    dropped_tokens = previous_tokens - current_tokens
    return (
        previous_tokens > 0
        and dropped_tokens >= _COMPACTION_MIN_DROP_TOKENS
        and dropped_tokens / previous_tokens >= _COMPACTION_MIN_DROP_FRACTION
    )


class BaseAdapter:
    """Base HTTP adapter: session lifecycle plus the shared one-turn pipeline.

    See the module docstring for the class attributes and hooks a subclass must
    supply; everything else is inherited.
    """

    logger: logging.Logger = logging.getLogger(__name__)
    log_prefix: str = "adapter"
    # body keys that cap max_new_tokens and carry stop sequences, in priority order
    max_token_keys: tuple[str, ...] = ()
    stop_keys: tuple[str, ...] = ()
    manager: Any

    def __init__(
        self,
        *,
        tokenizer,
        sglang_url,
        tool_parser=None,
        reasoning_parser=None,
        max_turns_per_sid: int | None = None,
        fork_threshold_tokens: int | None = None,
        debug_callback: Callable[..., None] | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.sglang_url = sglang_url.rstrip("/") if isinstance(sglang_url, str) else sglang_url
        self.tool_parser = tool_parser
        self.reasoning_parser = reasoning_parser
        self.store: dict[str, Any] = {}
        self.inflight: dict[str, set[asyncio.Task]] = {}
        self.closed: set[str] = set()
        self.app = web.Application(client_max_size=64 * 1024 * 1024)

        # one manager shared across all sids; per-sid trees live inside it.
        # fork_threshold_tokens left None means the manager uses its own default.
        mgr_kwargs: dict[str, int] = {}
        if fork_threshold_tokens is not None:
            mgr_kwargs["fork_threshold_tokens"] = fork_threshold_tokens
        self.manager = TrajectoryManager(**mgr_kwargs)

        self.debug_callback: Callable[..., None] | None = debug_callback
        # per-sid turn cap: return 429 to kill the run once exceeded
        self.max_turns_per_sid: int | None = max_turns_per_sid
        self._sid_turn_count: dict[str, int] = {}

        self.app.router.add_get("/healthz", _health)
        self.app.router.add_get("/v1/models", _health)
        self._register_routes(self.app)

    # -- wire hooks (subclass overrides) -------------------------------------

    def _register_routes(self, app: web.Application) -> None:
        """Register the protocol's POST route(s) and bind self._run_turn."""
        raise NotImplementedError

    def _session_id(self, request: web.Request, body: dict) -> str:
        raise NotImplementedError

    def _preprocess_body(self, body: dict) -> None:
        """Mutate the parsed body in place before sid resolution (default no-op)."""

    def _translate(self, body: dict) -> tuple[list[dict], list[dict] | None]:
        """Return (chat_messages, tools_schema) from the wire body."""
        raise NotImplementedError

    def _build_reply(self, parsed, raw_finish: str, translated: list[dict], tools_schema: list[dict] | None) -> Reply:
        """Pack parsed model output into a Reply."""
        raise NotImplementedError

    async def _respond(
        self,
        request: web.Request,
        body: dict,
        reply: Reply,
        in_tok: int,
        out_tok: int,
        stream: bool,
    ) -> web.StreamResponse:
        raise NotImplementedError

    async def _start_stream(
        self,
        request: web.Request,
        in_tok: int,
    ) -> web.StreamResponse | None:
        return None

    async def _stream_error(self, response: web.StreamResponse, error: Exception) -> web.StreamResponse:
        raise NotImplementedError

    async def _finish_stream(
        self,
        response: web.StreamResponse,
        reply: Reply,
        in_tok: int,
        out_tok: int,
    ) -> web.StreamResponse:
        raise NotImplementedError

    def _context_limit_response(self, error: ContextWindowExceeded) -> web.Response:
        return web.json_response(
            {
                "error": {
                    "type": "invalid_request_error",
                    "message": (
                        f"Prompt is too long: {error.prompt_tokens} input tokens plus "
                        f"{error.output_tokens} requested output tokens exceeds the "
                        f"{error.max_context_tokens} token context window"
                    ),
                }
            },
            status=400,
        )

    # -- session lifecycle ---------------------------------------------------

    def open_session(
        self,
        sid: str,
        *,
        sampling_defaults: dict | None = None,
        max_context_tokens: int = 0,
    ) -> None:
        """Register a fresh per-sid Session; sids must be unique."""
        if sid in self.store:
            raise ValueError(f"session_id {sid!r} already exists; sids must be unique per agent run")
        self.store[sid] = Session(
            sampling_defaults=dict(sampling_defaults or {}),
            max_context_tokens=int(max_context_tokens or 0),
        )

    async def shutdown_session(self, sid: str, *, wait_timeout: float = 5.0) -> None:
        """Mark a sid closed and drain its in-flight turn tasks."""
        self.closed.add(sid)
        tasks = [t for t in self.inflight.pop(sid, ()) if not t.done()]
        if not tasks:
            return

        async def _drain() -> None:
            _, pending = await asyncio.wait(tasks, timeout=wait_timeout)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        loop = tasks[0].get_loop()
        try:
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(_drain(), loop))
        except Exception:
            self.logger.exception("[%s] sid=%s shutdown drain failed", self.log_prefix, sid)

    def session_stats(self, sid: str) -> dict[str, int]:
        session = self.store.get(sid)
        return {
            "compaction_count": int(getattr(session, "compaction_count", 0) or 0),
            "max_prompt_tokens": int(getattr(session, "max_prompt_tokens", 0) or 0),
            "context_exceeded_count": int(getattr(session, "context_exceeded_count", 0) or 0),
        }

    async def finish_session(
        self,
        sid: str,
        *,
        base_sample,
        reward: float = 0.0,
        extra_metadata: dict | None = None,
        wait_timeout: float = 5.0,
    ) -> list:
        """Drain a session's trajectory into fully-formed Sample objects.

        Waits out in-flight requests for the sid, linearises the per-sid tree,
        then decodes each sample's trained tail into .response (the manager is
        tokenizer-free, so the adapter that owns the tokenizer fills this in).
        Idempotent: a second call for an already-popped sid returns [].
        """
        await self.shutdown_session(sid, wait_timeout=wait_timeout)
        metadata = {**(extra_metadata or {}), **self.session_stats(sid)}
        session = self.store.pop(sid, None)
        max_sample_tokens = int(getattr(session, "max_context_tokens", 0) or 0) if session is not None else 0
        samples = self.manager.get_trajectory(
            sid,
            base_sample=base_sample,
            reward=reward,
            extra_metadata=metadata,
            max_sample_tokens=max_sample_tokens,
        )
        for s in samples:
            rlen = int(s.response_length or 0)
            s.response = (
                self.tokenizer.decode(s.tokens[-rlen:], skip_special_tokens=False) if rlen and s.tokens else ""
            )
        return samples

    async def drop_session(self, sid: str, *, wait_timeout: float = 5.0) -> None:
        await self.shutdown_session(sid, wait_timeout=wait_timeout)
        self.store.pop(sid, None)
        self.manager.drop_session(sid)

    # -- shared request pipeline ---------------------------------------------

    def _check_turn_cap(self, sid: str) -> web.Response | None:
        """Enforce max_turns_per_sid, returning a 429 response once exceeded.

        Increments the per-sid counter as a side effect when under the cap.
        """
        cap = self.max_turns_per_sid
        if cap is None:
            return None
        prior = self._sid_turn_count.get(sid, 0)
        if prior >= cap:
            self.logger.warning("[%s] sid=%s exceeded max_turns_per_sid=%d; killing run", self.log_prefix, sid, cap)
            return web.json_response(
                {
                    "error": {
                        "type": "rate_limit_error",
                        "message": (f"adapter: sid {sid!r} exceeded max_turns_per_sid={cap}; killing run"),
                    }
                },
                status=429,
            )
        self._sid_turn_count[sid] = prior + 1
        return None

    def _run_debug_callback(self, sid, translated, tools_schema, manager_message, turn) -> None:
        """Run the optional debug-only data dump callback; unset in production."""
        callback = self.debug_callback
        if callback is None:
            return
        try:
            callback(sid, translated, tools_schema, manager_message, turn)
        except Exception:
            self.logger.exception("debug_callback failed (sid=%s)", sid)

    def _prepare_prompt(self, body: dict) -> tuple[list[dict], list[dict] | None, list[int]]:
        """Translate and render one wire request with the served chat template."""
        translated, tools_schema = self._translate(body)
        prompt_ids = _render_token_ids(
            translated,
            self.tokenizer,
            tools=tools_schema,
            add_generation_prompt=True,
        )
        return translated, tools_schema, prompt_ids

    async def _run_turn(self, request: web.Request) -> web.StreamResponse:
        """One full agent turn: translate -> sglang -> parse -> append -> respond.

        The wire-specific steps are delegated to the subclass hooks; the rest
        (sid resolution, closed/cap guards, inflight tracking, record_turn) is
        shared across protocols.
        """
        body = await request.json()
        self._preprocess_body(body)
        sid = self._session_id(request, body)
        if sid in self.closed:  # session drained; refuse stragglers
            self.logger.debug("[%s] sid=%s request after session closed", self.log_prefix, sid)
            return web.Response(status=503, text="session closed")
        capped = self._check_turn_cap(sid)
        if capped is not None:
            return capped

        tok = self.tokenizer
        s = self.store.setdefault(sid, Session())
        task = asyncio.current_task()
        self.inflight.setdefault(sid, set()).add(task)
        t0 = time.monotonic()
        try:
            translated, tools_schema, prompt_ids = self._prepare_prompt(body)
            prompt_tokens = len(prompt_ids)
            if _is_prompt_compaction(s.last_prompt_tokens, prompt_tokens):
                s.compaction_count += 1
                self.logger.info(
                    "[%s] sid=%s prompt_compaction_detected previous_tokens=%d current_tokens=%d count=%d",
                    self.log_prefix,
                    sid,
                    s.last_prompt_tokens,
                    prompt_tokens,
                    s.compaction_count,
                )
            s.last_prompt_tokens = prompt_tokens
            s.max_prompt_tokens = max(s.max_prompt_tokens, prompt_tokens)

            stream = body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", "")
            stream_response = await self._start_stream(request, prompt_tokens) if stream else None

            try:
                turn = await call_sglang_generate(prompt_ids, s, body, adapter=self, session_id=sid)
            except ContextWindowExceeded as error:
                if stream_response is not None:
                    return await self._stream_error(stream_response, error)
                return self._context_limit_response(error)
            except ConnectionResetError:
                raise
            except Exception as error:
                if stream_response is not None:
                    return await self._stream_error(stream_response, error)
                raise

            raw_output = tok.decode(turn.output_ids, skip_special_tokens=False) if turn.output_ids else ""
            parsed = parse_model_output(
                raw_output,
                tools_schema=tools_schema,
                tool_parser_name=self.tool_parser,
                reasoning_parser_name=self.reasoning_parser,
            )
            reply = self._build_reply(parsed, turn.finish_reason, translated, tools_schema)
            turn = dataclasses.replace(turn, ill_formed=parsed.ill_formed)

            in_tok, out_tok = len(prompt_ids), len(turn.output_ids)
            # Flush the response before recording the trajectory: a client that
            # disconnected during generation makes _respond raise here, and we
            # must not record a turn the client never received.
            try:
                if stream_response is not None:
                    response = await self._finish_stream(stream_response, reply, in_tok, out_tok)
                else:
                    response = await self._respond(request, body, reply, in_tok, out_tok, stream)
            except (ConnectionResetError, asyncio.CancelledError) as e:
                self.logger.warning(
                    "[%s] sid=%s client disconnected before response flush: %s after %.1fs",
                    self.log_prefix,
                    sid,
                    type(e).__name__,
                    time.monotonic() - t0,
                )
                if isinstance(e, asyncio.CancelledError):
                    raise
                return web.Response(status=499, text="client disconnected")

            self._run_debug_callback(
                sid,
                translated,
                tools_schema,
                reply.manager_message,
                turn,
            )

            self.manager.record_turn(
                sid,
                turn=turn,
                prompt_messages=translated,
                response_message=reply.manager_message,
                metadata={"sid": sid},
            )
            return response
        finally:
            self.inflight.get(sid, set()).discard(task)


def sid_from_bearer(request: web.Request) -> str | None:
    """sid from the Authorization: Bearer <sid> header, or None if absent."""
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return None


def sid_from_body(body: dict | None) -> str | None:
    """sid from the OpenAI-shape body (metadata.session_id / user), or None."""
    if not body:
        return None
    metadata = body.get("metadata")
    if isinstance(metadata, dict) and metadata.get("session_id"):
        return str(metadata["session_id"])
    if body.get("user"):
        return str(body["user"])
    return None


def _sampling_params(session: Any, body: dict, *, max_token_keys: tuple[str, ...], stop_keys: tuple[str, ...]) -> dict:
    sp: dict[str, Any] = {
        "skip_special_tokens": False,
        "spaces_between_special_tokens": False,
        "no_stop_trim": True,
        "max_new_tokens": 4096,
        **(session.sampling_defaults or {}),
    }

    for key in max_token_keys:
        if body.get(key) is not None:
            sp["max_new_tokens"] = min(int(sp.get("max_new_tokens", body[key])), int(body[key]))
            break

    for src_k, dst_k in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k")):
        if src_k in body:
            sp[dst_k] = body[src_k]

    for key in stop_keys:
        if body.get(key):
            sp["stop"] = body[key]
            break

    return sp


async def _abort_sglang_request(sglang_url: str, rid: str) -> int:
    timeout = aiohttp.ClientTimeout(total=5)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{sglang_url}/abort_request", json={"rid": rid}) as response:
            await response.read()
            return response.status


async def call_sglang_generate(
    prompt_ids: list[int],
    session: Any,
    body: dict,
    *,
    adapter: BaseAdapter,
    session_id: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TurnRecord:
    """POST one turn to SGLang and capture its terminal token metadata."""
    logger = adapter.logger
    sp = _sampling_params(session, body, max_token_keys=adapter.max_token_keys, stop_keys=adapter.stop_keys)

    if session.max_context_tokens > 0:
        output_tokens = int(sp.get("max_new_tokens", 0) or 0)
        if len(prompt_ids) + output_tokens > session.max_context_tokens:
            session.context_exceeded_count = int(getattr(session, "context_exceeded_count", 0) or 0) + 1
            logger.warning(
                "[%s] sid=%s insufficient context budget " "prompt_tokens=%d output_budget=%d max_context_tokens=%d",
                adapter.log_prefix,
                session_id,
                len(prompt_ids),
                output_tokens,
                session.max_context_tokens,
            )
            raise ContextWindowExceeded(
                prompt_tokens=len(prompt_ids),
                output_tokens=output_tokens,
                max_context_tokens=session.max_context_tokens,
            )

    sglang_url = adapter.sglang_url
    rid = uuid.uuid4().hex
    request_started = time.monotonic()
    logger.debug(
        "[%s] sid=%s rid=%s stage=adapter event=sglang_request_start router_url=%s "
        "prompt_tokens=%d max_new_tokens=%s",
        adapter.log_prefix,
        session_id,
        rid,
        sglang_url,
        len(prompt_ids),
        sp.get("max_new_tokens"),
    )
    headers = {"X-Request-ID": rid}
    if session_id and session_id != "default":
        headers["X-SMG-Routing-Key"] = session_id
    payload = {
        "rid": rid,
        "input_ids": prompt_ids,
        "sampling_params": sp,
        "return_logprob": True,
    }
    if progress_callback is not None:
        payload["stream"] = True

    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    first_chunk_ms: float | None = None
    stream_chunks = 0
    response_status = 0
    data: dict[str, Any] | None = None
    try:
        async with aiohttp.ClientSession(timeout=timeout) as client, client.post(
            f"{sglang_url}/generate", json=payload, headers=headers
        ) as response:
            response_status = response.status
            if response.status >= 400:
                text = await response.text()
                upstream_worker_url = _UPSTREAM_GENERATE_URL_RE.search(text)
                logger.warning(
                    "[%s] sid=%s rid=%s stage=adapter event=sglang_request_end "
                    "outcome=http_error router_url=%s upstream_worker_url=%s http_status=%d duration_ms=%.1f "
                    "error_type=upstream_http sglang upstream %d: %.200s",
                    adapter.log_prefix,
                    session_id,
                    rid,
                    sglang_url,
                    upstream_worker_url.group(0).removesuffix("/generate") if upstream_worker_url else "-",
                    response.status,
                    (time.monotonic() - request_started) * 1000,
                    response.status,
                    text,
                )
                raise RuntimeError(f"sglang upstream {response.status}: {text[:400]}")

            if progress_callback is None:
                data = await response.json(content_type=None)
            else:
                previous_pairs: tuple[tuple[float, int], ...] = ()
                previous_text = ""
                terminal_seen = False
                async for raw_line in response.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    encoded = line.removeprefix("data:").strip()
                    if not encoded or encoded == "[DONE]":
                        continue
                    if terminal_seen:
                        raise aiohttp.ClientPayloadError("sglang stream emitted data after its terminal chunk")
                    try:
                        chunk = json.loads(encoded)
                    except json.JSONDecodeError as error:
                        raise aiohttp.ClientPayloadError(f"invalid sglang SSE chunk: {encoded[:200]}") from error
                    if not isinstance(chunk, dict):
                        raise aiohttp.ClientPayloadError("sglang SSE chunk must be a JSON object")
                    if chunk.get("error"):
                        raise aiohttp.ClientPayloadError(f"sglang stream error: {str(chunk['error'])[:200]}")
                    text = chunk.get("text")
                    if not isinstance(text, str):
                        raise aiohttp.ClientPayloadError("sglang SSE chunk text must be a string")
                    if len(text) < len(previous_text):
                        raise aiohttp.ClientPayloadError("sglang cumulative text length rolled back")

                    meta = chunk.get("meta_info") or {}
                    raw_pairs = meta.get("output_token_logprobs") or []
                    try:
                        pairs = tuple((float(pair[0]), int(pair[1])) for pair in raw_pairs)
                    except (IndexError, TypeError, ValueError) as error:
                        raise aiohttp.ClientPayloadError("invalid cumulative output token logprobs") from error
                    if len(pairs) < len(previous_pairs) or pairs[: len(previous_pairs)] != previous_pairs:
                        raise aiohttp.ClientPayloadError("sglang cumulative output token logprobs were revised")

                    finish_info = meta.get("finish_reason") or {}
                    finish_reason = finish_info.get("type") if isinstance(finish_info, dict) else None
                    finish_reason = str(finish_reason) if finish_reason else None
                    stream_chunks += 1
                    if first_chunk_ms is None:
                        first_chunk_ms = (time.monotonic() - request_started) * 1000
                    await progress_callback(
                        GenerationProgress(
                            rid=rid,
                            chunk_index=stream_chunks,
                            text=text,
                            output_token_logprobs=pairs,
                            finish_reason=finish_reason,
                        )
                    )
                    previous_pairs = pairs
                    previous_text = text
                    if finish_reason is not None:
                        terminal_seen = True
                        data = chunk

                if data is None:
                    raise aiohttp.ClientPayloadError("sglang stream ended without a terminal chunk")
    except asyncio.CancelledError as error:
        await _abort_after_failure(adapter, session_id, rid, sglang_url, request_started, error)
        raise
    except Exception as error:
        await _abort_after_failure(adapter, session_id, rid, sglang_url, request_started, error)
        raise

    if not isinstance(data, dict):
        raise aiohttp.ClientPayloadError("sglang response must be a JSON object")
    meta = data.get("meta_info") or {}
    finish_info = meta.get("finish_reason") or {}
    finish = finish_info.get("type") if isinstance(finish_info, dict) else None
    if not finish:
        raise aiohttp.ClientPayloadError("sglang response did not contain a finish reason")
    try:
        output_token_logprobs = tuple(
            (float(pair[0]), int(pair[1])) for pair in (meta.get("output_token_logprobs") or [])
        )
    except (IndexError, TypeError, ValueError) as error:
        raise aiohttp.ClientPayloadError("invalid terminal output token logprobs") from error
    output_ids = [token_id for _, token_id in output_token_logprobs]
    output_log_probs = [logprob for logprob, _ in output_token_logprobs]
    logger.debug(
        "[%s] sid=%s rid=%s stage=adapter event=sglang_request_end outcome=success "
        "router_url=%s http_status=%d duration_ms=%.1f first_chunk_ms=%.1f "
        "stream_chunks=%d output_tokens=%d",
        adapter.log_prefix,
        session_id,
        rid,
        sglang_url,
        response_status,
        (time.monotonic() - request_started) * 1000,
        first_chunk_ms or 0.0,
        stream_chunks,
        len(output_ids),
    )
    return TurnRecord(
        prompt_ids=list(prompt_ids),
        output_ids=output_ids,
        finish_reason=str(finish),
        output_log_probs=output_log_probs,
        cached_tokens=int(meta.get("cached_tokens", 0) or 0),
        prompt_tokens=int(meta.get("prompt_tokens", len(prompt_ids)) or len(prompt_ids)),
    )


async def _abort_after_failure(
    adapter: BaseAdapter,
    session_id: str | None,
    rid: str,
    sglang_url: str,
    request_started: float,
    error: BaseException,
) -> None:
    adapter.logger.warning(
        "[%s] sid=%s rid=%s stage=adapter event=sglang_request_failed outcome=exception "
        "router_url=%s duration_ms=%.1f exception_type=%s error=%s",
        adapter.log_prefix,
        session_id,
        rid,
        sglang_url,
        (time.monotonic() - request_started) * 1000,
        type(error).__name__,
        str(error)[:200],
    )
    try:
        status = await asyncio.shield(_abort_sglang_request(sglang_url, rid))
        adapter.logger.debug(
            "[%s] sid=%s rid=%s stage=adapter event=abort_request outcome=completed " "router_url=%s http_status=%d",
            adapter.log_prefix,
            session_id,
            rid,
            sglang_url,
            status,
        )
    except Exception as abort_error:
        adapter.logger.warning(
            "[%s] sid=%s rid=%s stage=adapter event=abort_request outcome=exception "
            "router_url=%s exception_type=%s error=%s",
            adapter.log_prefix,
            session_id,
            rid,
            sglang_url,
            type(abort_error).__name__,
            str(abort_error)[:200],
        )


async def _health(request: web.Request) -> web.Response:
    """Handler for /healthz and /v1/models readiness probes."""
    return web.json_response({"ok": True})
