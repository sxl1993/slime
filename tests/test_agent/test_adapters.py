"""Unit tests for the agent HTTP adapters (Anthropic + OpenAI) and parsing.

Every test drives a REAL adapter over a real aiohttp loopback
(``TestServer``/``TestClient``) and a real ``/generate`` upstream
(:class:`tests.test_agent._fakes.FakeSGLangServer`) -- so the whole
translate -> sglang -> parse -> record_turn -> finish_session path runs
unmocked; only the model server and tokenizer are faked. Covers both wire
protocols plus the standalone parsing helpers in ``slime.agent.parsing``.

Replaces the pre-refactor ``tests/test_agent_adapters.py`` (which imported now-
removed symbols and a dropped ``/v1/responses`` endpoint).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_agent._fakes import FakeSGLangServer, FakeTokenizer  # noqa: E402

from slime.agent.adapters import anthropic, common, openai  # noqa: E402
from slime.agent.parsing import parse_model_output, parse_xml_tool_uses  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

NUM_GPUS = 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _Headers:
    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


def _parse_sse(raw: str) -> list[tuple[str, object]]:
    """Parse an SSE byte-stream body into ``(event_name, payload)`` pairs;
    ``payload`` is the decoded JSON, or the literal ``"[DONE]"``."""
    events: list[tuple[str, object]] = []
    event_name = "message"
    data_lines: list[str] = []

    def flush() -> None:
        nonlocal event_name, data_lines
        if data_lines:
            data = "\n".join(data_lines)
            events.append((event_name, data if data == "[DONE]" else json.loads(data)))
        event_name = "message"
        data_lines = []

    for line in raw.splitlines():
        if not line:
            flush()
        elif line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
        elif line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
    flush()
    return events


async def _drain(adapter, sid) -> list[Sample]:
    return await adapter.finish_session(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)


# ===========================================================================
# §1 session-id resolution
# ===========================================================================


def test_anthropic_session_id_prefers_bearer_then_api_key():
    assert anthropic._request_session_id(_Headers({"X-Api-Key": "key"})) == "key"
    assert anthropic._request_session_id(_Headers({"Authorization": "Bearer bsid", "X-Api-Key": "key"})) == "bsid"
    assert anthropic._request_session_id(_Headers({})) == "default"


def test_openai_session_id_prefers_bearer_then_body():
    req = _Headers({})
    assert openai._request_session_id(req, {"metadata": {"session_id": "meta"}, "user": "u"}) == "meta"
    assert openai._request_session_id(req, {"user": "u"}) == "u"
    assert openai._request_session_id(_Headers({"Authorization": "Bearer bsid"}), {"user": "u"}) == "bsid"
    assert openai._request_session_id(req, {}) == "default"


# ===========================================================================
# §2 translation (wire -> chat-template messages)
# ===========================================================================


def test_anthropic_translation_keeps_tool_results_thinking_and_tools():
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "plan"},
                {"type": "text", "text": "ok"},
                {"type": "tool_use", "name": "lookup", "input": {"q": "slime"}},
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "u1", "content": "result"}]},
    ]
    translated = anthropic._translate_messages(messages, system="sys")
    assert translated == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "ok",
            "reasoning_content": "plan",
            "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {"q": "slime"}}}],
        },
        {"role": "tool", "content": "result"},
    ]
    tools = anthropic._tools_to_chat_tools(
        [{"name": "lookup", "description": "search", "input_schema": {"type": "object"}}]
    )
    assert tools == [
        {"type": "function", "function": {"name": "lookup", "description": "search", "parameters": {"type": "object"}}}
    ]


def test_openai_translation_developer_to_system_and_tool_calls_to_dict():
    translated = openai._translate_messages(
        [
            {"role": "developer", "content": "rules"},
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": '{"q": "slime"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "found"},
        ]
    )
    assert translated == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hello"},
        # wire-only id dropped; arguments coerced JSON-string -> dict.
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "lookup", "arguments": {"q": "slime"}}}],
        },
        # tool_call_id dropped.
        {"role": "tool", "content": "found"},
    ]


# ===========================================================================
# §3 non-stream JSON + token capture (real HTTP, real /generate)
# ===========================================================================


def test_anthropic_messages_nonstream_records_token_segments():
    async def run_case():
        async with FakeSGLangServer([[(-0.1, 101), (-0.2, 102)]]) as sglang:
            tok = FakeTokenizer(outputs={(101, 102): "done now"})
            adapter = anthropic.AnthropicAdapter(tokenizer=tok, sglang_url=sglang.url)
            adapter.open_session("sid-a")
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                resp = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer sid-a"},
                    json={"model": "m", "max_tokens": 7, "messages": [{"role": "user", "content": "hi"}]},
                )
                data = await resp.json()
            finally:
                await client.close()
            samples = await _drain(adapter, "sid-a")

        assert resp.status == 200
        assert data["type"] == "message" and data["stop_reason"] == "end_turn"
        assert data["content"] == [{"type": "text", "text": "done now"}]
        # adapter posted the rendered prompt ids and capped max_new_tokens at max_tokens.
        assert sglang.requests[0]["sampling_params"]["max_new_tokens"] == 7
        assert sglang.routing_keys == ["sid-a"]
        # one trained turn: the two response ids carry loss=1 + real logprobs.
        assert len(samples) == 1
        s = samples[0]
        assert s.tokens[-2:] == [101, 102]
        assert s.loss_mask[-2:] == [1, 1]
        assert s.rollout_log_probs[-2:] == [-0.1, -0.2]
        assert s.response == "done now"

    asyncio.run(run_case())


def test_openai_chat_completions_nonstream_records_token_segments():
    async def run_case():
        async with FakeSGLangServer([[(-0.3, 201)]]) as sglang:
            tok = FakeTokenizer(outputs={(201,): "hello"})
            adapter = openai.OpenAIAdapter(tokenizer=tok, sglang_url=sglang.url)
            adapter.open_session("sid-o")
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                resp = await client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer sid-o"},
                    json={"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": "hi?"}]},
                )
                data = await resp.json()
            finally:
                await client.close()
            samples = await _drain(adapter, "sid-o")

        assert resp.status == 200
        assert data["object"] == "chat.completion"
        assert data["choices"][0]["message"] == {"role": "assistant", "content": "hello"}
        assert data["choices"][0]["finish_reason"] == "stop"
        assert sglang.requests[0]["sampling_params"]["max_new_tokens"] == 4
        assert len(samples) == 1 and samples[0].tokens[-1] == 201 and samples[0].loss_mask[-1] == 1

    asyncio.run(run_case())


# ===========================================================================
# §4 streaming SSE
# ===========================================================================


def test_anthropic_messages_streams_blocks():
    async def run_case():
        async with FakeSGLangServer([[(-0.1, 301)]]) as sglang:
            tok = FakeTokenizer(outputs={(301,): "streamed"})
            adapter = anthropic.AnthropicAdapter(tokenizer=tok, sglang_url=sglang.url)
            adapter.open_session("sid-as")
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                resp = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer sid-as", "Accept": "text/event-stream"},
                    json={
                        "model": "m",
                        "stream": True,
                        "max_tokens": 8,
                        "messages": [{"role": "user", "content": "x"}],
                    },
                )
                raw = await resp.text()
            finally:
                await client.close()
            await _drain(adapter, "sid-as")

        names = [name for name, _ in _parse_sse(raw)]
        assert names[0] == "message_start"
        assert "content_block_delta" in names
        assert names[-1] == "message_stop"
        deltas = [p for n, p in _parse_sse(raw) if n == "content_block_delta"]
        assert any(d["delta"].get("text") == "streamed" for d in deltas)

    asyncio.run(run_case())


def test_openai_chat_completions_streams_chunks_until_done():
    async def run_case():
        async with FakeSGLangServer([[(-0.1, 401)]]) as sglang:
            tok = FakeTokenizer(outputs={(401,): "streamed text"})
            adapter = openai.OpenAIAdapter(tokenizer=tok, sglang_url=sglang.url)
            adapter.open_session("sid-os")
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                resp = await client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": "Bearer sid-os"},
                    json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "x"}]},
                )
                raw = await resp.text()
            finally:
                await client.close()
            await _drain(adapter, "sid-os")

        events = _parse_sse(raw)
        chunks = [p for _, p in events if isinstance(p, dict)]
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert events[-1] == ("message", "[DONE]")

    asyncio.run(run_case())


# ===========================================================================
# §5 multi-turn token alignment (tool call -> tool result -> answer)
# ===========================================================================


def test_anthropic_multiturn_wire_roundtrip_and_token_capture():
    """Two-turn round-trip: a tool-call turn, then a tool-result + answer turn.

    Asserts the adapter-level behaviour the branching test can't see: the wire
    tool_use block round-trips, both turns route to the same sid, and
    finish_session yields aligned training samples. (The fine-grained
    clean/drift linearization is owned by test_trajectory_manager_branching.)"""

    async def run_case():
        r1 = "<tool_call><function=lookup><parameter=query>slime</parameter></function></tool_call>"
        async with FakeSGLangServer([[(-0.5, 700), (-0.5, 701)], [(-0.4, 800)]]) as sglang:
            tok = FakeTokenizer(outputs={(700, 701): r1, (800,): "the answer"})
            adapter = anthropic.AnthropicAdapter(tokenizer=tok, sglang_url=sglang.url)
            adapter.open_session("sid-mt")
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            tools = [
                {"name": "lookup", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}}
            ]
            try:
                first = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer sid-mt"},
                    json={
                        "model": "m",
                        "max_tokens": 5,
                        "tools": tools,
                        "messages": [{"role": "user", "content": [{"type": "text", "text": "find slime"}]}],
                    },
                )
                fdata = await first.json()
                tool_use = next(b for b in fdata["content"] if b["type"] == "tool_use")
                second = await client.post(
                    "/v1/messages",
                    headers={"Authorization": "Bearer sid-mt"},
                    json={
                        "model": "m",
                        "max_tokens": 7,
                        "tools": tools,
                        "messages": [
                            {"role": "user", "content": [{"type": "text", "text": "find slime"}]},
                            {"role": "assistant", "content": fdata["content"]},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "tool_result", "tool_use_id": tool_use["id"], "content": "found"}
                                ],
                            },
                        ],
                    },
                )
                await second.json()
            finally:
                await client.close()
            samples = await _drain(adapter, "sid-mt")

        assert first.status == 200 and second.status == 200
        assert fdata["stop_reason"] == "tool_use"
        assert tool_use["name"] == "lookup" and tool_use["input"] == {"query": "slime"}
        # both turns routed to the same sid; the adapter posted the growing prompt.
        assert sglang.routing_keys == ["sid-mt", "sid-mt"]
        assert (
            sglang.requests[1]["input_ids"][: len(sglang.requests[0]["input_ids"])] == sglang.requests[0]["input_ids"]
        )
        # finish_session produces at least one aligned, partly-trained sample.
        assert samples
        for s in samples:
            assert len(s.loss_mask) == len(s.rollout_log_probs) == s.response_length
            assert sum(s.loss_mask) > 0

    asyncio.run(run_case())


# ===========================================================================
# §6 adapter behaviour: turn cap, mid-list system fold
# ===========================================================================


def test_max_turns_per_sid_returns_429():
    async def run_case():
        async with FakeSGLangServer([[(-0.1, 501)], [(-0.1, 502)]]) as sglang:
            tok = FakeTokenizer()
            adapter = anthropic.AnthropicAdapter(tokenizer=tok, sglang_url=sglang.url, max_turns_per_sid=1)
            adapter.open_session("sid-cap")
            client = TestClient(TestServer(adapter.app))
            await client.start_server()
            try:
                body = {"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": "x"}]}
                h = {"Authorization": "Bearer sid-cap"}
                first = await client.post("/v1/messages", headers=h, json=body)
                second = await client.post("/v1/messages", headers=h, json=body)
            finally:
                await client.close()
            await _drain(adapter, "sid-cap")
        assert first.status == 200
        assert second.status == 429

    asyncio.run(run_case())


def test_mid_list_system_folds_into_user():
    body = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "skills list"},
            {"role": "user", "content": "next"},
        ]
    }
    changed = anthropic._fold_mid_list_system_into_user(body)
    assert changed
    # the mid-list system message is gone; its text is wrapped into the prior user.
    assert [m["role"] for m in body["messages"]] == ["user", "user"]
    folded = body["messages"][0]["content"]
    assert any(b.get("text", "").startswith("<system-reminder>") for b in folded)


# ===========================================================================
# §6b adapter context compression
# ===========================================================================


def test_replace_system_prompt_swaps_long_system():
    body = {"system": "A" * 70_000, "messages": [{"role": "user", "content": "hi"}]}
    common._replace_system_prompt(body)
    assert body["system"] == common._COMPACT_SYSTEM_PROMPT
    assert len(body["system"]) < 1_000


def test_replace_system_prompt_passthrough_when_env_empty(monkeypatch):
    monkeypatch.setenv("SLIME_ADAPTER_SYSTEM_PROMPT", "")
    body = {"system": "original", "messages": []}
    common._replace_system_prompt(body)
    assert body["system"] == "original"


def test_replace_system_prompt_uses_custom_env(monkeypatch):
    monkeypatch.setenv("SLIME_ADAPTER_SYSTEM_PROMPT", "custom prompt")
    body = {"system": "original", "messages": []}
    common._replace_system_prompt(body)
    assert body["system"] == "custom prompt"


def test_filter_tools_keeps_whitelist_only():
    body = {
        "tools": [
            {"name": "Bash", "input_schema": {}},
            {"name": "Read", "input_schema": {}},
            {"name": "Agent", "input_schema": {}},
            {"name": "Workflow", "input_schema": {}},
            {"name": "Edit", "input_schema": {}},
        ],
        "messages": [],
    }
    common._filter_tools(body)
    assert [t["name"] for t in body["tools"]] == ["Bash", "Read", "Edit"]


def test_filter_tools_passthrough_when_env_empty(monkeypatch):
    monkeypatch.setenv("SLIME_ADAPTER_TOOL_WHITELIST", "")
    body = {
        "tools": [{"name": "Bash"}, {"name": "Agent"}],
        "messages": [],
    }
    common._filter_tools(body)
    assert len(body["tools"]) == 2


def test_filter_tools_custom_whitelist(monkeypatch):
    monkeypatch.setenv("SLIME_ADAPTER_TOOL_WHITELIST", "Bash,Read,Grep")
    body = {
        "tools": [{"name": "Bash"}, {"name": "Edit"}, {"name": "Grep"}],
        "messages": [],
    }
    common._filter_tools(body)
    assert [t["name"] for t in body["tools"]] == ["Bash", "Grep"]


def test_filter_tools_no_tools_key_is_noop():
    body = {"messages": []}
    common._filter_tools(body)  # no crash


def test_truncate_tool_results_clips_long_content():
    long_text = "x" * 20_000
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": long_text},
                ],
            },
        ]
    }
    common._truncate_tool_results(body)
    result = body["messages"][0]["content"][0]["content"]
    assert len(result) < 15_000  # 10k + clipping message
    assert "<response clipped" in result
    assert "10,000 chars" in result


def test_truncate_tool_results_short_content_untouched():
    short_text = "hello"
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": short_text},
                ],
            },
        ]
    }
    common._truncate_tool_results(body)
    assert body["messages"][0]["content"][0]["content"] == short_text


def test_truncate_tool_results_passthrough_when_env_zero(monkeypatch):
    monkeypatch.setenv("SLIME_ADAPTER_MAX_TOOL_RESULT_CHARS", "0")
    long_text = "y" * 20_000
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": long_text},
                ],
            },
        ]
    }
    common._truncate_tool_results(body)
    assert len(body["messages"][0]["content"][0]["content"]) == 20_000


def test_truncate_tool_results_custom_limit(monkeypatch):
    monkeypatch.setenv("SLIME_ADAPTER_MAX_TOOL_RESULT_CHARS", "100")
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "z" * 200},
                ],
            },
        ]
    }
    common._truncate_tool_results(body)
    result = body["messages"][0]["content"][0]["content"]
    assert "<response clipped" in result
    assert "100 chars" in result


def test_truncate_tool_results_non_user_messages_untouched():
    body = {
        "messages": [
            {"role": "assistant", "content": "some very long content " * 1000},
        ]
    }
    common._truncate_tool_results(body)
    assert len(body["messages"][0]["content"]) > 10_000


# ===========================================================================
# §7 parsing helpers (slime.agent.parsing)
# ===========================================================================


def test_parse_model_output_plain_text_no_parsers():
    parsed = parse_model_output("just text", tools_schema=None, tool_parser_name=None, reasoning_parser_name=None)
    assert parsed.text == "just text"
    assert parsed.tool_uses == []
    assert parsed.reasoning == ""


def test_parse_model_output_think_split_fallback():
    # The qwen3 reasoning parser lives in sglang (lazy import); skip where the
    # lean CPU CI env has no sglang installed.
    pytest.importorskip("sglang")
    parsed = parse_model_output(
        "<think>reason here</think>visible",
        tools_schema=None,
        tool_parser_name=None,
        reasoning_parser_name="qwen3",
    )
    # the qwen3 reasoning parser (or the </think> fallback) splits reasoning out.
    assert "visible" in parsed.text
    assert "reason here" in parsed.reasoning


def test_parse_xml_tool_uses_fallback():
    raw = "lead <tool_call><function=lookup><parameter=q>slime</parameter></function></tool_call> tail"
    schema = [{"function": {"name": "lookup"}}]
    cleaned, uses = parse_xml_tool_uses(raw, schema)
    assert uses == [{"name": "lookup", "input": {"q": "slime"}}]
    assert "<tool_call>" not in cleaned
    assert "lead" in cleaned and "tail" in cleaned


def test_parse_xml_tool_uses_ignores_unknown_tool():
    raw = "<tool_call><function=unknown><parameter=q>x</parameter></function></tool_call>"
    cleaned, uses = parse_xml_tool_uses(raw, [{"function": {"name": "lookup"}}])
    assert uses == []
    assert "<tool_call>" in cleaned  # left untouched


# ===========================================================================
# §8 call_sglang_generate context-length guard
# ===========================================================================


class _StubAdapter:
    """Minimal adapter stand-in that ``call_sglang_generate`` reads."""

    log_prefix = "test"
    max_token_keys = ["max_tokens"]
    stop_keys = ["stop_sequences"]

    def __init__(self, sglang_url: str):
        self.sglang_url = sglang_url
        self.logger = _StubLogger()
        self._http: aiohttp.ClientSession | None = None

    def _get_http_session(self):
        import aiohttp as _aiohttp

        if self._http is None or self._http.closed:
            self._http = _aiohttp.ClientSession()
        return self._http

    async def close(self):
        if self._http and not self._http.closed:
            await self._http.close()


class _StubLogger:
    def __getattr__(self, name):
        def _log(*a, **kw):
            pass

        return _log


@pytest.mark.asyncio
async def test_sglang_context_reserve_prevents_overlength():
    """Adapter-side reserve should clamp prompt_ids before they reach SGLang."""
    from slime.agent.adapters.common import Session, _SGLANG_CONTEXT_RESERVE, call_sglang_generate

    adapter = _StubAdapter("http://127.0.0.1:1")  # unreachable — should not be called
    session = Session(max_context_tokens=100)
    # prompt_ids at exactly (max_context_tokens - reserve) should still pass
    prompt_ids = list(range(100 - _SGLANG_CONTEXT_RESERVE))
    # We can't actually hit SGLang in a unit test, but we can verify the
    # remaining_context calculation: the reserve is subtracted before comparison.
    effective_max = session.max_context_tokens - _SGLANG_CONTEXT_RESERVE
    assert len(prompt_ids) == effective_max
    # A prompt one token longer should trigger the length guard
    long_prompt = list(range(effective_max + 1))
    turn = await call_sglang_generate(long_prompt, session, {}, adapter=adapter, session_id="s1")
    assert turn.finish_reason == "length"
    assert turn.output_ids == []
    await adapter.close()


@pytest.mark.asyncio
async def test_sglang_400_context_length_returns_length_not_error():
    """When SGLang returns 400 for context-length, adapter returns length-capped turn."""
    from aiohttp import web as aio_web

    from slime.agent.adapters.common import Session, call_sglang_generate

    # Spin up a tiny aiohttp server that always returns 400 with the
    # SGLang context-length error payload.
    async def _handler(request):
        return aio_web.Response(
            status=400,
            text='{"error":{"message":"Input length (31996 tokens) exceeds the maximum allowed length (31994 tokens). Use a shorter input or enable --allow-auto-truncate."}}',
            content_type="application/json",
        )

    app = aio_web.Application()
    app.router.add_post("/generate", _handler)
    runner = aio_web.AppRunner(app)
    await runner.setup()
    site = aio_web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]

    adapter = _StubAdapter(f"http://127.0.0.1:{port}")
    session = Session(max_context_tokens=0)  # 0 = no adapter-side cap, rely on SGLang 400 fallback
    turn = await call_sglang_generate(
        prompt_ids=list(range(10)),
        session=session,
        body={},
        adapter=adapter,
        session_id="s2",
    )
    assert turn.finish_reason == "length"
    assert turn.output_ids == []
    await runner.cleanup()
    await adapter.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
