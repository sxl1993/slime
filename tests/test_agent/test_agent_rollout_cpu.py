"""CPU-only agent rollout test: the whole pipeline, no GPU / E2B / sglang.

This walks a real agent rollout end to end on CPU. Only four external edges are
faked (see ``tests/test_agent/_fakes.py``): the tokenizer, the E2B sandbox, the
sglang ``/generate`` server, and the agent CLI process inside the sandbox.
Everything between -- ``generate.generate`` orchestration, the in-thread adapter
HTTP app, wire translation, ``record_turn`` / tree building, ``finish_session``
linearization, ``swe`` workspace-prep / diff / eval, the harness lifecycle and
its detached-launch transport -- is the real code.

The "agent" is a coroutine standing in for ``claude -p`` / ``codex exec``: the
sandbox fake invokes it on launch, and it dials the adapter back over real HTTP
loopback (``trust_env=False`` so the cluster proxy can't hijack 127.0.0.1),
firing a couple of turns the way the real CLI would.

Two protocol chains are covered:

  * ``test_generate_*`` -- the production path: real ``generate.generate()``,
    which is hardwired to ClaudeCodeHarness + AnthropicAdapter.
  * ``test_codex_openai_rollout_closes_loop`` -- the same loop for the
    CodexHarness + OpenAIAdapter pair, hand-wired (generate() does not select it).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import gzip
import json
import logging
import stat
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp import web

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Importing generate pulls slime.utils.processing_utils at module load, which
# eagerly imports transformers -- a heavy dep deliberately absent from the
# CPU-only CI env for this test. We never touch a real tokenizer (load_tokenizer
# is patched with FakeTokenizer below), so stub transformers before the import
# so the chain resolves without it.
if "transformers" not in sys.modules:
    _tf_stub = types.ModuleType("transformers")
    for _name in ("AutoProcessor", "AutoTokenizer", "PreTrainedTokenizerBase", "ProcessorMixin"):
        setattr(_tf_stub, _name, type(_name, (), {}))
    sys.modules["transformers"] = _tf_stub

# generate.generate() uses asyncio.timeout(), a 3.11+ API. CI runs the agent
# tests on 3.10, so shim it onto wait_for. The wall-clock guard never fires in
# these tests (every case finishes well under the guard), so a thin pass-through
# context manager is enough.
if not hasattr(asyncio, "timeout"):

    @contextlib.asynccontextmanager
    async def _timeout_shim(_delay):
        yield

    asyncio.timeout = _timeout_shim

import examples.coding_agent_rl.generate as gen  # noqa: E402
import examples.coding_agent_rl.swe as swe  # noqa: E402
from tests.test_agent._fakes import FakeSandbox, FakeTokenizer, fake_call_sglang_generate  # noqa: E402

from slime.agent.adapters import OpenAIAdapter  # noqa: E402
from slime.agent.adapters import common as adapters_common  # noqa: E402
from slime.agent.adapters.common import AdapterFailure  # noqa: E402
from slime.agent.aiohttp_threaded import run_app_in_thread  # noqa: E402
from slime.agent.harness import ClaudeCodeHarness, CodexHarness, HarnessRunResult  # noqa: E402
from slime.agent.harness import common as harness_common  # noqa: E402
from fanout_test_helpers import grpo_normalize_by_group_index  # noqa: E402
from slime.utils.misc import SingletonMeta  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

NUM_GPUS = 0

_REAL_SLEEP = asyncio.sleep
_TEST_SESSION_ID = "37c411bf-5b8f-5ceb-a5de-a944cc16e136"


def test_swe_config_defaults_trajectory_dir(monkeypatch):
    monkeypatch.delenv("SLIME_AGENT_TRAJECTORY_DIR", raising=False)
    monkeypatch.delenv("SLIME_AGENT_TRAJECTORY_SAVE", raising=False)

    assert gen.SweConfig.from_env().trajectory_dir == "/personal/muchen"


def test_code_agent_fanout_reward_post_process_normalizes_unique_trajectories_per_prompt():
    args = SimpleNamespace(reward_key=None, grpo_std_normalization=False)
    samples = [
        Sample(group_index=0, rollout_id=10, reward=1.0),
        Sample(group_index=0, rollout_id=10, reward=1.0),
        Sample(group_index=0, rollout_id=11, reward=0.0),
        Sample(group_index=1, index=20, rollout_id=20, reward=10.0),
        Sample(group_index=1, index=21, rollout_id=21, reward=8.0),
        Sample(group_index=1, index=21, rollout_id=21, reward=8.0),
        Sample(group_index=1, index=22, rollout_id=22, reward=2.0),
    ]

    raw_rewards, normalized_rewards = grpo_normalize_by_group_index(args, samples)

    assert raw_rewards == [1.0, 1.0, 0.0, 10.0, 8.0, 8.0, 2.0]
    assert normalized_rewards == pytest.approx([0.5, 0.5, -0.5, 10 / 3, 4 / 3, 4 / 3, -14 / 3])


def test_code_agent_fanout_reward_post_process_applies_grpo_std_normalization():
    args = SimpleNamespace(reward_key=None, grpo_std_normalization=True)
    samples = [
        Sample(group_index=0, rollout_id=10, reward=1.0),
        Sample(group_index=0, rollout_id=10, reward=1.0),
        Sample(group_index=0, rollout_id=11, reward=0.0),
    ]

    _, normalized_rewards = grpo_normalize_by_group_index(args, samples)

    assert normalized_rewards == pytest.approx([2**-0.5, 2**-0.5, -(2**-0.5)], rel=1e-5)


def test_code_agent_fanout_reward_post_process_rejects_missing_group_index():
    args = SimpleNamespace(reward_key=None, grpo_std_normalization=True)
    samples = [Sample(group_index=None, rollout_id=10, reward=1.0)]

    with pytest.raises(ValueError, match="group_index and rollout_id/index"):
        grpo_normalize_by_group_index(args, samples)


def test_code_agent_fanout_reward_post_process_rejects_missing_trajectory_id():
    args = SimpleNamespace(reward_key=None, grpo_std_normalization=True)
    samples = [Sample(group_index=0, index=None, rollout_id=None, reward=1.0)]

    with pytest.raises(ValueError, match="group_index and rollout_id/index"):
        grpo_normalize_by_group_index(args, samples)


def test_code_agent_fanout_reward_post_process_rejects_inconsistent_segment_rewards():
    args = SimpleNamespace(reward_key=None, grpo_std_normalization=True)
    samples = [
        Sample(group_index=0, rollout_id=10, reward=1.0),
        Sample(group_index=0, rollout_id=10, reward=0.0),
    ]

    with pytest.raises(ValueError, match="inconsistent rewards"):
        grpo_normalize_by_group_index(args, samples)


def test_arca_32gpu_launcher_enables_fanout_reward_post_process():
    launcher = (REPO_ROOT / "examples/coding_agent_rl/run_qwen36_27b_swe_32gpu_arca.sh").read_text()

    assert ("--custom-reward-post-process-path " "fanout_test_helpers.grpo_normalize_by_group_index") in launcher


class _ThetaGateway:
    """Minimal reverse proxy used to keep the rollout test on the Theta path."""

    def __init__(self) -> None:
        self.target: str | None = None
        app = web.Application()
        app.router.add_get("/healthz", self.healthz)
        app.router.add_route("*", "/{path:.*}", self.forward)
        self.handle = run_app_in_thread(app, host="127.0.0.1", port=0, thread_name="test-theta-gateway")

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.handle.port}"

    async def healthz(self, _request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def forward(self, request: web.Request) -> web.Response:
        assert self.target is not None, "Theta service was not registered"
        body = await request.read()
        headers = {key: value for key, value in request.headers.items() if key.lower() != "host"}
        async with aiohttp.ClientSession(trust_env=False) as session:
            async with session.request(
                request.method, f"{self.target}{request.rel_url}", headers=headers, data=body
            ) as response:
                return web.Response(status=response.status, headers=response.headers, body=await response.read())


_THETA_GATEWAY = _ThetaGateway()


async def _fast_sleep(_secs):  # collapse run_command's 5s poll loop
    await _REAL_SLEEP(0)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        hf_checkpoint="unused",  # load_tokenizer is patched
        rollout_max_context_len=0,
        sglang_tool_call_parser=None,
        sglang_reasoning_parser=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=1,  # never dialed (call_sglang_generate is patched)
        sglang_incremental_streaming_output=False,
    )


def _base_sample(**md) -> Sample:
    meta = {
        "instance_id": "demo-1",
        "image": "fake-image",
        "workdir": "/workspace/repo",
        "problem_statement": "fix the bug",
        "eval_cmd": "true",
        **md,
    }
    return Sample(index=0, group_index=0, prompt="fix the bug", metadata=meta)


async def _anthropic_agent(env: dict, *, n_turns: int = 2) -> int:
    """Stand-in for ``claude -p``: dial the adapter back over real HTTP and fire
    a few Anthropic turns, reading wiring from the env the harness exported."""
    base_url = env["ANTHROPIC_BASE_URL"]
    token = env["ANTHROPIC_AUTH_TOKEN"]
    history = [{"role": "user", "content": [{"type": "text", "text": "solve the issue"}]}]
    async with aiohttp.ClientSession(trust_env=False) as sess:
        for _ in range(n_turns):
            async with sess.post(
                f"{base_url}/v1/messages",
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "model": "ckpt:test-service",
                    "max_tokens": 64,
                    "metadata": {"user_id": json.dumps({"session_id": _TEST_SESSION_ID})},
                    "messages": history,
                },
            ) as r:
                data = await r.json()
            history.append({"role": "assistant", "content": data["content"]})
            history.append({"role": "user", "content": [{"type": "text", "text": "continue"}]})
    return 0


def _patch_generate(monkeypatch, tokenizer: FakeTokenizer, sandbox_factory) -> None:
    """Wire generate.generate()'s four external edges to CPU fakes."""
    # The Adapter remains loopback-only; the fake gateway is its only public route.
    monkeypatch.setattr(
        gen,
        "CONFIG",
        dataclasses.replace(
            gen.CONFIG,
            adapter_bind_host="127.0.0.1",
            adapter_port=0,
            theta_base_url=_THETA_GATEWAY.base_url,
            theta_service_name="test-service",
            theta_api_key="test-theta-token",
            rollout_guard_sec=60,
            agent_time_budget_sec=30,
            eval_timeout_sec=30,
            boot_retries=1,
        ),
    )
    monkeypatch.setattr(gen, "load_tokenizer", lambda *a, **k: tokenizer)
    monkeypatch.setattr(gen, "create_sandbox", sandbox_factory)  # boot sandbox
    monkeypatch.setattr(swe, "create_sandbox", sandbox_factory)  # eval sandbox
    monkeypatch.setattr(
        gen,
        "_register_adapter_to_theta",
        lambda port: setattr(_THETA_GATEWAY, "target", f"http://127.0.0.1:{port}"),
    )
    monkeypatch.setattr(ClaudeCodeHarness, "install_cli", _noop_install)
    monkeypatch.setattr(harness_common.asyncio, "sleep", _fast_sleep)
    monkeypatch.setattr(
        adapters_common, "call_sglang_generate", fake_call_sglang_generate(_two_turn_script(), tokenizer)
    )
    # _AdapterService is a SingletonMeta singleton; drop any cached instance so
    # each test builds a fresh adapter + app thread.
    SingletonMeta.clear_instances(gen._AdapterService)


async def _noop_install(self, sb) -> None:
    return None


def _two_turn_script():
    # (response_text, finish_reason, logprobs) per sglang call; encoded by the
    # FakeTokenizer so the adapter's decode round-trips it.
    return [
        ("let me look at the code", "stop", None),
        ("the fix is applied done", "stop", None),
    ]


# ===========================================================================
# §1 production path: real generate() over ClaudeCode + Anthropic
# ===========================================================================


def test_coding_agent_session_id_is_a_stable_uuid():
    sample = _base_sample()
    assert gen._session_id(sample, "demo-1") == "37c411bf-5b8f-5ceb-a5de-a944cc16e136"

    sample.session_id = "cagent-legacy"
    assert gen._session_id(sample, "demo-1") == "1f88c9f3-b6e1-5408-ba1d-79dd287aef28"

    sample.session_id = "11111111-1111-4111-8111-111111111111"
    assert gen._session_id(sample, "demo-1") == sample.session_id


def test_generate_produces_trained_samples(caplog):
    async def run_case(monkeypatch):
        tok = FakeTokenizer()
        sandbox_factory = FakeSandbox.factory(on_launch=_anthropic_agent)
        _patch_generate(monkeypatch, tok, sandbox_factory)

        async def fail_if_probed(*_args, **_kwargs):
            raise AssertionError("probe ran on healthy rollout")

        monkeypatch.setattr(gen, "_probe_adapter_connectivity", fail_if_probed)

        samples = await gen.generate(_args(), _base_sample(), sampling_params={"max_new_tokens": 32})

        assert samples, "rollout produced no samples"
        for s in samples:
            assert s.status == Sample.Status.COMPLETED
            assert len(s.loss_mask) == len(s.rollout_log_probs) == s.response_length
            assert sum(s.loss_mask) > 0  # at least one trained token
            assert s.metadata.get("agent_exit_code") == 0
            assert s.metadata.get("trainable") is True
            assert s.metadata.get("invalid_reason") is None
            assert s.metadata.get("compaction_count") == 0
            assert s.metadata.get("max_prompt_tokens", 0) > 0
            assert s.metadata["rollout_agent_time"] >= 0
            assert s.metadata["rollout_eval_time"] >= 0
        # eval_cmd "true" applied cleanly on a clean (empty) diff -> reward 1.0,
        # split evenly across the emitted samples.
        assert abs(sum(s.reward for s in samples) - 1.0) < 1e-9

    caplog.set_level(logging.WARNING)
    with pytest.MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))

    assert "empty_diff=True" in caplog.text
    assert "git_diff_exit_code=0" in caplog.text
    assert "git_status_exit_code=0" in caplog.text


def test_generate_keeps_valid_trajectory_after_time_budget_exceeded():
    async def timing_out_agent(env):
        await _anthropic_agent(env, n_turns=1)
        return harness_common.EXIT_TIME_BUDGET_EXCEEDED

    async def run_case(monkeypatch):
        tok = FakeTokenizer()
        sandbox_factory = FakeSandbox.factory(on_launch=timing_out_agent)
        _patch_generate(monkeypatch, tok, sandbox_factory)

        samples = await gen.generate(_args(), _base_sample(), sampling_params={"max_new_tokens": 32})

        assert len(samples) == 1
        sample = samples[0]
        assert sample.status == Sample.Status.COMPLETED
        assert sample.remove_sample is False
        assert sample.metadata["agent_exit_code"] == harness_common.EXIT_TIME_BUDGET_EXCEEDED
        assert sample.metadata["harness_error_type"] == "time_budget_exceeded"
        assert sample.metadata["trainable"] is True
        assert sample.metadata["grading_solved"] is True
        assert sum(sample.loss_mask) > 0

    with pytest.MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))


def test_generate_persists_complete_trajectory_as_atomic_gzip(tmp_path, caplog):
    async def run_case(monkeypatch):
        tok = FakeTokenizer()

        class SizeLimitedReadSandbox(FakeSandbox):
            async def read_file(self, sandbox_path, *, user="root") -> str:
                content = await super().read_file(sandbox_path, user=user)
                if sandbox_path.endswith("/.harness/trajectory.jsonl") and len(content.encode()) > 5 * 1024 * 1024:
                    raise RuntimeError("read file: file is too large")
                return content

            async def download_file(self, sandbox_path, host_path, *, user="root") -> None:
                content = self.files.get(sandbox_path, b"")
                Path(host_path).write_bytes(content.encode() if isinstance(content, str) else content)

        sandbox_factory = SizeLimitedReadSandbox.factory(on_launch=_anthropic_agent)
        _patch_generate(monkeypatch, tok, sandbox_factory)
        monkeypatch.setattr(
            gen,
            "CONFIG",
            dataclasses.replace(
                gen.CONFIG,
                trajectory_save="all",
                trajectory_dir=str(tmp_path),
            ),
        )

        full_trajectory = json.dumps({"type": "assistant", "text": "x" * (6 * 1024 * 1024)}) + "\n"
        original_run = ClaudeCodeHarness.run

        async def run_with_trajectory(self, sb, *, workdir, **kwargs):
            result = await original_run(self, sb, workdir=workdir, **kwargs)
            sb.files[f"{workdir}/.harness/trajectory.jsonl"] = full_trajectory
            return result

        monkeypatch.setattr(ClaudeCodeHarness, "run", run_with_trajectory)
        samples = await gen.generate(_args(), _base_sample(), sampling_params={"max_new_tokens": 32})

        assert samples
        trajectory_paths = {s.metadata.get("trajectory_path") for s in samples}
        assert len(trajectory_paths) == 1
        trajectory_path = Path(trajectory_paths.pop())
        assert trajectory_path.parent.parent.parent == tmp_path
        assert trajectory_path.suffixes == [".jsonl", ".gz"]
        with gzip.open(trajectory_path, "rt", encoding="utf-8") as fp:
            assert fp.read() == full_trajectory
        assert stat.S_IMODE(trajectory_path.stat().st_mode) == 0o600
        assert all(s.metadata["trajectory_raw_bytes"] == len(full_trajectory.encode()) for s in samples)
        assert all(s.metadata["trajectory_gzip_bytes"] == trajectory_path.stat().st_size for s in samples)
        assert not list(tmp_path.rglob("*.tmp"))

    caplog.set_level(logging.INFO)
    with pytest.MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))

    assert "trajectory_saved path=" in caplog.text


def test_generate_reports_trajectory_download_failure_without_calling_it_empty(tmp_path, caplog):
    async def run_case(monkeypatch):
        class InterruptedDownloadSandbox(FakeSandbox):
            async def download_file(self, sandbox_path, host_path, *, user="root") -> None:
                Path(host_path).write_bytes(b"partial")
                raise RuntimeError("download interrupted")

        tok = FakeTokenizer()
        sandbox_factory = InterruptedDownloadSandbox.factory(on_launch=_anthropic_agent)
        _patch_generate(monkeypatch, tok, sandbox_factory)
        monkeypatch.setattr(
            gen,
            "CONFIG",
            dataclasses.replace(
                gen.CONFIG,
                trajectory_save="all",
                trajectory_dir=str(tmp_path),
            ),
        )

        samples = await gen.generate(_args(), _base_sample(), sampling_params={"max_new_tokens": 32})

        assert samples
        assert not list(tmp_path.rglob("*.tmp"))
        assert not list(tmp_path.rglob("*.jsonl.gz"))

    caplog.set_level(logging.WARNING)
    with pytest.MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))

    assert "trajectory_save failed: RuntimeError: download interrupted" in caplog.text
    assert "sandbox file is empty" not in caplog.text


def test_generate_logs_structured_nonzero_harness_exit_and_continues_evaluation(caplog):
    launch_count = 0

    async def failing_agent(env):
        nonlocal launch_count
        launch_count += 1
        await _anthropic_agent(env, n_turns=1)
        return 1

    async def run_case(monkeypatch):
        tok = FakeTokenizer()
        output_tail = json.dumps(
            {
                "type": "result",
                "is_error": True,
                "terminal_reason": "api_error",
                "result": "API Error: Claude's response exceeded the 32000 output token maximum.",
            }
        )
        sandbox_factory = FakeSandbox.factory(
            on_launch=failing_agent,
            responses=[
                ("tail -c", (0, output_tail, "")),
                ("git add -N . && git diff", (0, "diff --git a/a.py b/a.py\n+fixed\n", "")),
            ],
        )
        _patch_generate(monkeypatch, tok, sandbox_factory)
        monkeypatch.setattr(
            adapters_common,
            "call_sglang_generate",
            fake_call_sglang_generate(_two_turn_script()[:1] * 3, tok),
        )

        samples = await gen.generate(_args(), _base_sample(), sampling_params={"max_new_tokens": 32})

        assert len(samples) == 1
        sample = samples[0]
        assert sample.status == Sample.Status.TERMINAL_FAILED
        assert sample.remove_sample is True
        assert sample.metadata.get("trainable") is False
        assert sample.metadata.get("invalid_reason") == "agent_exit:max_output_tokens"
        assert sample.metadata.get("abort_reason") == "agent_exit:max_output_tokens"
        assert sample.metadata.get("agent_exit_code") == 1
        assert sample.metadata.get("agent_error_type") == "max_output_tokens"
        assert sample.metadata.get("grading_solved") is True
        assert sample.metadata.get("applied_cleanly") is True
        assert sample.metadata.get("compaction_count") == 0
        assert sample.metadata.get("max_prompt_tokens", 0) > 0
        assert launch_count == 1 + ClaudeCodeHarness.max_recovery_attempts

    caplog.set_level(logging.WARNING)
    with pytest.MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))

    assert "agent_exit_code=1" in caplog.text
    assert "error_type=max_output_tokens" in caplog.text
    assert "terminal_reason=api_error" in caplog.text
    assert "evaluation_continues=True" in caplog.text
    assert "trainable=False" in caplog.text
    assert "CLI error (exit 1)" not in caplog.text
    assert "trajectory_tail=" not in caplog.text


def test_generate_marks_upstream_server_error_retryable():
    async def failing_agent(env):
        await _anthropic_agent(env, n_turns=1)
        return 1

    async def run_case(monkeypatch):
        tok = FakeTokenizer()
        output_tail = json.dumps(
            {
                "type": "result",
                "is_error": True,
                "terminal_reason": "api_error",
                "error": "server_error",
                "result": "API Error: upstream service unavailable",
            }
        )
        sandbox_factory = FakeSandbox.factory(
            on_launch=failing_agent,
            responses=[
                ("tail -c", (0, output_tail, "")),
                ("git add -N . && git diff", (0, "diff --git a/a.py b/a.py\n+fixed\n", "")),
            ],
        )
        _patch_generate(monkeypatch, tok, sandbox_factory)
        monkeypatch.setattr(
            adapters_common,
            "call_sglang_generate",
            fake_call_sglang_generate(_two_turn_script()[:1], tok),
        )

        samples = await gen.generate(_args(), _base_sample(), sampling_params={"max_new_tokens": 32})

        assert len(samples) == 1
        assert samples[0].status == Sample.Status.FAILED
        assert samples[0].metadata["invalid_reason"] == "agent_exit:server_error"

    with pytest.MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))


def test_session_closed_server_error_is_terminal():
    result = HarnessRunResult(
        exit_code=1,
        error_type="server_error",
        terminal_reason="api_error",
        error_message="API Error: session closed",
    )

    assert gen._agent_failure_status(result) == Sample.Status.TERMINAL_FAILED


@pytest.mark.parametrize("failure_code", ["incomplete_tool_parameter", "protocol_error"])
def test_stream_protocol_error_is_retryable_with_native_reason(failure_code):
    result = HarnessRunResult(
        exit_code=1,
        error_type="server_error",
        terminal_reason="api_error",
        error_message="API Error: upstream service unavailable",
    )
    failure = AdapterFailure(
        request_index=3,
        family="stream_protocol_error",
        code=failure_code,
    )

    assert gen._agent_failure_reason(result, failure) == "stream_protocol_error"
    assert gen._agent_failure_status(result, failure) == Sample.Status.FAILED
    assert gen._adapter_failure_metadata(failure) == {
        "adapter_failure_request_index": 3,
        "adapter_failure_family": "stream_protocol_error",
        "adapter_failure_code": failure_code,
    }


def test_upstream_generation_abort_is_interrupted_with_native_reason():
    result = HarnessRunResult(
        exit_code=1,
        error_type="server_error",
        terminal_reason="api_error",
        error_message="API Error: upstream generation aborted",
    )
    failure = AdapterFailure(
        request_index=4,
        family="stream_interrupted",
        code="upstream_abort",
    )

    assert gen._agent_failure_reason(result, failure) == "stream_interrupted"
    assert gen._agent_failure_status(result, failure) == Sample.Status.ABORTED


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_retry_after"),
    [
        (gen.SandboxCreateRateLimitError(retry_after=7), Sample.Status.FAILED, 7.0),
        (gen.SandboxLeaseError("ambiguous lease"), Sample.Status.TERMINAL_FAILED, None),
    ],
)
def test_generate_classifies_sandbox_creation_failure(error, expected_status, expected_retry_after):
    async def run_case(monkeypatch):
        tok = FakeTokenizer()

        class FailingSandbox(FakeSandbox):
            async def __aenter__(self):
                raise error

        _patch_generate(monkeypatch, tok, FailingSandbox.factory())

        samples = await gen.generate(_args(), _base_sample(), sampling_params={"max_new_tokens": 32})

        assert len(samples) == 1
        assert samples[0].status == expected_status
        assert samples[0].retry_after_seconds == expected_retry_after

    with pytest.MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))


def test_adapter_connectivity_probe_rejects_malformed_url_without_credentials():
    async def run_case():
        result = await gen._probe_adapter_connectivity(
            FakeSandbox(),
            "http://secret@example.invalid:not-a-port/path?token=hidden",
        )
        assert result == {
            "target": "<invalid>",
            "sandbox": {"status": "not_run"},
            "host": {"status": "not_run"},
            "classification": "malformed_adapter_url",
        }
        assert "secret" not in repr(result)
        assert "hidden" not in repr(result)

    asyncio.run(run_case())


def test_adapter_connectivity_probe_classifies_sandbox_only_failure():
    async def run_case():
        async def health(_request):
            return web.json_response({"ok": True})

        app = web.Application()
        app.router.add_get("/healthz", health)
        handle = run_app_in_thread(app, host="127.0.0.1", port=0, thread_name="test-connectivity-probe")
        sb = FakeSandbox(
            responses=[
                ("ip route get", (0, "127.0.0.1 dev lo src 127.0.0.1\n", "")),
                ("curl", (7, "http_code=000\n", "Failed to connect")),
            ]
        )
        try:
            result = await gen._probe_adapter_connectivity(sb, f"http://127.0.0.1:{handle.port}")
        finally:
            handle.stop()

        assert result["target"] == f"http://127.0.0.1:{handle.port}"
        assert result["sandbox"]["route_status"] == "ok"
        assert result["sandbox"]["curl_exit"] == 7
        assert result["sandbox"]["http_code"] == 0
        assert result["host"]["status"] == "healthy"
        assert result["classification"] == "sandbox_connect_failure"

    asyncio.run(run_case())


def test_adapter_connectivity_probe_distinguishes_sandbox_http_error():
    async def run_case():
        async def health(_request):
            return web.json_response({"ok": True})

        app = web.Application()
        app.router.add_get("/healthz", health)
        handle = run_app_in_thread(app, host="127.0.0.1", port=0, thread_name="test-connectivity-http-error")
        sb = FakeSandbox(
            responses=[
                ("ip route get", (0, "127.0.0.1 dev lo src 127.0.0.1\n", "")),
                ("curl", (0, "http_code=503\n", "")),
            ]
        )
        try:
            result = await gen._probe_adapter_connectivity(sb, f"http://127.0.0.1:{handle.port}")
        finally:
            handle.stop()

        assert result["sandbox"]["curl_exit"] == 0
        assert result["sandbox"]["http_code"] == 503
        assert result["host"]["status"] == "healthy"
        assert result["classification"] == "sandbox_adapter_http_error"

    asyncio.run(run_case())


def test_generate_marks_empty_trajectory_terminal(caplog):
    """If the agent never drives a turn, the session is empty and generate()
    returns one non-trainable terminal sample rather than crashing."""

    async def silent_agent(_env) -> int:
        return 0  # never contacts the adapter -> empty trajectory

    async def run_case(monkeypatch):
        tok = FakeTokenizer()
        sandbox_factory = FakeSandbox.factory(
            on_launch=silent_agent,
            responses=[
                ("ip route get", (0, "33.215.27.151 dev eth0 src 10.0.0.2\n", "")),
                ("curl", (7, "http_code=000\n", "Failed to connect")),
            ],
        )
        _patch_generate(monkeypatch, tok, sandbox_factory)

        async def silent_run(self, sb, *, workdir, **_kwargs):
            sb.files[f"{workdir}/.harness/trajectory.jsonl"] = '{"type":"error","error":"adapter unreachable"}\n'
            return HarnessRunResult(exit_code=0)

        monkeypatch.setattr(ClaudeCodeHarness, "run", silent_run)

        samples = await gen.generate(_args(), _base_sample(), sampling_params={})

        assert len(samples) == 1
        assert samples[0].status == Sample.Status.TERMINAL_FAILED
        assert samples[0].metadata.get("abort_reason") == "adapter_session_empty"

    caplog.set_level(logging.WARNING)
    with pytest.MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))

    assert "adapter_turns=0" in caplog.text
    assert "agent_exit_code=0" in caplog.text
    assert "diff_bytes=0" in caplog.text
    assert "adapter unreachable" in caplog.text
    assert "adapter_connectivity=" in caplog.text
    assert "sandbox_connect_failure" in caplog.text
    assert "ANTHROPIC_AUTH_TOKEN" not in caplog.text


def test_generate_keeps_empty_session_terminal_when_connectivity_probe_fails(caplog):
    async def run_case(monkeypatch):
        tok = FakeTokenizer()
        _patch_generate(monkeypatch, tok, FakeSandbox.factory())

        async def silent_run(self, sb, **_kwargs):
            return HarnessRunResult(exit_code=0)

        async def failed_probe(*_args, **_kwargs):
            raise RuntimeError("probe transport failed")

        monkeypatch.setattr(ClaudeCodeHarness, "run", silent_run)
        monkeypatch.setattr(gen, "_probe_adapter_connectivity", failed_probe)
        samples = await gen.generate(_args(), _base_sample(), sampling_params={})

        assert len(samples) == 1
        assert samples[0].status == Sample.Status.TERMINAL_FAILED
        assert samples[0].metadata["abort_reason"] == "adapter_session_empty"

    caplog.set_level(logging.WARNING)
    with pytest.MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))

    assert "diagnostic_failed" in caplog.text
    assert "probe transport failed" in caplog.text


def test_generate_marks_missing_image_terminal():
    async def run_case(monkeypatch):
        tok = FakeTokenizer()
        _patch_generate(monkeypatch, tok, FakeSandbox.factory(on_launch=_anthropic_agent))
        # blank image -> early abort before any sandbox boot.
        samples = await gen.generate(_args(), _base_sample(image=""), sampling_params={})
        assert len(samples) == 1
        assert samples[0].status == Sample.Status.TERMINAL_FAILED
        assert samples[0].rollout_id == samples[0].index == 0
        assert samples[0].metadata.get("abort_reason") == "missing_image_or_workdir"

    with pytest.MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))


def test_generate_rejects_missing_sample_index():
    async def run_case(monkeypatch):
        tok = FakeTokenizer()
        _patch_generate(monkeypatch, tok, FakeSandbox.factory(on_launch=_anthropic_agent))
        sample = _base_sample(image="")
        sample.index = None

        with pytest.raises(ValueError, match="index is required"):
            await gen.generate(_args(), sample, sampling_params={})

    with pytest.MonkeyPatch.context() as mp:
        asyncio.run(run_case(mp))


# ===========================================================================
# §2 the Codex + OpenAI pair closes the same loop (hand-wired)
# ===========================================================================


async def _codex_agent(env: dict, *, n_turns: int = 2) -> int:
    base_url = env["OPENAI_BASE_URL"]  # already includes /v1
    token = env["OPENAI_API_KEY"]
    history = [{"role": "user", "content": "solve the issue"}]
    async with aiohttp.ClientSession(trust_env=False) as sess:
        for _ in range(n_turns):
            async with sess.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {token}"},
                json={"model": "m", "max_tokens": 64, "messages": history},
            ) as r:
                data = await r.json()
            msg = data["choices"][0]["message"]
            history.append({"role": "assistant", "content": msg.get("content") or ""})
            history.append({"role": "user", "content": "continue"})
    return 0


def test_codex_openai_rollout_closes_loop(monkeypatch):
    """CodexHarness drives an in-thread OpenAIAdapter through a FakeSandbox; the
    loop produces trained samples just like the production Anthropic path."""

    async def run_case():
        tok = FakeTokenizer()
        monkeypatch.setattr(
            adapters_common, "call_sglang_generate", fake_call_sglang_generate(_two_turn_script(), tok)
        )
        monkeypatch.setattr(harness_common.asyncio, "sleep", _fast_sleep)

        adapter = OpenAIAdapter(tokenizer=tok, sglang_url="http://unused")
        handle = run_app_in_thread(adapter.app, host="127.0.0.1", port=0, thread_name="test-openai-adapter")
        adapter_url = f"http://127.0.0.1:{handle.port}"
        sid = "codex-sess"
        adapter.open_session(sid)
        try:
            sb = FakeSandbox(on_launch=_codex_agent)
            result = await CodexHarness().run(
                sb,
                workdir="/workspace/repo",
                session_id=sid,
                adapter_url=adapter_url,
                time_budget_sec=30,
                prompt="fix it",
            )
            samples = await adapter.finish_session(sid, base_sample=Sample(index=0, prompt=""), reward=1.0)
        finally:
            handle.stop()

        assert result == HarnessRunResult(exit_code=0)
        assert samples
        for s in samples:
            assert len(s.loss_mask) == len(s.rollout_log_probs) == s.response_length
            assert sum(s.loss_mask) > 0

    asyncio.run(run_case())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
