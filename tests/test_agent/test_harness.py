"""Unit tests for the coding-agent harness + sandbox layers.

These cover the parts a happy-path rollout can't pin down precisely: that each
harness writes the right CLI config and launches with the right command + env,
that ``run_agent``'s detached-launch / poll-marker handshake returns the right
exit code (and times out correctly), and that ``ensure_agent_user`` issues the
expected provisioning command. A :class:`tests.test_agent._fakes.FakeSandbox`
records every ``exec`` / ``write_file`` so we assert on the issued commands
without a real sandbox or any root privilege.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_agent._fakes import FakeSandbox  # noqa: E402

from slime.agent import sandbox as sandbox_mod  # noqa: E402
from slime.agent.harness import ClaudeCodeHarness, CodexHarness, HarnessContext, HarnessRunResult  # noqa: E402
from slime.agent.harness import claude_code as claude_code_mod  # noqa: E402
from slime.agent.harness import common as hc  # noqa: E402

NUM_GPUS = 0

# Run the 5s poll loop instantly without recursing into the patched function.
_REAL_SLEEP = asyncio.sleep


async def _fast_sleep(_secs):
    await _REAL_SLEEP(0)


def _ctx(
    workdir="/workspace/repo", sid="sess-1", url="http://host:18001", *, model="slime-actor", token=None
) -> HarnessContext:
    return HarnessContext(
        workdir=workdir,
        session_id=sid,
        adapter_url=url,
        model_label=model,
        adapter_auth_token=token,
    )


def _find(exec_log, needle):
    return [cmd for cmd, _user in exec_log if needle in cmd]


# ===========================================================================
# §1 run_agent handshake (the E2B detached-launch transport)
# ===========================================================================


def test_run_agent_returns_marker_exit_code():
    async def run_case():
        seen = {}

        async def fake_agent(env):
            seen["env"] = env
            return 0  # exit code written into the done marker

        sb = FakeSandbox(on_launch=fake_agent)
        with patch.object(hc.asyncio, "sleep", new=_fast_sleep):
            rc, output_tail = await hc.run_agent(
                sb, workdir="/workspace/repo", start_cmd="claude -p hi", env={"A": "1"}, time_budget_sec=30
            )
        assert rc == 0
        assert output_tail == ""
        assert seen["env"] == {"A": "1"}
        # launcher script + detached setsid launch all issued, exit code captured.
        assert any("run.sh" in p for p in sb.files)
        assert _find(sb.exec_log, "setsid")
        assert any("echo $?" in v for v in sb.files.values())

    asyncio.run(run_case())


def test_run_agent_propagates_nonzero_exit():
    async def run_case():
        async def fail_agent(_env):
            return 7

        sb = FakeSandbox(on_launch=fail_agent)
        with patch.object(hc.asyncio, "sleep", new=_fast_sleep):
            rc, output_tail = await hc.run_agent(sb, workdir="/w", start_cmd="x", env={}, time_budget_sec=30)
        assert rc == 7
        assert output_tail == ""

    asyncio.run(run_case())


def test_run_agent_times_out_when_marker_never_appears():
    async def run_case():
        sb = FakeSandbox(on_launch=None)  # no agent -> marker never written
        with patch.object(hc.asyncio, "sleep", new=_fast_sleep):
            rc, output_tail = await hc.run_agent(sb, workdir="/w", start_cmd="x", env={}, time_budget_sec=0)
        assert rc == sandbox_mod.EXIT_TIME_BUDGET_EXCEEDED
        assert output_tail == ""

    asyncio.run(run_case())


def test_run_agent_preserves_bounded_nonzero_output_tail():
    async def run_case():
        async def fail_agent(_env):
            return 1

        sb = FakeSandbox(
            on_launch=fail_agent,
            responses=[("tail -c", (0, "structured failure detail", ""))],
        )
        with patch.object(hc.asyncio, "sleep", new=_fast_sleep):
            rc, output_tail = await hc.run_agent(sb, workdir="/w", start_cmd="x", env={}, time_budget_sec=30)
        assert rc == 1
        assert output_tail == "structured failure detail"

    asyncio.run(run_case())


# ===========================================================================
# §2 ClaudeCodeHarness config + launch
# ===========================================================================


def test_claude_code_write_config_preacks_bypass_permissions():
    async def run_case():
        sb = FakeSandbox()
        await ClaudeCodeHarness().write_config(sb, _ctx())
        joined = " ".join(cmd for cmd, _ in sb.exec_log)
        assert "/home/agent/.claude/settings.json" in joined
        assert "bypassPermissionsModeAccepted" in joined
        assert "hasCompletedOnboarding" in joined

    asyncio.run(run_case())


def test_claude_code_uses_admin_home_without_agent_provisioning_for_arca():
    async def run_case():
        sb = FakeSandbox()
        sb.work_user = "admin"
        sb.privileged_user = "admin"
        sb.home_dir = "/home/admin"
        sb.cli_preinstalled = True
        with patch.object(sandbox_mod, "ensure_agent_user") as ensure:
            await ClaudeCodeHarness().write_config(sb, _ctx(workdir="/testbed"))
            ensure.assert_not_called()
        joined = " ".join(cmd for cmd, _ in sb.exec_log)
        assert "/home/admin/.claude/settings.json" in joined
        assert "/home/agent" not in joined
        assert "chown" not in joined
        assert all(user == "admin" for _, user in sb.exec_log)

    asyncio.run(run_case())


def test_preinstalled_claude_is_verified_without_installing_tarballs():
    async def run_case():
        sb = FakeSandbox()
        sb.work_user = "admin"
        sb.privileged_user = "admin"
        sb.home_dir = "/home/admin"
        sb.cli_preinstalled = True
        harness = ClaudeCodeHarness()
        with patch.object(harness, "install_cli") as install:
            await harness.prepare_cli(sb)
            install.assert_not_called()
        joined = " ".join(cmd for cmd, _ in sb.exec_log)
        assert "command -v claude" in joined
        assert "claude --version" in joined
        assert all(user == "admin" for _, user in sb.exec_log)

    asyncio.run(run_case())


def test_claude_code_launch_command_and_env():
    async def run_case():
        captured = {}
        session_id = "11111111-1111-4111-8111-111111111111"

        async def agent(env):
            captured["env"] = env
            return 0

        capturing = FakeSandbox(on_launch=agent)
        with patch.object(hc.asyncio, "sleep", new=_fast_sleep):
            result = await ClaudeCodeHarness().launch_and_wait(
                capturing,
                _ctx(
                    sid=session_id,
                    url="https://theta.example/api/anthropic",
                    model="ckpt:theta-service",
                    token="theta-api-key",
                ),
                prompt="solve it",
                time_budget_sec=30,
            )
        assert result == HarnessRunResult(exit_code=0)
        # the prompt + flags land in the launcher script body.
        body = next(v for k, v in capturing.files.items() if k.endswith("run.sh"))
        assert "claude -p 'solve it'" in body
        assert f"--session-id {session_id}" in body
        assert "--permission-mode bypassPermissions" in body
        # env carries the adapter wiring under the Anthropic var names.
        env = captured["env"]
        assert env["ANTHROPIC_BASE_URL"] == "https://theta.example/api/anthropic"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "theta-api-key"
        assert env["ANTHROPIC_MODEL"] == "ckpt:theta-service"
        assert env["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"

    asyncio.run(run_case())


def test_claude_code_forwards_only_supported_context_budget_envs_to_sandbox():
    async def run_case():
        captured = {}

        async def agent(env):
            captured["env"] = env
            return 0

        sb = FakeSandbox(on_launch=agent)
        process_env = {
            "CLAUDE_CODE_AUTO_COMPACT_WINDOW": "65536",
            "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE": "70",
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32768",
            "SLIME_AGENT_CC_EXTRA_ENVS": json.dumps(
                {
                    "CUSTOM_SANDBOX_ENV": "enabled",
                }
            ),
        }
        with (
            patch.dict(claude_code_mod.os.environ, process_env, clear=False),
            patch.object(hc.asyncio, "sleep", new=_fast_sleep),
        ):
            await ClaudeCodeHarness().launch_and_wait(
                sb,
                _ctx(),
                prompt="solve it",
                time_budget_sec=30,
            )

        env = captured["env"]
        assert env["CLAUDE_CODE_AUTO_COMPACT_WINDOW"] == "65536"
        assert env["CLAUDE_AUTOCOMPACT_PCT_OVERRIDE"] == "70"
        assert "CLAUDE_CODE_MAX_OUTPUT_TOKENS" not in env
        assert env["CUSTOM_SANDBOX_ENV"] == "enabled"

    asyncio.run(run_case())


def test_claude_code_persists_extra_envs_and_preflights_conda_python():
    async def run_case():
        captured = {}

        async def agent(env):
            captured["env"] = env
            return 0

        sb = FakeSandbox(on_launch=agent)
        extra_envs = {
            "PATH": "/opt/miniconda3/envs/testbed/bin:/usr/bin:/bin",
            "CONDA_PREFIX": "/opt/miniconda3/envs/testbed",
            "CONDA_DEFAULT_ENV": "testbed",
        }
        with (
            patch.dict(
                claude_code_mod.os.environ,
                {"SLIME_AGENT_CC_EXTRA_ENVS": json.dumps(extra_envs)},
                clear=False,
            ),
            patch.object(hc.asyncio, "sleep", new=_fast_sleep),
        ):
            harness = ClaudeCodeHarness()
            await harness.write_config(sb, _ctx())
            result = await harness.launch_and_wait(
                sb,
                _ctx(),
                prompt="solve it",
                time_budget_sec=30,
            )

        assert result == HarnessRunResult(exit_code=0)
        settings_cmd = next(cmd for cmd, _ in sb.exec_log if "settings.json" in cmd)
        assert '"env"' in settings_cmd
        assert '"CONDA_PREFIX": "/opt/miniconda3/envs/testbed"' in settings_cmd
        preflight_cmd = next(cmd for cmd, _ in sb.exec_log if "command -v python" in cmd)
        assert "/opt/miniconda3/envs/testbed/bin/python" in preflight_cmd
        assert "import sys; print(sys.executable)" in preflight_cmd
        body = next(v for k, v in sb.files.items() if k.endswith("run.sh"))
        assert "Use /opt/miniconda3/envs/testbed/bin/python for all Python commands" in body
        assert "Do not install or upgrade dependencies" in body
        assert captured["env"]["PATH"] == extra_envs["PATH"]

    asyncio.run(run_case())


def test_claude_code_does_not_launch_when_python_preflight_fails():
    async def run_case():
        launched = False

        async def agent(_env):
            nonlocal launched
            launched = True
            return 0

        sb = FakeSandbox(
            on_launch=agent,
            responses=[("command -v python", (1, "/opt/python/bin/python\n", "No module named numpy"))],
        )
        extra_envs = {
            "PATH": "/opt/miniconda3/envs/testbed/bin:/usr/bin:/bin",
            "CONDA_PREFIX": "/opt/miniconda3/envs/testbed",
        }
        with patch.dict(
            claude_code_mod.os.environ,
            {"SLIME_AGENT_CC_EXTRA_ENVS": json.dumps(extra_envs)},
            clear=False,
        ):
            with pytest.raises(RuntimeError, match="Claude Code Python preflight failed"):
                await ClaudeCodeHarness().launch_and_wait(
                    sb,
                    _ctx(),
                    prompt="solve it",
                    time_budget_sec=30,
                )

        assert not launched
        assert not _find(sb.exec_log, "setsid")

    asyncio.run(run_case())


def test_claude_code_classifies_max_output_tokens_without_leaking_unrelated_fields():
    async def run_case():
        async def fail_agent(_env):
            return 1

        message = "Claude's response exceeded the 32000 output token maximum."
        output_tail = json.dumps(
            {
                "type": "result",
                "is_error": True,
                "terminal_reason": "api_error",
                "result": f"API Error: {message}",
                "authorization": "Bearer secret-token",
            }
        )
        sb = FakeSandbox(on_launch=fail_agent, responses=[("tail -c", (0, output_tail, ""))])
        with (
            patch.object(ClaudeCodeHarness, "max_recovery_attempts", 0),
            patch.object(hc.asyncio, "sleep", new=_fast_sleep),
        ):
            result = await ClaudeCodeHarness().launch_and_wait(
                sb,
                _ctx(sid="sess-fail", url="https://theta.example/api/anthropic"),
                prompt="solve it",
                time_budget_sec=30,
            )

        assert result == HarnessRunResult(
            exit_code=1,
            error_type="max_output_tokens",
            terminal_reason="api_error",
            error_message=f"API Error: {message}",
        )
        assert "secret-token" not in repr(result)

    asyncio.run(run_case())


def test_claude_code_resumes_same_session_after_max_output_error():
    async def run_case():
        message = "Claude's response exceeded the 32768 output token maximum."
        failure_tail = json.dumps(
            {
                "type": "result",
                "is_error": True,
                "terminal_reason": "api_error",
                "result": f"API Error: {message}",
            }
        )
        calls = []

        async def fake_run_agent(sb, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return 1, failure_tail
            return 0, ""

        sb = FakeSandbox()
        with patch.object(claude_code_mod, "run_agent", new=fake_run_agent):
            result = await ClaudeCodeHarness().launch_and_wait(
                sb,
                _ctx(sid="11111111-1111-4111-8111-111111111111"),
                prompt="solve it",
                time_budget_sec=30,
            )

        assert result == HarnessRunResult(exit_code=0)
        assert len(calls) == 2
        assert "--session-id 11111111-1111-4111-8111-111111111111" in calls[0]["start_cmd"]
        assert "--resume 11111111-1111-4111-8111-111111111111" in calls[1]["start_cmd"]
        assert "Continue the task from the current state" in calls[1]["start_cmd"]
        assert calls[0]["tag"] == "run"
        assert calls[1]["tag"] == "run-resume-1"
        assert calls[1]["out_file"].endswith("trajectory.resume-1.jsonl")
        assert _find(sb.exec_log, "cat /workspace/repo/.harness/trajectory.resume-1.jsonl >>")

    asyncio.run(run_case())


def test_claude_code_resumes_same_session_after_rapid_refill_breaker():
    async def run_case():
        failure_tail = json.dumps(
            {
                "type": "result",
                "is_error": True,
                "terminal_reason": "rapid_refill_breaker",
                "error": "invalid_request",
                "result": (
                    "Autocompact is thrashing: the context refilled to the limit within 3 turns of the previous "
                    "compact, 3 times in a row."
                ),
            }
        )
        calls = []

        async def fake_run_agent(sb, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return 1, failure_tail
            return 0, ""

        sb = FakeSandbox()
        with patch.object(claude_code_mod, "run_agent", new=fake_run_agent):
            result = await ClaudeCodeHarness().launch_and_wait(
                sb,
                _ctx(sid="33333333-3333-4333-8333-333333333333"),
                prompt="solve it",
                time_budget_sec=30,
            )

        assert result == HarnessRunResult(exit_code=0)
        assert len(calls) == 2
        assert "--session-id 33333333-3333-4333-8333-333333333333" in calls[0]["start_cmd"]
        assert "--resume 33333333-3333-4333-8333-333333333333" in calls[1]["start_cmd"]
        assert calls[1]["tag"] == "run-resume-1"
        assert _find(sb.exec_log, "cat /workspace/repo/.harness/trajectory.resume-1.jsonl >>")

    asyncio.run(run_case())


def test_claude_code_classifies_adapter_context_rejection():
    output_tail = json.dumps(
        {
            "type": "result",
            "is_error": True,
            "terminal_reason": "api_error",
            "result": "API Error: Prompt is too long for the configured context window",
        }
    )

    assert claude_code_mod._claude_run_result(1, output_tail) == HarnessRunResult(
        exit_code=1,
        error_type="context_overflow",
        terminal_reason="api_error",
        error_message="API Error: Prompt is too long for the configured context window",
    )


def test_claude_code_recovery_is_bounded_and_shares_original_deadline():
    async def run_case():
        failure_tail = json.dumps(
            {
                "type": "result",
                "is_error": True,
                "terminal_reason": "api_error",
                "result": "API Error: Claude's response exceeded the output token maximum.",
            }
        )
        calls = []

        async def fake_run_agent(sb, **kwargs):
            calls.append(kwargs)
            return 1, failure_tail

        sb = FakeSandbox()
        with (
            patch.object(claude_code_mod, "run_agent", new=fake_run_agent),
            patch.object(claude_code_mod.time, "monotonic", side_effect=[100.0, 101.0, 102.0]),
        ):
            result = await ClaudeCodeHarness().launch_and_wait(
                sb,
                _ctx(sid="22222222-2222-4222-8222-222222222222"),
                prompt="solve it",
                time_budget_sec=10,
            )

        assert result.error_type == "max_output_tokens"
        assert len(calls) == 3
        assert [call["time_budget_sec"] for call in calls] == [10, 9, 8]
        assert [call["tag"] for call in calls] == ["run", "run-resume-1", "run-resume-2"]
        assert len(_find(sb.exec_log, ">> /workspace/repo/.harness/trajectory.jsonl")) == 2

    asyncio.run(run_case())


# ===========================================================================
# §3 CodexHarness config + launch
# ===========================================================================


def test_codex_write_config_base64_roundtrips_inline_base_url():
    async def run_case():
        sb = FakeSandbox()
        await CodexHarness().write_config(sb, _ctx(url="http://host:18001"))
        # config written via base64 round-trip; decode the captured payload.
        cmd = next(c for c, _ in sb.exec_log if "base64 -d > /home/agent/.codex/config.toml" in c)
        b64 = cmd.split("echo ")[1].split(" | base64")[0].strip("'")
        toml = base64.b64decode(b64).decode()
        assert 'base_url = "http://host:18001/v1"' in toml  # MUST be inline
        assert 'wire_api = "chat"' in toml
        assert 'model_provider = "slime"' in toml

    asyncio.run(run_case())


def test_codex_launch_command_and_env():
    async def run_case():
        captured = {}

        async def agent(env):
            captured["env"] = env
            return 0

        sb = FakeSandbox(on_launch=agent)
        with patch.object(hc.asyncio, "sleep", new=_fast_sleep):
            result = await CodexHarness().launch_and_wait(
                sb, _ctx(sid="sess-cx", url="http://host:18001"), prompt="do work", time_budget_sec=30
            )
        assert result == HarnessRunResult(exit_code=0)
        body = next(v for k, v in sb.files.items() if k.endswith("run.sh"))
        assert "codex exec" in body and "do work" in body and "--skip-git-repo-check" in body
        env = captured["env"]
        assert env["OPENAI_API_KEY"] == "sess-cx"
        assert env["OPENAI_BASE_URL"] == "http://host:18001/v1"

    asyncio.run(run_case())


def test_codex_nonzero_exit_uses_unclassified_error_without_raw_output():
    async def run_case():
        async def fail_agent(_env):
            return 2

        sb = FakeSandbox(
            on_launch=fail_agent,
            responses=[("tail -c", (0, "OPENAI_API_KEY=secret-token internal failure", ""))],
        )
        with patch.object(hc.asyncio, "sleep", new=_fast_sleep):
            result = await CodexHarness().launch_and_wait(
                sb, _ctx(sid="sess-cx", url="http://host:18001"), prompt="do work", time_budget_sec=30
            )

        assert result == HarnessRunResult(exit_code=2, error_type="unclassified_cli_error")
        assert "secret-token" not in repr(result)

    asyncio.run(run_case())


# ===========================================================================
# §4 ensure_agent_user (sandbox infra)
# ===========================================================================


def test_ensure_agent_user_provisions_user_and_git_safe_dir():
    async def run_case():
        sb = FakeSandbox()
        await sandbox_mod.ensure_agent_user(sb, "/workspace/repo")
        cmd = next(c for c, _ in sb.exec_log if "useradd" in c)
        assert "id agent" in cmd
        assert "chown -R agent:agent" in cmd and "/workspace/repo" in cmd
        assert "git config --system --add safe.directory '*'" in cmd

    asyncio.run(run_case())


# ===========================================================================
# §5 harness.run wires the steps in order
# ===========================================================================


def test_base_harness_run_calls_steps_in_order():
    async def run_case():
        async def agent(_env):
            return 0

        sb = FakeSandbox(on_launch=agent)
        with patch.object(hc.asyncio, "sleep", new=_fast_sleep):
            result = await ClaudeCodeHarness().run(
                sb,
                workdir="/workspace/repo",
                session_id="sess-run",
                adapter_url="http://host:18001",
                time_budget_sec=30,
                prompt="go",
            )
        assert result == HarnessRunResult(exit_code=0)
        joined = " ".join(c for c, _ in sb.exec_log)
        # ensure_agent_user (useradd) -> write_config (settings.json) -> launch (setsid)
        order = [k for k in ("useradd", "settings.json", "setsid") if k in joined]
        assert order == ["useradd", "settings.json", "setsid"]

    asyncio.run(run_case())


def test_base_harness_run_uses_admin_and_never_calls_ensure_agent_user_for_arca():
    async def run_case():
        async def agent(_env):
            return 0

        sb = FakeSandbox(on_launch=agent)
        sb.work_user = "admin"
        sb.privileged_user = "admin"
        sb.home_dir = "/home/admin"
        sb.cli_preinstalled = True
        with (
            patch.object(sandbox_mod, "ensure_agent_user") as ensure,
            patch.object(hc.asyncio, "sleep", new=_fast_sleep),
        ):
            result = await ClaudeCodeHarness().run(
                sb,
                workdir="/testbed",
                session_id="sess-arca",
                adapter_url="https://adapter.example",
                time_budget_sec=30,
                prompt="go",
            )
        assert result == HarnessRunResult(exit_code=0)
        ensure.assert_not_called()
        assert all(user == "admin" for _, user in sb.exec_log)
        assert not _find(sb.exec_log, "useradd")
        assert not _find(sb.exec_log, "chown agent:agent")

    asyncio.run(run_case())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
