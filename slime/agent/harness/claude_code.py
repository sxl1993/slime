"""Claude Code harness."""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import time
from pathlib import Path

from slime.agent.sandbox import EXIT_TIME_BUDGET_EXCEEDED, Sandbox

from .common import BaseHarness, HarnessContext, HarnessRunResult, install_npm_cli, run_agent

_ERROR_MESSAGE_LIMIT = 300


def _last_json_string_field(text: str, field: str) -> str | None:
    pattern = re.compile(rf'"{re.escape(field)}"\s*:\s*("(?:\\.|[^"\\])*")')
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    try:
        value = json.loads(matches[-1].group(1))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, str) else None


def _bounded_message(value: str | None) -> str | None:
    if not value:
        return None
    normalized = " ".join(value.split())
    return normalized[:_ERROR_MESSAGE_LIMIT] or None


def _claude_run_result(exit_code: int, output_tail: str) -> HarnessRunResult:
    if exit_code == 0:
        return HarnessRunResult(exit_code=0)
    if exit_code == EXIT_TIME_BUDGET_EXCEEDED:
        return HarnessRunResult(
            exit_code=exit_code,
            error_type="time_budget_exceeded",
            terminal_reason="time_budget_exceeded",
        )

    terminal_reason = _last_json_string_field(output_tail, "terminal_reason")
    error_type = _last_json_string_field(output_tail, "error")
    is_error_record = (
        '"is_error":true' in output_tail.replace(" ", "")
        or '"is_api_error_message":true' in output_tail.replace(" ", "")
        or terminal_reason == "api_error"
    )
    error_message = _bounded_message(_last_json_string_field(output_tail, "result")) if is_error_record else None
    if terminal_reason == "rapid_refill_breaker":
        error_type = "context_thrashing"
    elif error_message and (
        "output token maximum" in error_message or "CLAUDE_CODE_MAX_OUTPUT_TOKENS" in error_message
    ):
        error_type = "max_output_tokens"
    elif error_message and (
        "prompt is too long" in error_message.lower()
        or "model_context_window_exceeded" in error_message.lower()
        or "context window exceeded" in error_message.lower()
    ):
        error_type = "context_overflow"
    elif not error_type and terminal_reason == "api_error":
        error_type = "api_error"
    if not error_type:
        error_type = "unclassified_cli_error"
    return HarnessRunResult(
        exit_code=exit_code,
        error_type=error_type,
        terminal_reason=terminal_reason,
        error_message=error_message,
    )


class ClaudeCodeHarness(BaseHarness):
    name = "claude_code"
    max_recovery_attempts = 2
    recovery_prompt = "Continue the task from the current state. Finish concisely without repeating prior analysis."

    # host paths + CLI knobs, all under the agent-layer SLIME_AGENT_* prefix
    node_tarball_env = "SLIME_AGENT_NODE_TARBALL"
    cli_tarball_env = "SLIME_AGENT_CC_TARBALL"
    extra_args_env = "SLIME_AGENT_CC_EXTRA_ARGS"
    extra_envs_env = "SLIME_AGENT_CC_EXTRA_ENVS"
    forwarded_envs = (
        "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
        "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    )

    launch_flags = (
        "--permission-mode bypassPermissions "
        "--output-format stream-json --include-partial-messages "
        "--include-hook-events --verbose"
    )

    static_env = {
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    }

    async def install_cli(self, sb: Sandbox) -> None:
        await install_npm_cli(
            sb,
            node_runtime=Path(os.environ[self.node_tarball_env]),
            npm_package=Path(os.environ[self.cli_tarball_env]),
            check_cmd="ls -la /usr/local/bin/claude && /usr/local/bin/claude --version",
        )

    async def verify_cli(self, sb: Sandbox) -> None:
        await sb.exec(
            "command -v claude && claude --version",
            user=sb.work_user,
            check=True,
            timeout=60,
        )

    def _configured_extra_envs(self) -> dict[str, str]:
        raw = os.environ.get(self.extra_envs_env, "").strip()
        if not raw:
            return {}
        env = json.loads(raw)
        if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
            raise ValueError(f"{self.extra_envs_env} must be a JSON object with string keys and values")
        return env

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        """Pre-ack bypass-permissions so claude-code starts headless."""
        settings_data: dict[str, object] = {
            "hasCompletedOnboarding": True,
            "bypassPermissionsModeAccepted": True,
        }
        extra_envs = self._configured_extra_envs()
        if extra_envs:
            settings_data["env"] = extra_envs
        settings = json.dumps(settings_data)
        claude_dir = f"{sb.home_dir}/.claude"
        claude_json = f"{sb.home_dir}/.claude.json"
        ownership = ""
        if sb.privileged_user != sb.work_user:
            ownership = f" && chown -R {sb.work_user}:{sb.work_user} {claude_dir} {claude_json}"
        await sb.exec(
            f"mkdir -p {claude_dir} && "
            f"echo {shlex.quote(settings)} "
            f"| tee {claude_json} {claude_dir}/settings.json > /dev/null"
            f"{ownership}",
            user=sb.privileged_user,
            check=True,
            timeout=60,
        )

    async def launch_and_wait(
        self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int
    ) -> HarnessRunResult:
        extra = os.environ.get(self.extra_args_env, "").strip()
        extra_envs = self._configured_extra_envs()
        conda_prefix = extra_envs.get("CONDA_PREFIX", "").rstrip("/")
        python_bin = f"{conda_prefix}/bin/python" if conda_prefix else None

        def _command(turn_prompt: str, *, resume: bool) -> str:
            if python_bin:
                turn_prompt = (
                    f"{turn_prompt}\n\n"
                    f"Use {python_bin} for all Python commands, including pytest and pip. "
                    "Do not install or upgrade dependencies."
                )
            session_flag = "--resume" if resume else "--session-id"
            command = (
                f"/usr/local/bin/claude -p {shlex.quote(turn_prompt)} "
                f"{session_flag} {shlex.quote(ctx.session_id)} {self.launch_flags}"
            )
            return f"{command} {extra}" if extra else command

        env = {
            "ANTHROPIC_BASE_URL": ctx.adapter_url,
            "ANTHROPIC_AUTH_TOKEN": ctx.adapter_auth_token or ctx.session_id,
            "ANTHROPIC_MODEL": ctx.model_label,
            **self.static_env,
            **{name: os.environ[name] for name in self.forwarded_envs if os.environ.get(name)},
            **extra_envs,
        }

        if python_bin:
            python_bin_quoted = shlex.quote(python_bin)
            preflight_cmd = (
                "actual=$(command -v python || true); "
                'printf "python=%s\\n" "$actual"; '
                'test -n "$actual" && '
                f'test "$(readlink -f "$actual")" = "$(readlink -f {python_bin_quoted})" && '
                f"{python_bin_quoted} -c {shlex.quote('import sys; print(sys.executable)')}"
            )
            exit_code, stdout, stderr = await sb.exec(
                preflight_cmd,
                user=sb.work_user,
                env=env,
                check=False,
                timeout=60,
            )
            if exit_code != 0:
                detail = _bounded_message(f"{stdout} {stderr}") or f"exit {exit_code}"
                raise RuntimeError(f"Claude Code Python preflight failed for {python_bin}: {detail}")

        trajectory_file = f"{ctx.workdir}/.harness/trajectory.jsonl"
        deadline = time.monotonic() + max(0, time_budget_sec)
        exit_code, output_tail = await run_agent(
            sb,
            workdir=ctx.workdir,
            start_cmd=_command(prompt, resume=False),
            env=env,
            time_budget_sec=time_budget_sec,
            tag="run",
            out_file=trajectory_file,
        )
        result = _claude_run_result(exit_code, output_tail)

        for attempt in range(1, self.max_recovery_attempts + 1):
            if result.error_type not in {"max_output_tokens", "context_overflow", "context_thrashing"}:
                return result
            remaining = math.ceil(deadline - time.monotonic())
            if remaining <= 0:
                return HarnessRunResult(
                    exit_code=EXIT_TIME_BUDGET_EXCEEDED,
                    error_type="time_budget_exceeded",
                    terminal_reason="time_budget_exceeded",
                )

            retry_file = f"{ctx.workdir}/.harness/trajectory.resume-{attempt}.jsonl"
            exit_code, output_tail = await run_agent(
                sb,
                workdir=ctx.workdir,
                start_cmd=_command(self.recovery_prompt, resume=True),
                env=env,
                time_budget_sec=remaining,
                tag=f"run-resume-{attempt}",
                out_file=retry_file,
            )
            await sb.exec(
                f"cat {shlex.quote(retry_file)} >> {shlex.quote(trajectory_file)}",
                user=sb.work_user,
                check=True,
                timeout=30,
            )
            result = _claude_run_result(exit_code, output_tail)

        return result
