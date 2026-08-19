"""Harness-agnostic coding-agent lifecycle in a sandbox.

A harness is a swappable coding agent (Claude Code, Codex, ...). Each one
installs a CLI, writes its own config, and runs the agent against a prompt. The
shared parts (create the agent user, the run skeleton, the launch-detached-and-
poll transport) live here; adding a CLI-style harness means subclassing
BaseHarness and implementing install_cli, write_config and launch_and_wait.
Two module-level helpers cover the common cases: install_npm_cli for
npm-packaged CLIs, and run_agent for the launch-the-agent-to-completion case.

The base knows nothing about the task: run() takes only generic fields
(workdir / session_id / adapter_url / prompt). Task-specific workspace prep and
scoring live in the example layer.
"""

from __future__ import annotations

import asyncio
import lzma
import os
import shutil
import tempfile
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from slime.agent import sandbox as _sandbox
from slime.agent.sandbox import Sandbox, exec_and_wait
from slime.utils.misc import SingletonMeta


class SingletonABCMeta(ABCMeta, SingletonMeta):
    pass


# In-sandbox retry budget for the npm global install (transient flakes like
# exit 217). Cheaper than a full sandbox recreate by the caller.
NPM_INSTALL_RETRIES = 3
NPM_INSTALL_BACKOFF_SEC = 2.0


@dataclass(frozen=True)
class HarnessContext:
    """Generic run context, free of any task fields.

    model_label is the model name the harness advertises to its CLI. The slime
    adapter ignores it and serves whatever upstream sglang has loaded, so it is
    not a run() parameter.
    """

    workdir: str
    session_id: str
    adapter_url: str
    model_label: str = "slime-actor"
    adapter_auth_token: str | None = None


@dataclass(frozen=True)
class HarnessRunResult:
    """Normalized harness outcome safe for orchestration logs."""

    exit_code: int
    error_type: str | None = None
    terminal_reason: str | None = None
    error_message: str | None = None


class BaseHarness(ABC, metaclass=SingletonABCMeta):
    """Base lifecycle for a sandbox-resident coding agent."""

    # short identifier set by each subclass (claude_code / codex)
    name: str = ""

    @abstractmethod
    async def install_cli(self, sb: Sandbox) -> None:
        """Install the harness CLI into the sandbox.
        npm-packaged harnesses delegate to install_npm_cli."""

    async def verify_cli(self, sb: Sandbox) -> None:
        """Verify a backend-provided CLI without mutating the sandbox."""
        raise RuntimeError(f"{self.name} does not support a preinstalled CLI contract")

    async def prepare_cli(self, sb: Sandbox) -> None:
        """Install the CLI for legacy images or verify a preinstalled one."""
        if sb.cli_preinstalled:
            await self.verify_cli(sb)
        else:
            await self.install_cli(sb)

    @abstractmethod
    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        """Write any CLI config files into the sandbox."""

    @abstractmethod
    async def launch_and_wait(
        self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int
    ) -> HarnessRunResult:
        """Run the agent to completion and return its normalized outcome.

        A non-interactive CLI builds one shell command and hands it to
        run_agent. An interactive or long-running harness drives its own loop
        here instead.
        """

    async def run(
        self,
        sb: Sandbox,
        *,
        workdir: str,
        session_id: str,
        adapter_url: str,
        adapter_auth_token: str | None = None,
        model_label: str = "slime-actor",
        time_budget_sec: int,
        prompt: str,
    ) -> HarnessRunResult:
        """Run the harness in the sandbox and return its normalized outcome.

        Steps: prepare the backend work user -> write config -> launch and wait.
        Workspace prep (writing the problem statement etc.) is the caller's job
        and must run before this.
        """
        await _sandbox.prepare_work_user(sb, workdir)
        ctx = HarnessContext(
            workdir=workdir,
            session_id=session_id,
            adapter_url=adapter_url,
            adapter_auth_token=adapter_auth_token,
            model_label=model_label,
        )
        await self.write_config(sb, ctx)
        return await self.launch_and_wait(sb, ctx, prompt, time_budget_sec)


async def run_agent(
    sb: Sandbox,
    *,
    workdir: str,
    start_cmd: str,
    env: dict[str, str],
    time_budget_sec: int,
    tag: str = "run",
    out_file: str | None = None,
) -> tuple[int, str]:
    """Launch the agent and return its exit code plus bounded failure output."""
    meta_dir = f"{workdir}/.harness"
    setup_cmd = f"mkdir -p {meta_dir}"
    if sb.privileged_user != sb.work_user:
        setup_cmd += f" && chown {sb.work_user}:{sb.work_user} {meta_dir}"
    await sb.exec(setup_cmd, user=sb.privileged_user, check=True, timeout=30)
    return await exec_and_wait(
        sb,
        cmd=start_cmd,
        user=sb.work_user,
        env=env,
        workdir=workdir,
        out_file=out_file or f"{meta_dir}/trajectory.jsonl",
        time_budget_sec=time_budget_sec,
        tag=tag,
        want_output=False,
    )


async def install_npm_cli(
    sb: Sandbox,
    *,
    node_runtime: Path,
    npm_package: Path,
    check_cmd: str,
) -> None:
    """Install an npm-packaged CLI into the sandbox: the Node 22 runtime first,
    then the CLI's npm package (global install, then self-check via check_cmd).
    Non-npm harnesses write their own install_cli."""
    await install_node22(sb, node_runtime)

    await sb.write_file("/tmp/harness-cli.tgz", npm_package)
    install_cmd = "npm install -g --prefix=/usr/local --no-audit --no-fund /tmp/harness-cli.tgz && " + check_cmd
    # Detached install with a few in-place retries for transient disk flakes.
    last_log = ""
    for attempt in range(NPM_INSTALL_RETRIES):
        exit_code, last_log = await exec_and_wait(
            sb, cmd=install_cmd, user="root", time_budget_sec=300, tag="harness-npm-install"
        )
        if exit_code == 0:
            return
        if attempt + 1 < NPM_INSTALL_RETRIES:
            await asyncio.sleep(NPM_INSTALL_BACKOFF_SEC * (attempt + 1))
    raise RuntimeError(
        f"npm install failed after {NPM_INSTALL_RETRIES} attempts (exit={exit_code}):\n{last_log[-1000:]}"
    )


async def install_node22(sb: Sandbox, host_tarball: Path) -> None:
    """Install Node 22 over the base image (some base images ship a version too
    old for the CLI). A .xz tarball is decompressed on the host (cached) so
    sandboxes without xz-utils can still run a plain `tar xf`."""
    host_tarball = Path(host_tarball)
    if host_tarball.suffix == ".xz":
        plain = Path(tempfile.gettempdir()) / f"coding_agent_rl.{host_tarball.stem}.tar"
        if not plain.exists():
            tmp = plain.with_suffix(".tar.partial")
            with lzma.open(host_tarball, "rb") as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(tmp, plain)
        host_tarball = plain
    await sb.write_file("/tmp/node22.tar", host_tarball)
    await sb.exec(
        "set -e && mkdir -p /opt/node22 && "
        "tar xf /tmp/node22.tar -C /opt/node22 --strip-components=1 && "
        "ln -sf /opt/node22/bin/node /usr/local/bin/node && "
        "ln -sf /opt/node22/bin/npm  /usr/local/bin/npm && "
        "ln -sf /opt/node22/bin/npx  /usr/local/bin/npx && "
        "hash -r 2>/dev/null || true && node --version && npm --version",
        user="root",
        timeout=180,
        check=True,
    )
