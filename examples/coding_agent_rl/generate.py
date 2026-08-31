"""Coding-Agent RL: per-sample generate() function for slime.

    --custom-generate-function-path examples.coding_agent_rl.generate.generate

generate() is a four-stage orchestrator: swe.prepare_workspace + harness.run
-> swe.git_diff -> swe.run_evaluation -> adapter.finish_session. The (harness,
adapter) pair is chosen by the SWE_AGENT env var (claude_code | codex); see
_AGENTS below.
Sandbox-side work is split across three layers: the provider-agnostic sandbox
contract (slime.agent.sandbox), the swappable harness lifecycle
(slime.agent.harness), and the SWE task layer (examples.coding_agent_rl.swe --
dataset parsing, workspace prep, diff, eval). LLM plumbing (Anthropic / OpenAI
<-> SGLang /generate, token capture, segment split) is the matching
slime.agent.adapters adapter. swe.get_metadata documents the dataset row schema
and produces the md dict consumed below.
"""

from __future__ import annotations

import asyncio
import gzip
import logging
import os
import random
import re
import shlex
import shutil
import time
import traceback
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from slime.agent.adapters import AnthropicAdapter, OpenAIAdapter
from slime.agent.adapters.common import AdapterFailure
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.agent.harness import ClaudeCodeHarness, CodexHarness, HarnessRunResult
from slime.agent.sandbox import Sandbox, SandboxCreateRateLimitError, SandboxLeaseError, create_sandbox
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from . import swe

logger = logging.getLogger(__name__)
logging.getLogger("e2b").setLevel(logging.WARNING)

_AGENTS = {
    "claude_code": (ClaudeCodeHarness, AnthropicAdapter),
    "codex": (CodexHarness, OpenAIAdapter),
}
AGENT_NAME = os.environ.get("SWE_AGENT", "claude_code")
if AGENT_NAME not in _AGENTS:
    raise ValueError(f"SWE_AGENT={AGENT_NAME!r} not in {sorted(_AGENTS)}")
if AGENT_NAME == "codex":
    raise ValueError("SWE_AGENT=codex is not supported by the Theta-only route yet")
HARNESS_CLS, ADAPTER_CLS = _AGENTS[AGENT_NAME]


@dataclass(frozen=True)
class SweConfig:
    eval_protocol: str  # eval-path schema/grader (SWE_EVAL_PROTOCOL)
    train_protocol: str  # train-path schema/grader (SWE_TRAIN_PROTOCOL)
    adapter_bind_host: str
    adapter_port: int
    theta_base_url: str
    theta_service_name: str
    theta_api_key: str
    fork_merge_threshold: int | None
    agent_time_budget_sec: int
    eval_timeout_sec: int
    rollout_guard_sec: int
    boot_concurrency: int
    boot_retries: int
    max_tool_result_chars: int | None = None
    trajectory_save: str = "none"
    trajectory_dir: str = "/personal/muchen"
    trajectory_write_concurrency: int = 4

    @classmethod
    def from_env(cls) -> SweConfig:
        agent_time_budget = int(os.environ.get("SWE_AGENT_TIME_BUDGET_SEC", "1800"))
        eval_timeout = int(os.environ.get("SWE_EVAL_TIMEOUT_SEC", "600"))
        guard = int(os.environ.get("SWE_ROLLOUT_GUARD_SEC", "0") or 0) or (agent_time_budget + eval_timeout + 180)
        fork = int(v) if (v := os.environ.get("SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS")) else None
        max_tool_result_chars = (
            int(v) if (v := os.environ.get("SLIME_ADAPTER_MAX_TOOL_RESULT_CHARS", "").strip()) else None
        )
        if max_tool_result_chars is not None and max_tool_result_chars <= 0:
            raise ValueError("SLIME_ADAPTER_MAX_TOOL_RESULT_CHARS must be positive")
        trajectory_save = os.environ.get("SLIME_AGENT_TRAJECTORY_SAVE", "none").strip().lower()
        if trajectory_save not in {"all", "abnormal", "none"}:
            raise ValueError("SLIME_AGENT_TRAJECTORY_SAVE must be one of: all, abnormal, none")
        trajectory_dir = os.environ.get("SLIME_AGENT_TRAJECTORY_DIR", "/personal/muchen").strip()
        if trajectory_save != "none" and not trajectory_dir:
            raise ValueError("SLIME_AGENT_TRAJECTORY_DIR is required when trajectory saving is enabled")
        return cls(
            eval_protocol=os.environ.get("SWE_EVAL_PROTOCOL", swe.PROTOCOL_SCALESWE),
            train_protocol=os.environ.get("SWE_TRAIN_PROTOCOL", swe.PROTOCOL_SCALESWE),
            adapter_bind_host=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"),
            adapter_port=int(os.environ.get("ADAPTER_PORT", "18001")),
            theta_base_url=os.environ.get("THETA_BASE_URL", "").rstrip("/"),
            theta_service_name=os.environ.get("THETA_SERVICE_NAME", "").strip(),
            theta_api_key=os.environ.get("THETA_API_KEY", "").strip(),
            fork_merge_threshold=fork,
            agent_time_budget_sec=agent_time_budget,
            eval_timeout_sec=eval_timeout,
            rollout_guard_sec=guard,
            boot_concurrency=int(os.environ.get("SWE_BOOT_CONCURRENCY", "16")),
            boot_retries=int(os.environ.get("SWE_BOOT_RETRIES", "2")),
            max_tool_result_chars=max_tool_result_chars,
            trajectory_save=trajectory_save,
            trajectory_dir=trajectory_dir,
            trajectory_write_concurrency=max(1, int(os.environ.get("SLIME_AGENT_TRAJECTORY_WRITE_CONCURRENCY", "4"))),
        )


CONFIG = SweConfig.from_env()

_BOOT_SEM = asyncio.Semaphore(CONFIG.boot_concurrency)
_TRAJECTORY_WRITE_SEM = asyncio.Semaphore(CONFIG.trajectory_write_concurrency)


def _path_component(value: Any) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return component or "unknown"


def _trajectory_destination(base_sample: Sample, instance_id: str, session_id: str) -> Path:
    rollout_id = base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index
    return (
        Path(CONFIG.trajectory_dir)
        / f"rollout-{_path_component(rollout_id)}"
        / _path_component(instance_id)
        / f"{_path_component(session_id)}.jsonl.gz"
    )


def _write_trajectory_gzip(destination: Path, source: Path) -> int:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with source.open("rb") as input_file, open(temporary, "xb") as raw:
            os.chmod(temporary, 0o600)
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=1, mtime=0) as compressed:
                shutil.copyfileobj(input_file, compressed)
        os.replace(temporary, destination)
        os.chmod(destination, 0o600)
        return destination.stat().st_size
    finally:
        temporary.unlink(missing_ok=True)


async def _persist_trajectory(
    sb: Sandbox,
    sandbox_path: str,
    *,
    user: str,
    base_sample: Sample,
    instance_id: str,
    session_id: str,
) -> dict[str, Any]:
    destination = _trajectory_destination(base_sample, instance_id, session_id)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    source = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.raw.tmp")
    source.touch(mode=0o600)
    try:
        read_started = time.monotonic()
        await sb.download_file(sandbox_path, source, user=user)
        read_ms = (time.monotonic() - read_started) * 1000
        raw_bytes = source.stat().st_size
        if raw_bytes == 0:
            return {}

        queued = time.monotonic()
        async with _TRAJECTORY_WRITE_SEM:
            queue_ms = (time.monotonic() - queued) * 1000
            started = time.monotonic()
            gzip_bytes = await asyncio.to_thread(_write_trajectory_gzip, destination, source)
        write_ms = (time.monotonic() - started) * 1000
    finally:
        source.unlink(missing_ok=True)
    logger.info(
        "[coding_agent_rl] %s: trajectory_saved path=%s raw_bytes=%d gzip_bytes=%d "
        "read_ms=%.1f queue_ms=%.1f write_ms=%.1f",
        instance_id,
        destination,
        raw_bytes,
        gzip_bytes,
        read_ms,
        queue_ms,
        write_ms,
    )
    return {
        "trajectory_path": str(destination),
        "trajectory_raw_bytes": raw_bytes,
        "trajectory_gzip_bytes": gzip_bytes,
        "trajectory_read_ms": read_ms,
        "trajectory_queue_ms": queue_ms,
        "trajectory_write_ms": write_ms,
    }


def _should_save_trajectory(*, agent_exit_code: int, empty_diff: bool, adapter_turns: int) -> bool:
    if CONFIG.trajectory_save == "all":
        return True
    if CONFIG.trajectory_save == "abnormal":
        return agent_exit_code != 0 or empty_diff or adapter_turns == 0
    return False


@asynccontextmanager
async def boot_agent_sandbox(image: str, instance_id: str, session_id: str) -> AsyncIterator[Sandbox]:
    """Boot a fresh selected-backend sandbox and prepare its harness CLI.

    E2B images receive the existing Node/CLI install. ARCA images carry the CLI
    already and are only verified. An ambiguous ARCA create is never retried.
    """
    sb = None
    last_err: Exception | None = None
    for attempt in range(CONFIG.boot_retries):
        cand = create_sandbox(
            image,
            metadata={
                "instance_id": instance_id,
                "session_id": session_id,
                "role": "agent",
                "attempt": str(attempt + 1),
            },
        )
        try:
            async with _BOOT_SEM:
                await cand.__aenter__()
                try:
                    await HARNESS_CLS().prepare_cli(cand)
                except BaseException:
                    await cand.__aexit__(None, None, None)
                    raise
            sb = cand
            break
        except SandboxLeaseError:
            logger.error(
                "[coding_agent_rl] %s: sandbox lease requires reconciliation; automatic retry disabled",
                instance_id,
            )
            raise
        except Exception as e:
            last_err = e
            if attempt + 1 >= CONFIG.boot_retries:
                logger.warning(
                    "[coding_agent_rl] %s: provision attempt %d/%d failed: %s: %s; giving up",
                    instance_id,
                    attempt + 1,
                    CONFIG.boot_retries,
                    type(e).__name__,
                    str(e)[:200],
                )
                continue
            retry_delay = (
                max(
                    1 + attempt,
                    e.retry_after if isinstance(e, SandboxCreateRateLimitError) else 0,
                )
                + random.random()
            )
            logger.warning(
                "[coding_agent_rl] %s: provision attempt %d/%d failed: %s: %s; backoff %.1fs",
                instance_id,
                attempt + 1,
                CONFIG.boot_retries,
                type(e).__name__,
                str(e)[:200],
                retry_delay,
            )
            await asyncio.sleep(retry_delay)
    if sb is None:
        assert last_err is not None
        raise last_err
    try:
        yield sb
    finally:
        await sb.__aexit__(None, None, None)


class _AdapterService(metaclass=SingletonMeta):
    def __init__(self, args) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        self.tool_parser = getattr(args, "sglang_tool_call_parser", None) or None
        self.reasoning_parser = getattr(args, "sglang_reasoning_parser", None) or None
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        missing_theta = [
            name
            for name, value in (
                ("THETA_BASE_URL", CONFIG.theta_base_url),
                ("THETA_SERVICE_NAME", CONFIG.theta_service_name),
                ("THETA_API_KEY", CONFIG.theta_api_key),
            )
            if not value
        ]
        if missing_theta:
            raise RuntimeError("Coding-agent RL requires the Theta gateway; set " + ", ".join(missing_theta))
        self.adapter = ADAPTER_CLS(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
            fork_threshold_tokens=CONFIG.fork_merge_threshold,
            max_tool_result_chars=CONFIG.max_tool_result_chars,
            sglang_incremental_streaming_output=bool(args.sglang_incremental_streaming_output),
        )
        # handler_cancellation=True so a client disconnect cancels the handler
        # coroutine, arming the fire-and-forget /abort_request in the adapter.
        # Otherwise a cancelled client leaves an inflight sglang /generate that
        # races the next release_memory_occupation and trips its idle assertion.
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=CONFIG.adapter_bind_host,
            port=CONFIG.adapter_port,
            thread_name="anthropic-adapter",
            runner_kwargs={
                "handler_cancellation": True,
                "access_log_class": FilteredAccessLogger,
            },
        )
        self.adapter_url = CONFIG.theta_base_url
        self.adapter_auth_token = CONFIG.theta_api_key
        self.model_label = f"ckpt:{CONFIG.theta_service_name}"
        logger.info(
            "[coding_agent_rl] tokenizer=%s theta=%s model=%s max_context_len=%s "
            "max_tool_result_chars=%s tool_parser=%s reasoning_parser=%s",
            args.hf_checkpoint,
            self.adapter_url,
            self.model_label,
            self.max_context_len,
            CONFIG.max_tool_result_chars,
            self.tool_parser,
            self.reasoning_parser,
        )
        _register_adapter_to_theta(self.app_handle.port)


_THETA_DEFAULT_HOST = "https://aistudio.alipay.com"


def _register_adapter_to_theta(port: int) -> None:
    """Register this adapter pod to the Theta gateway so antchat can route
    ``ckpt:<THETA_SERVICE_NAME>`` requests back here.

    Runs inside the RolloutManager / adapter pod, where AIStudio injects
    ``POD_IP``, ``SYSTEM_API_JWT_TAG`` (register auth) and, on aidc clusters,
    ``DV_ENDPOINT_ADDR`` (the proxy address; absence means pod-direct reachable
    from antchat). The URL is registered to ``/v1`` -- not the CLI's hardcoded
    ``/v1/chat/completions`` -- so llmpivotbase maps antchat
    ``/api/anthropic/v1/messages`` onto the adapter's ``/v1/messages`` route.

    Raises on missing env or registration failure: in Theta mode an unregistered
    adapter means every sandbox rollout 404s at antchat, so fail fast and loud
    rather than degrade into per-sample ambiguity.
    """
    pod_ip = os.environ.get("POD_IP")
    jwt = os.environ.get("SYSTEM_API_JWT_TAG")
    service = os.environ.get("THETA_SERVICE_NAME")
    missing = [
        k for k, v in (("POD_IP", pod_ip), ("SYSTEM_API_JWT_TAG", jwt), ("THETA_SERVICE_NAME", service)) if not v
    ]
    if missing:
        raise RuntimeError(
            "Theta registration requires " + ", ".join(missing) + " to be injected into the adapter pod env"
        )
    dv = os.environ.get("DV_ENDPOINT_ADDR")
    base = f"{dv}/proxy?target={pod_ip}:{port}" if dv else f"http://{pod_ip}:{port}"
    url = f"{base}/v1"
    # Prefer the package's own register path (it handles JWT read, host
    # resolution and retries); fall back to a direct POST if the package is
    # unavailable on this image.
    try:
        from aistudio_checkpoint.cli.caller import register_theta_service

        register_theta_service(service, url, None, None, None)
    except ImportError:
        import requests

        host = os.environ.get("AISTUDIO_CHECKPOINT_HOST") or _THETA_DEFAULT_HOST
        if not host.startswith("http"):
            host = os.environ.get("AISTUDIO_NETWORK_PROTOCOL", "https://") + host
        resp = requests.post(
            f"{host}/api/theta/model/register",
            json={"name": service, "url": url},
            headers={"Authorization": f"Bearer {jwt}"},
            timeout=10,
        )
        resp.raise_for_status()
    logger.info("[coding_agent_rl] registered adapter to Theta: %s -> %s", service, url)


async def _probe_adapter_connectivity(sb: Sandbox, adapter_url: str) -> dict[str, Any]:
    try:
        parsed = urlsplit(adapter_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("unsupported adapter URL")
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
        if not 1 <= port <= 65535:
            raise ValueError("invalid adapter port")
    except (TypeError, ValueError):
        return {
            "target": "<invalid>",
            "sandbox": {"status": "not_run"},
            "host": {"status": "not_run"},
            "classification": "malformed_adapter_url",
        }

    host = parsed.hostname
    display_host = f"[{host}]" if ":" in host else host
    target = f"{parsed.scheme}://{display_host}:{port}"

    route_cmd = f"if command -v ip >/dev/null 2>&1; then ip route get {shlex.quote(host)}; else echo route_tool_unavailable; exit 127; fi"
    try:
        route_exit, route_stdout, route_stderr = await sb.exec(
            route_cmd,
            user=sb.work_user,
            timeout=3,
            check=False,
            idempotent=True,
        )
        route_status = "unavailable" if route_exit == 127 else ("ok" if route_exit == 0 else "failed")
        route_result = {
            "route_status": route_status,
            "route_exit": route_exit,
            "route_stdout": (route_stdout or "")[:512],
            "route_stderr": (route_stderr or "")[:512],
        }
    except Exception as e:
        route_result = {
            "route_status": "probe_error",
            "route_error": f"{type(e).__name__}: {str(e)[:200]}",
        }

    health_url = f"{target}/healthz"
    curl_cmd = f"curl --noproxy '*' --connect-timeout 5 --max-time 8 --silent --show-error --output /dev/null --write-out 'http_code=%{{http_code}}\\n' {shlex.quote(health_url)}"
    try:
        curl_exit, curl_stdout, curl_stderr = await sb.exec(
            curl_cmd,
            user=sb.work_user,
            timeout=10,
            check=False,
            idempotent=True,
        )
        http_code = 0
        for line in (curl_stdout or "").splitlines():
            if line.startswith("http_code="):
                try:
                    http_code = int(line.removeprefix("http_code="))
                except ValueError:
                    pass
        sandbox_result = {
            **route_result,
            "curl_exit": curl_exit,
            "http_code": http_code,
            "curl_stdout": (curl_stdout or "")[:512],
            "curl_stderr": (curl_stderr or "")[:512],
        }
    except Exception as e:
        sandbox_result = {
            **route_result,
            "curl_status": "probe_error",
            "curl_error": f"{type(e).__name__}: {str(e)[:200]}",
        }

    timeout = aiohttp.ClientTimeout(total=8, connect=5)
    try:
        async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
            async with session.get(health_url) as response:
                await response.content.read(512)
                host_result = {
                    "status": "healthy" if response.status == 200 else "http_error",
                    "http_code": response.status,
                }
    except Exception as e:
        host_result = {
            "status": "connect_error",
            "error": f"{type(e).__name__}: {str(e)[:200]}",
        }

    sandbox_http_reached = sandbox_result.get("curl_exit") == 0 and sandbox_result.get("http_code", 0) > 0
    sandbox_healthy = sandbox_http_reached and sandbox_result.get("http_code") == 200
    host_healthy = host_result["status"] == "healthy"
    if sandbox_healthy and host_healthy:
        classification = "adapter_reachable_no_turn"
    elif sandbox_http_reached and not sandbox_healthy:
        classification = "sandbox_adapter_http_error"
    elif not sandbox_healthy and host_healthy:
        curl_exit = sandbox_result.get("curl_exit")
        if sandbox_result.get("route_status") == "failed":
            classification = "sandbox_route_failure"
        elif curl_exit == 6:
            classification = "sandbox_dns_failure"
        elif curl_exit == 7:
            classification = "sandbox_connect_failure"
        elif curl_exit == 28:
            classification = "sandbox_connect_timeout"
        else:
            classification = "sandbox_cannot_connect_adapter"
    elif sandbox_healthy:
        classification = "host_cannot_connect_adapter"
    else:
        classification = "adapter_unreachable_from_both"

    return {
        "target": target,
        "sandbox": sandbox_result,
        "host": host_result,
        "classification": classification,
    }


async def generate(args, base_sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    """Per-sample agent function with wall-clock guard (see rollout_guard_sec)."""
    if base_sample.index is None:
        raise ValueError("index is required for Code Agent rollout generation.")
    if base_sample.rollout_id is None:
        base_sample.rollout_id = base_sample.index

    state = _AdapterService(args)
    protocol = CONFIG.eval_protocol if evaluation else CONFIG.train_protocol
    md = swe.get_metadata(base_sample, protocol)
    instance_id = md["instance_id"]
    if not md["image"] or not md["workdir"]:
        return _abort_result(base_sample, "missing_image_or_workdir", instance_id)
    reason = swe.evaluability_check(md)
    if reason:
        return _abort_result(base_sample, f"unevaluatable:{reason}", instance_id)

    session_id = base_sample.session_id = _session_id(base_sample, instance_id)
    state.adapter.open_session(
        session_id,
        sampling_defaults=sampling_params,
        max_context_tokens=state.max_context_len,
    )
    t0 = time.time()
    phase_started = time.perf_counter()
    try:
        async with asyncio.timeout(CONFIG.rollout_guard_sec):
            async with boot_agent_sandbox(md["image"], instance_id, session_id) as sb:
                await swe.prepare_workspace(sb, md["workdir"], md)
                run_result = await HARNESS_CLS().run(
                    sb,
                    workdir=md["workdir"],
                    session_id=session_id,
                    adapter_url=state.adapter_url,
                    adapter_auth_token=state.adapter_auth_token,
                    model_label=state.model_label,
                    time_budget_sec=CONFIG.agent_time_budget_sec,
                    prompt=swe.SWE_PROMPT,
                )
                agent_exit_code = run_result.exit_code
                adapter_failure = state.adapter.session_failure(session_id)
                adapter_failure_metadata = _adapter_failure_metadata(adapter_failure)
                diff_text, git_diff_exit_code, git_diff_stderr = await swe.git_diff(sb, md["workdir"])
                adapter_turns = state.adapter.manager.turn_count(session_id)
                session_stats = state.adapter.session_stats(session_id)
                empty_diff = not diff_text.strip()
                diff_bytes = len(diff_text.encode("utf-8"))
                needs_diagnostic = empty_diff or git_diff_exit_code != 0 or adapter_turns == 0
                save_trajectory = _should_save_trajectory(
                    agent_exit_code=agent_exit_code,
                    empty_diff=empty_diff,
                    adapter_turns=adapter_turns,
                )
                trajectory_metadata: dict[str, Any] = {}
                trajectory_tail = ""
                trajectory_path = f"{md['workdir']}/.harness/trajectory.jsonl"
                if needs_diagnostic:
                    try:
                        tail_exit_code, trajectory_tail, tail_stderr = await sb.exec(
                            f"tail -c 4096 {shlex.quote(trajectory_path)}",
                            user=sb.work_user,
                            timeout=15,
                            check=False,
                        )
                        if tail_exit_code != 0:
                            raise RuntimeError(tail_stderr[-200:] or f"tail exited {tail_exit_code}")
                    except Exception as e:
                        trajectory_tail = f"<read failed: {type(e).__name__}: {str(e)[:200]}>"
                        logger.warning(
                            "[coding_agent_rl] %s: trajectory_tail failed: %s: %s",
                            instance_id,
                            type(e).__name__,
                            str(e)[:200],
                        )

                if save_trajectory:
                    try:
                        trajectory_metadata = await _persist_trajectory(
                            sb,
                            trajectory_path,
                            user=sb.work_user,
                            base_sample=base_sample,
                            instance_id=instance_id,
                            session_id=session_id,
                        )
                    except Exception as e:
                        logger.warning(
                            "[coding_agent_rl] %s: trajectory_save failed: %s: %s",
                            instance_id,
                            type(e).__name__,
                            str(e)[:200],
                        )
                    else:
                        if not trajectory_metadata:
                            logger.warning(
                                "[coding_agent_rl] %s: trajectory_save skipped because sandbox file is empty",
                                instance_id,
                            )
                if agent_exit_code != 0:
                    logger.warning(
                        "[coding_agent_rl] %s: agent_exit_code=%d error_type=%s terminal_reason=%s "
                        "error_message=%r adapter_turns=%d diff_bytes=%d evaluation_continues=True",
                        instance_id,
                        agent_exit_code,
                        run_result.error_type,
                        run_result.terminal_reason,
                        run_result.error_message,
                        adapter_turns,
                        diff_bytes,
                    )
                if needs_diagnostic:
                    status_exit_code, status_out, status_err = await sb.exec(
                        f"cd {md['workdir']} && git status --short --untracked-files=all --ignored",
                        user=sb.work_user,
                        timeout=30,
                    )
                    adapter_connectivity = None
                    if adapter_turns == 0:
                        try:
                            adapter_connectivity = await _probe_adapter_connectivity(sb, state.adapter_url)
                        except Exception as e:
                            adapter_connectivity = {
                                "classification": "diagnostic_failed",
                                "error": f"{type(e).__name__}: {str(e)[:200]}",
                            }
                    logger.warning(
                        "[coding_agent_rl] %s: empty_diff=%s agent_exit_code=%d adapter_turns=%d diff_bytes=%d "
                        "git_diff_exit_code=%d git_diff_stderr=%r git_status_exit_code=%d git_status=%r "
                        "git_status_stderr=%r adapter_connectivity=%r trajectory_tail=%r",
                        instance_id,
                        empty_diff,
                        agent_exit_code,
                        adapter_turns,
                        diff_bytes,
                        git_diff_exit_code,
                        git_diff_stderr[-400:],
                        status_exit_code,
                        status_out[-4096:],
                        status_err[-400:],
                        adapter_connectivity,
                        trajectory_tail,
                    )
            rollout_agent_time = time.perf_counter() - phase_started
            evaluation_started = time.perf_counter()
            reward, applied_cleanly = await swe.run_evaluation(
                md,
                diff_text=diff_text,
                timeout_sec=CONFIG.eval_timeout_sec,
            )
            rollout_eval_time = time.perf_counter() - evaluation_started
            rollout_timing_metadata = {
                "rollout_agent_time": rollout_agent_time,
                "rollout_eval_time": rollout_eval_time,
            }
            if evaluation:
                logger.info(
                    "[coding_agent_rl] %s: reward=%.2f applied=%s agent_exit_code=%d elapsed=%.1fs (eval-only)",
                    instance_id,
                    float(reward),
                    bool(applied_cleanly),
                    agent_exit_code,
                    time.time() - t0,
                )
                return _eval_result(
                    base_sample,
                    reward=float(reward),
                    applied_cleanly=bool(applied_cleanly),
                    agent_exit_code=agent_exit_code,
                    instance_id=instance_id,
                    extra_metadata={
                        **session_stats,
                        **trajectory_metadata,
                        **rollout_timing_metadata,
                        "harness_error_type": run_result.error_type,
                        **adapter_failure_metadata,
                    },
                )

            allow_timeout_sample = run_result.error_type == "time_budget_exceeded" and adapter_failure is None
            if agent_exit_code != 0 and not allow_timeout_sample:
                invalid_reason = f"agent_exit:{_agent_failure_reason(run_result, adapter_failure)}"
                logger.warning(
                    "[coding_agent_rl] %s: reward=%.2f applied=%s agent_exit_code=%d trainable=False "
                    "invalid_reason=%s compaction_count=%d max_prompt_tokens=%d context_exceeded_count=%d",
                    instance_id,
                    float(reward),
                    bool(applied_cleanly),
                    agent_exit_code,
                    invalid_reason,
                    session_stats["compaction_count"],
                    session_stats["max_prompt_tokens"],
                    session_stats["context_exceeded_count"],
                )
                return _abort_result(
                    base_sample,
                    invalid_reason,
                    instance_id,
                    status=_agent_failure_status(run_result, adapter_failure),
                    extra_metadata={
                        **session_stats,
                        **trajectory_metadata,
                        **rollout_timing_metadata,
                        "agent_exit_code": agent_exit_code,
                        "agent_error_type": run_result.error_type,
                        "harness_error_type": run_result.error_type,
                        "agent_terminal_reason": run_result.terminal_reason,
                        "agent_error_message": run_result.error_message,
                        **adapter_failure_metadata,
                        "grading_solved": float(reward) == 1.0,
                        "applied_cleanly": bool(applied_cleanly),
                    },
                )

            samples = await state.adapter.finish_session(
                session_id,
                base_sample=base_sample,
                reward=float(reward),
                extra_metadata={
                    "grading_solved": float(reward) == 1.0,
                    "applied_cleanly": bool(applied_cleanly),
                    "instance_id": instance_id,
                    "agent_exit_code": agent_exit_code,
                    "trainable": True,
                    "invalid_reason": None,
                    **rollout_timing_metadata,
                    **trajectory_metadata,
                },
            )
            if not samples:
                return _abort_result(
                    base_sample,
                    "adapter_session_empty",
                    instance_id,
                    extra_metadata=trajectory_metadata,
                )

            logger.info(
                "[coding_agent_rl] %s: reward=%.2f applied=%s agent_exit_code=%d elapsed=%.1fs segments=%d "
                "trainable=True compaction_count=%d max_prompt_tokens=%d context_exceeded_count=%d",
                instance_id,
                float(reward),
                bool(applied_cleanly),
                agent_exit_code,
                time.time() - t0,
                len(samples),
                session_stats["compaction_count"],
                session_stats["max_prompt_tokens"],
                session_stats["context_exceeded_count"],
            )
            return samples

    except asyncio.TimeoutError:
        _log_timeout_diagnostic(t0, instance_id)
        return _abort_result(base_sample, "wall_clock_timeout", instance_id)
    except SandboxCreateRateLimitError as e:
        return _abort_result(
            base_sample,
            "exception:SandboxCreateRateLimitError",
            instance_id,
            status=Sample.Status.FAILED,
            retry_after_seconds=e.retry_after,
        )
    except Exception as e:
        logger.warning(
            "[coding_agent_rl] %s: rollout failed: %s\n%s",
            instance_id,
            e,
            traceback.format_exc(),
        )
        return _abort_result(base_sample, f"exception:{type(e).__name__}", instance_id)
    finally:
        await state.adapter.drop_session(session_id, wait_timeout=30)  # cleanup only, idempotent
        await asyncio.sleep(10)


def _log_timeout_diagnostic(t0: float, instance_id: str) -> None:
    # Dump pending-task names when the wall-clock guard fires. Must not crash.
    try:
        elapsed = time.time() - t0
        pending = [t for t in asyncio.all_tasks() if not t.done()]
        stuck = []
        for t in pending[:5]:  # cap to avoid log spam
            coro = getattr(t, "_coro", None)
            stuck.append(getattr(coro, "__qualname__", repr(coro)))
        logger.warning(
            "[coding_agent_rl] %s: wall_clock_timeout after %.1fs "
            "(guard=%ds); %d tasks pending; sample of stuck: %s",
            instance_id,
            elapsed,
            CONFIG.rollout_guard_sec,
            len(pending),
            stuck,
        )
    except Exception:  # pragma: no cover - diag must never crash
        pass


def _session_id(sample: Sample, instance_id: str) -> str:
    if sample.session_id:
        try:
            return str(uuid.UUID(sample.session_id))
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_URL, f"slime:coding-agent:session:{sample.session_id}"))
    if sample.index is not None and sample.group_index is not None:
        return str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"slime:coding-agent:{instance_id}:{sample.index}:{sample.group_index}")
        )
    return str(uuid.uuid4())


def _abort_result(
    sample: Sample,
    reason: str,
    instance_id: str,
    *,
    status: Sample.Status = Sample.Status.TERMINAL_FAILED,
    retry_after_seconds: float | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> list[Sample]:
    """Mark ``sample`` failed and return it in the list shape this
    fan-out generate function always yields."""
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = 0.0
    sample.remove_sample = True
    sample.status = status
    sample.retry_after_seconds = retry_after_seconds
    sample.metadata = {
        **(sample.metadata or {}),
        **(extra_metadata or {}),
        "abort_reason": reason,
        "instance_id": instance_id,
        "trainable": False,
        "invalid_reason": reason,
    }
    logger.warning("[coding_agent_rl] %s failed status=%s reason=%s", instance_id, status.value, reason)
    return [sample]


def _adapter_failure_metadata(failure: AdapterFailure | None) -> dict[str, Any]:
    if failure is None:
        return {}
    return {
        "adapter_failure_request_index": failure.request_index,
        "adapter_failure_family": failure.family,
        "adapter_failure_code": failure.code,
    }


def _agent_failure_reason(result: HarnessRunResult, failure: AdapterFailure | None = None) -> str:
    if result.error_type == "server_error" and failure is not None:
        if failure.family in {"stream_interrupted", "stream_protocol_error"}:
            return failure.family
    return str(result.error_type or result.terminal_reason or result.exit_code)


def _agent_failure_status(
    result: HarnessRunResult,
    failure: AdapterFailure | None = None,
) -> Sample.Status:
    if result.error_type == "server_error" and failure is not None:
        if failure.family == "stream_interrupted":
            return Sample.Status.ABORTED
        if failure.family == "stream_protocol_error":
            return Sample.Status.FAILED
    message = (result.error_message or "").lower()
    if result.error_type == "server_error" and "session closed" not in message:
        return Sample.Status.FAILED
    return Sample.Status.TERMINAL_FAILED


def _eval_result(
    sample: Sample,
    *,
    reward: float,
    applied_cleanly: bool,
    agent_exit_code: int | None,
    instance_id: str,
    extra_metadata: dict[str, Any] | None = None,
) -> list[Sample]:
    """Eval-path placeholder: only ``reward`` matters for ``eval/sweb``."""

    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = float(reward)
    sample.remove_sample = True
    sample.status = Sample.Status.COMPLETED
    sample.metadata = {
        **(sample.metadata or {}),
        **(extra_metadata or {}),
        "instance_id": instance_id,
        "grading_solved": float(reward) == 1.0,
        "applied_cleanly": applied_cleanly,
        "agent_exit_code": agent_exit_code,
        "trainable": False,
        "invalid_reason": "evaluation_only",
    }
    return [sample]
