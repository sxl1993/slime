"""Sandbox backends for agent rollouts.

The public sandbox contract is intentionally small: async context management,
command execution, and file read/write. Agent examples can build task-specific
setup, runner, and evaluator logic on top of this without depending directly on
one sandbox provider.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import random
import re
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Protocol, runtime_checkable

import yaml

logger = logging.getLogger(__name__)


ExecResult = tuple[int, str, str]
FileContent = str | bytes | Path


@runtime_checkable
class Sandbox(Protocol):
    """Minimal async sandbox interface used by agent rollouts.

    ``write_file`` accepts either in-memory content (``str``/``bytes``) or a
    host ``Path`` to stream into the sandbox.

    ``idempotent`` is a hint for the backend's transport-retry policy: callers
    mark whether re-sending the command after a severed response is safe to
    replay (see ``E2BSandbox._rpc_retry``). Backends without retries may
    ignore it.
    """

    sandbox_id: str
    work_user: str
    privileged_user: str
    home_dir: str
    cli_preinstalled: bool

    async def __aenter__(self) -> Sandbox: ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult: ...

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None: ...

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str: ...

    async def destroy(self) -> bool: ...


EXIT_TIME_BUDGET_EXCEEDED = -1


async def _await_done_marker(sb: Sandbox, done_file: str, *, user: str, time_budget_sec: int) -> int:
    """Poll a detached command's exit-code marker until it appears, returning the
    exit code (or ``EXIT_TIME_BUDGET_EXCEEDED`` if the budget runs out first).

    The 5s ``test -f && cat`` polls are deliberately short, idempotent RPCs --
    they keep the sandbox alive against idle GC while the detached command runs
    over a stream the gateway can't sever.
    """
    deadline = time.time() + time_budget_sec
    while time.time() < deadline:
        await asyncio.sleep(5)
        ec, out, _ = await sb.exec(f"test -f {done_file} && cat {done_file}", user=user, timeout=15, check=False)
        if ec == 0 and (out or "").strip():
            return int(out.strip())
    return EXIT_TIME_BUDGET_EXCEEDED


async def exec_and_wait(
    sb: Sandbox,
    *,
    cmd: str,
    time_budget_sec: int,
    tag: str,
    user: str = "root",
    env: dict[str, str] | None = None,
    workdir: str | None = None,
    out_file: str | None = None,
    want_output: bool = False,
) -> tuple[int, str]:
    """Run ``cmd`` to completion detached, returning ``(exit_code, output)``.

    A plain ``sb.exec`` keeps an HTTP/2 stream open for the command's whole
    runtime, so a long-running command (build, test suite) outlives what the
    E2B gateway will hold a single response stream open for: the stream gets
    severed mid-run and we lose the exit code with no safe way to retry a
    non-idempotent command. Instead we ``setsid`` the command fully detached,
    redirect its output to a file, and have it drop its exit code into a marker
    file. The caller side then becomes a sequence of short, idempotent RPCs --
    write the launcher, fire-and-forget the spawn, then poll for the marker (see
    ``_await_done_marker``) -- none of which depend on a stream staying alive,
    and the polling doubles as an idle-GC keepalive while the command runs.
    """
    out_file = out_file or f"/tmp/.{tag}.out"
    done_file = f"/tmp/.{tag}.done"
    launcher = f"/tmp/.{tag}.sh"
    lock_dir = f"/tmp/.{tag}.spawned"
    prefix = f"cd {workdir}\nexport HOME=/home/{user}\n" if workdir else ""
    launcher_body = f"#!/bin/bash\n{prefix}{cmd}\necho $? > {done_file}\n"
    await sb.write_file(launcher, launcher_body, user=user)

    # Clear the previous invocation's state in its own idempotent RPC, *before*
    # the guarded spawn. The mkdir guard below exists only to dedupe transport
    # retries of this one spawn (a severed response replayed by _rpc_retry); it
    # must not survive into the next logical invocation of the same tag (e.g.
    # install_npm_cli's retry loop), which would skip the spawn entirely and
    # read the previous run's stale exit-code marker. Callers must not overlap
    # two exec_and_wait calls with the same tag.
    await sb.exec(
        f"rm -rf {lock_dir}; rm -f {out_file} {done_file}",
        user=user,
        timeout=30,
        check=True,
        idempotent=True,
    )
    await sb.exec(
        f"chmod +x {launcher}; "
        f"mkdir {lock_dir} 2>/dev/null || exit 0; "
        f"setsid bash {launcher} < /dev/null > {out_file} 2>&1 &",
        user=user,
        env=env,
        timeout=30,
        check=True,
        idempotent=True,
    )
    exit_code = await _await_done_marker(sb, done_file, user=user, time_budget_sec=time_budget_sec)
    if exit_code == 0 and not want_output:
        return exit_code, ""
    if want_output:
        return exit_code, await sb.read_file(out_file, user=user)
    _, tail, _ = await sb.exec(f"tail -c 4096 {out_file} 2>/dev/null", user=user, timeout=15, check=False)
    return exit_code, tail or ""


def _getenv(*names: str, default: str = "") -> str:
    """First non-empty environment value among ``names`` (else ``default``).

    Lets a setting carry a primary name plus legacy aliases: list the canonical
    ``SLIME_AGENT_*`` name first, older names after."""
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return default


class E2BSandbox:
    """Async context manager around e2b.AsyncSandbox."""

    work_user = "agent"
    privileged_user = "root"
    home_dir = "/home/agent"
    cli_preinstalled = False

    image_metadata_key_env = ("SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY", "SWE_SANDBOX_IMAGE_METADATA_KEY")
    lifetime_sec_env = ("SLIME_AGENT_SANDBOX_LIFETIME_SEC", "SWE_SANDBOX_LIFETIME_SEC")
    rpc_retries_env = ("SLIME_AGENT_SANDBOX_RPC_RETRIES", "SWE_RPC_RETRIES")
    size_env = ("SLIME_AGENT_E2B_SANDBOX_SIZE", "SWE_E2B_SANDBOX_SIZE")

    default_lifetime_sec = 3600
    default_rpc_retries = 6
    default_size = "md"
    rpc_backoff_base_sec = 1.0
    rpc_backoff_cap_sec = 32.0

    def __init__(
        self,
        image: str,
        *,
        timeout: int | None = None,
        image_metadata_key: str | None = None,
        rpc_retries: int | None = None,
        size: str | None = None,
    ) -> None:
        self.image = image
        self.timeout = timeout if timeout is not None else self._lifetime_sec_from_env()
        self.image_metadata_key = image_metadata_key or self._image_metadata_key_from_env()
        self.rpc_retries = rpc_retries if rpc_retries is not None else self._rpc_retries_from_env()
        self.size = size if size is not None else self._size_from_env()
        self._sb = None
        self.sandbox_id = ""

    @classmethod
    def _image_metadata_key_from_env(cls) -> str | None:
        return _getenv(*cls.image_metadata_key_env) or None

    @classmethod
    def _lifetime_sec_from_env(cls) -> int:
        return int(_getenv(*cls.lifetime_sec_env, default=str(cls.default_lifetime_sec)))

    @classmethod
    def _rpc_retries_from_env(cls) -> int:
        return int(_getenv(*cls.rpc_retries_env, default=str(cls.default_rpc_retries)))

    @classmethod
    def _size_from_env(cls) -> str:
        return _getenv(*cls.size_env, default=cls.default_size)

    # Transient client-side failures safe to retry.
    _TRANSIENT_RPC_ERRORS = frozenset(
        {
            "ProtocolError",
            "LocalProtocolError",
            "WriteError",
            "ReadError",
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "RemoteProtocolError",
            "SSLError",
        }
    )

    @classmethod
    def _is_transient_rpc_error(cls, e: BaseException) -> bool:
        """True if e is a transient E2B client-side failure safe to retry."""
        name = type(e).__name__
        if name in cls._TRANSIENT_RPC_ERRORS:
            return True
        msg = str(e)
        if name == "SandboxException":
            if "does not exist" in msg or "STOPPED state" in msg:
                return False
            return True
        return False

    async def _rpc_retry(self, op_name: str, coro_factory, *, idempotent: bool = True):
        """Run coro_factory() with retries for transient E2B RPC failures.

        :param idempotent: When False, a transient failure is re-raised instead
            of retried: re-running a non-idempotent op (e.g. a process-spawning
            exec) after a severed response could double-execute it. Idempotent
            ops (the default: create / read_file / write_file / short read-only
            execs) retry as before.
        """
        last_err = None
        for attempt in range(self.rpc_retries):
            try:
                return await coro_factory()
            except Exception as e:
                if not self._is_transient_rpc_error(e):
                    raise
                if not idempotent:
                    raise
                last_err = e
                if attempt + 1 < self.rpc_retries:
                    await self._reset_conn_pool()
                    ceiling = min(self.rpc_backoff_cap_sec, self.rpc_backoff_base_sec * (2**attempt))
                    backoff = random.uniform(0.0, ceiling)
                    logger.debug(
                        "[agent.sandbox] %s transient %s, retry %d/%d in %.1fs: %s",
                        op_name,
                        type(e).__name__,
                        attempt + 1,
                        self.rpc_retries,
                        backoff,
                        str(e)[:120],
                    )
                    await asyncio.sleep(backoff)
        assert last_err is not None
        raise last_err

    async def _reset_conn_pool(self) -> None:
        """Tear down the sandbox's httpcore pool so the next RPC reconnects."""
        try:
            pool = self._sb._transport.pool  # httpcore.AsyncConnectionPool
            await pool.aclose()
        except Exception as e:
            logger.debug("[agent.sandbox] conn-pool reset skipped: %s", e)

    async def __aenter__(self) -> E2BSandbox:
        if self.image_metadata_key is None:
            raise RuntimeError(
                "SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY is not set. Export it "
                "to the metadata key your E2B gateway uses for image routing. "
                "The legacy SWE_SANDBOX_IMAGE_METADATA_KEY name is also "
                "accepted for coding-agent examples."
            )
        from e2b import AsyncSandbox  # type: ignore

        md = {self.image_metadata_key: self.image}

        if self.size:
            prefix = self.image_metadata_key.rsplit("/", 1)[0] if "/" in self.image_metadata_key else ""
            size_key = f"{prefix}/size" if prefix else "size"
            md[size_key] = self.size

        self._sb = await self._rpc_retry("create", lambda: AsyncSandbox.create(timeout=self.timeout, metadata=md))
        self.sandbox_id = self._sb.sandbox_id
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.destroy()

    async def destroy(self) -> bool:
        try:
            if self._sb is not None:
                await self._sb.kill()
                self._sb = None
            return True
        except Exception as e:
            logger.warning("[agent.sandbox] kill %s failed: %s", self.sandbox_id[:8], e)
            return False

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult:
        from e2b.sandbox.commands.command_handle import CommandExitException

        try:
            res = await self._rpc_retry(
                f"exec({cmd[:60]!r})",
                lambda: self._sb.commands.run(
                    cmd,
                    user=user,
                    envs=env,
                    timeout=timeout,
                    on_stdout=lambda s: None,
                    on_stderr=lambda s: None,
                ),
                idempotent=idempotent,
            )
            return res.exit_code, res.stdout or "", res.stderr or ""
        except CommandExitException as e:
            if check:
                raise RuntimeError(
                    f"e2b exec failed (exit={e.exit_code}): {cmd[:120]}\n{(e.stderr or '')[:400]}"
                ) from None
            return e.exit_code, e.stdout or "", e.stderr or ""

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        if isinstance(content, Path):
            host_path = content

            async def _do_path():
                with open(host_path, "rb") as fp:
                    await self._sb.files.write(
                        sandbox_path,
                        fp,
                        user=user,
                        gzip=False,
                        use_octet_stream=True,
                        request_timeout=600,
                    )

            await self._rpc_retry(f"write_file({sandbox_path} <- {host_path.name})", _do_path)
            return

        if isinstance(content, bytes):

            async def _do_bytes():
                await self._sb.files.write(
                    sandbox_path,
                    io.BytesIO(content),
                    user=user,
                    gzip=False,
                    use_octet_stream=True,
                    request_timeout=600,
                )

            await self._rpc_retry(f"write_file({sandbox_path}, bytes={len(content)})", _do_bytes)
            return

        await self._rpc_retry(
            f"write_file({sandbox_path})",
            lambda: self._sb.files.write(sandbox_path, content, user=user),
        )

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        try:
            return await self._rpc_retry(
                f"read_file({sandbox_path})",
                lambda: self._sb.files.read(sandbox_path, user=user),
            )
        except Exception:
            return ""


class SandboxLeaseError(RuntimeError):
    """The ARCA sandbox lease is unsafe for automatic retry."""


class SandboxCreateRateLimitError(RuntimeError):
    """ARCA explicitly rejected sandbox creation before allocating a lease."""

    def __init__(self, *, retry_after: float) -> None:
        self.retry_after = max(0.0, retry_after)
        super().__init__(f"ARCA lifecycle rate limit exceeded; retry after {self.retry_after:g}s")


def _arca_lifecycle_rate_limit_retry_after(error: BaseException) -> float | None:
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        compact_message = "".join(str(current).split())
        if '"code":42911' in compact_message and '"limitType":"LIFECYCLE"' in compact_message:
            match = re.search(r'"retryAfter":(\d+(?:\.\d+)?)', compact_message)
            return float(match.group(1)) if match else 1.0

        pending.extend(linked for linked in (current.__cause__, current.__context__) if linked is not None)
    return None


class ArcaImageResolver:
    """Resolve E2B-compatible local image keys for the ARCA backend.

    A ``local/<instance_id>`` key expands to the canonical registry tag
    ``<registry>:<instance_id>-<tag_suffix>``. Complete image references
    bypass resolution unchanged.
    """

    local_prefix = "local/"

    image_registry_env = "SLIME_AGENT_ARCA_IMAGE_REGISTRY"
    image_tag_suffix_env = "SLIME_AGENT_ARCA_IMAGE_TAG_SUFFIX"

    default_image_registry = "asr.antgroup-inc.cn/arcaslimeagentrl/sweb.instance"
    default_image_tag_suffix = "claude-code-2.1.220-latest"

    def __init__(self, registry: str | None = None, tag_suffix: str | None = None) -> None:
        self.registry = (
            registry if registry is not None else _getenv(self.image_registry_env, default=self.default_image_registry)
        )
        self.tag_suffix = (
            tag_suffix
            if tag_suffix is not None
            else _getenv(self.image_tag_suffix_env, default=self.default_image_tag_suffix)
        )

    def resolve(self, image: str) -> str:
        if not image.startswith(self.local_prefix):
            return image

        instance_id = image[len(self.local_prefix) :]
        if not instance_id:
            raise RuntimeError("ARCA image key 'local/' has an empty instance ID")

        return f"{self.registry}:{instance_id}-{self.tag_suffix}"


class ArcaSandbox:
    """Async ARCA sandbox backend for prebuilt coding-agent instance images."""

    work_user = "admin"
    privileged_user = "admin"
    home_dir = "/home/admin"
    cli_preinstalled = True

    template_id_env = "SLIME_AGENT_ARCA_TEMPLATE_ID"
    app_name_env = "SLIME_AGENT_ARCA_APP_NAME"
    base_url_env = "SLIME_AGENT_ARCA_BASE_URL"
    api_key_env = "SLIME_AGENT_ARCA_API_KEY"
    ttl_minutes_env = "SLIME_AGENT_ARCA_TTL_MINUTES"
    cpu_env = "SLIME_AGENT_ARCA_CPU"
    memory_env = "SLIME_AGENT_ARCA_MEMORY"
    disk_env = "SLIME_AGENT_ARCA_DISK"
    create_timeout_sec_env = "SLIME_AGENT_ARCA_CREATE_TIMEOUT_SEC"

    default_ttl_minutes = 40
    default_cpu = 2
    default_memory = 4
    default_disk = 25
    default_create_timeout_sec = 150.0

    def __init__(
        self,
        image: str,
        *,
        metadata: dict[str, str] | None = None,
        template_id: str | None = None,
        ttl_in_minutes: int | None = None,
        cpu: int | None = None,
        memory: int | None = None,
        disk: float | None = None,
        create_timeout_sec: float | None = None,
    ) -> None:
        self.image = image
        self.image_resolver = ArcaImageResolver()
        self.metadata = dict(metadata or {})
        self.template_id = template_id or _getenv(self.template_id_env)
        if not self.template_id:
            raise RuntimeError(f"{self.template_id_env} is required for the ARCA sandbox backend")
        self.ttl_in_minutes = (
            ttl_in_minutes
            if ttl_in_minutes is not None
            else int(_getenv(self.ttl_minutes_env, default=str(self.default_ttl_minutes)))
        )
        self.cpu = cpu if cpu is not None else int(_getenv(self.cpu_env, default=str(self.default_cpu)))
        self.memory = memory if memory is not None else int(_getenv(self.memory_env, default=str(self.default_memory)))
        self.disk = disk if disk is not None else float(_getenv(self.disk_env, default=str(self.default_disk)))
        self.create_timeout_sec = (
            create_timeout_sec
            if create_timeout_sec is not None
            else float(_getenv(self.create_timeout_sec_env, default=str(self.default_create_timeout_sec)))
        )
        self._sb = None
        self._factory = None
        self.sandbox_id = ""

    @staticmethod
    @contextmanager
    def _create_config_file(*, app_name: str, base_url: str, api_key: str) -> Iterator[str]:
        with tempfile.TemporaryDirectory(prefix="slime-arca-") as directory:
            config_path = Path(directory) / "config.yaml"
            config_path.touch(mode=0o600)
            with config_path.open("w", encoding="utf-8") as fp:
                yaml.safe_dump(
                    {
                        "app_name": app_name,
                        "sandbox": {"base_url": base_url, "api_key": api_key},
                    },
                    fp,
                    sort_keys=False,
                )
            yield str(config_path)

    async def __aenter__(self) -> ArcaSandbox:
        resolved_image = self.image_resolver.resolve(self.image)
        try:
            from arca import SandboxFactory  # type: ignore
            from arca.model.sandbox import ResourceSpecification  # type: ignore
        except ImportError:
            raise RuntimeError(
                "The ARCA sandbox backend requires arca-sandbox==1.1.0. "
                "Install it from the approved package index before selecting backend=arca."
            ) from None

        values = {
            name: _getenv(name)
            for name in (
                self.app_name_env,
                self.base_url_env,
                self.api_key_env,
            )
        }
        if missing := [name for name, value in values.items() if not value]:
            raise RuntimeError(f"{', '.join(missing)} is required for the ARCA sandbox backend")
        app_name = values[self.app_name_env]
        base_url = values[self.base_url_env]
        api_key = values[self.api_key_env]

        with self._create_config_file(app_name=app_name, base_url=base_url, api_key=api_key) as config_path:
            try:
                factory = SandboxFactory(config_file=config_path)
            except Exception:
                raise RuntimeError("Failed to initialize ARCA SandboxFactory") from None

        self._factory = factory
        resource_spec = ResourceSpecification(cpu=self.cpu, memory=self.memory, disk=self.disk)
        try:
            provider_sandbox = await factory.create_async_sandbox(
                template_id=self.template_id,
                ttl_in_minutes=self.ttl_in_minutes,
                resource_spec=resource_spec,
                image=resolved_image,
                timeout_in_millis=int(self.create_timeout_sec * 1000),
                metadata=self.metadata,
            )
            sandbox_id = getattr(provider_sandbox, "id", "") or ""
            if not sandbox_id:
                raise SandboxLeaseError("ARCA create returned without a sandbox ID")
        except Exception as error:
            self._factory = None
            with suppress(Exception):
                factory.close()
            if isinstance(error, SandboxLeaseError):
                raise
            if (retry_after := _arca_lifecycle_rate_limit_retry_after(error)) is not None:
                raise SandboxCreateRateLimitError(retry_after=retry_after) from error
            raise SandboxLeaseError(
                "ARCA create outcome is ambiguous because the request returned without a sandbox ID"
            ) from None

        self._sb = provider_sandbox
        self.sandbox_id = sandbox_id
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if not await self.destroy():
            raise SandboxLeaseError(f"ARCA sandbox {self.sandbox_id} could not be destroyed")

    @staticmethod
    def _require_admin(user: str) -> None:
        if user != "admin":
            raise ValueError(f"ARCA sandbox operations must explicitly use admin, got {user!r}")

    async def destroy(self) -> bool:
        provider_sandbox, self._sb = self._sb, None
        factory, self._factory = self._factory, None
        if provider_sandbox is None:
            if factory is not None:
                factory.close()
            return True

        try:
            result = await provider_sandbox.destroy()
            success = bool(getattr(result, "success", False))
        except Exception as error:
            logger.warning(
                "[agent.sandbox] ARCA destroy failed sandbox_id=%s error=%s: %s",
                self.sandbox_id,
                type(error).__name__,
                str(error)[:160],
            )
            return False
        finally:
            if factory is not None:
                with suppress(Exception):
                    factory.close()

        if not success:
            logger.warning("[agent.sandbox] ARCA destroy reported failure sandbox_id=%s", self.sandbox_id)
        return success

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "admin",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult:
        del idempotent  # ARCA SDK 1.1.0 has no client-side RPC retry knob.
        self._require_admin(user)
        result = await self._sb.terminal.exec_command(
            cmd,
            shell="bash",
            envs=env,
            user=user,
            timeout_in_millis=int(timeout * 1000),
        )
        exit_code = int(result.exit_code)
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if check and exit_code != 0:
            raise RuntimeError(f"ARCA exec failed (exit={exit_code}): {cmd[:120]}\n{stderr[:400]}")
        return exit_code, stdout, stderr

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "admin") -> None:
        self._require_admin(user)
        if isinstance(content, str):
            await self._sb.filesystem.write(sandbox_path, content)
            return

        if isinstance(content, bytes):
            with tempfile.TemporaryDirectory(prefix="slime-arca-upload-") as directory:
                source_path = Path(directory) / "payload"
                source_path.touch(mode=0o600)
                source_path.write_bytes(content)
                result = await self._sb.filesystem.upload(str(source_path), sandbox_path)
        else:
            result = await self._sb.filesystem.upload(str(content), sandbox_path)
        if not bool(getattr(result, "success", False)):
            raise RuntimeError(f"ARCA file upload failed for {sandbox_path}")

    async def read_file(self, sandbox_path: str, *, user: str = "admin") -> str:
        self._require_admin(user)
        result = await self._sb.filesystem.read(sandbox_path, raw=True)
        content = result.content
        return content.decode("utf-8") if isinstance(content, bytes) else content


def create_sandbox(image: str, *, metadata: dict[str, str] | None = None) -> Sandbox:
    """Construct the explicitly selected backend; E2B remains the default."""
    backend = _getenv("SLIME_AGENT_SANDBOX_BACKEND", default="e2b").lower()
    if backend == "e2b":
        return E2BSandbox(image)
    if backend == "arca":
        return ArcaSandbox(image, metadata=metadata)
    raise ValueError(f"Unsupported SLIME_AGENT_SANDBOX_BACKEND={backend!r}; expected 'e2b' or 'arca'")


async def ensure_agent_user(sb: Sandbox, workdir: str) -> None:
    """Create the unprivileged 'agent' user that owns workdir + can git diff."""
    await sb.exec(
        f"id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent && "
        f"chown -R agent:agent /home/agent {workdir} && "
        f"git config --system --add safe.directory '*' && id agent",
        user="root",
        check=True,
        timeout=60,
    )


async def prepare_work_user(sb: Sandbox, workdir: str) -> None:
    """Provision E2B's ``agent`` user; image-provided users need no mutation."""
    if sb.work_user == "agent":
        await ensure_agent_user(sb, workdir)
