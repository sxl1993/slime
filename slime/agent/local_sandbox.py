"""Local process-level sandbox — runs commands as subprocesses in per-instance workspaces.

Drop-in replacement for ``E2BSandbox`` when ``SLIME_SANDBOX_PROVIDER=local``.
Each sandbox gets an isolated directory tree under ``LOCAL_SANDBOX_WORKSPACE_ROOT``
that mirrors the path layout the SWE harness expects (``/testbed``, ``/workspace``,
``/home/agent``, ``/tmp``). Commands execute via ``asyncio.subprocess`` with
environment variable overrides in place of OS-level user switching.

When ``repo`` and ``base_commit`` are provided, ``__aenter__`` lazily creates a
git worktree from a shared per-repo clone under ``_repo_clones/``, avoiding the
need to pre-create 500 full checkouts.

Each sandbox receives a unique ``sandbox_id`` embedded in the workspace path so
that `n-samples-per-prompt > 1` (concurrent sandboxes for the same instance)
never collide on the ``testbed`` directory.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pwd
import shlex
import shutil
import uuid
from pathlib import Path

from .sandbox import ExecResult, FileContent

logger = logging.getLogger(__name__)

# Per-repo lock serializing concurrent worktree add/remove on the same clone.
_repo_locks: dict[str, asyncio.Lock] = {}

# Directories created inside each workspace root (testbed is handled by worktree).
_WORKSPACE_SUBDIRS = ("workspace", "home/agent", "home/root", "tmp", "root")

# Sandbox-absolute paths that should be bind-mounted from the workspace inside
# each exec()'s mount namespace.  Each entry is (sandbox_path, workspace_subpath).
# /tmp is bind-mounted because exec_and_wait writes launcher scripts and
# done-marker files via write_file (which resolves /tmp to workspace/tmp), and
# the mount namespace must see them at /tmp for `setsid bash /tmp/...` to work.
# This is safe because each exec() runs in its own mount namespace.
_BIND_MOUNTS = [
    ("/testbed", "testbed"),
    ("/workspace", "workspace"),
    ("/home/agent", "home/agent"),
    ("/home/root", "home/root"),
    ("/tmp", "tmp"),
]

# Shared per-repo clones and bare repos live under the workspace root.
_CLONES_SUBDIR = "_repo_clones"
_BARE_SUBDIR = "_bare_repos"


class LocalSandbox:
    """Process-level sandbox backed by local directories and subprocess execution."""

    def __init__(
        self,
        image: str,
        *,
        instance_id: str | None = None,
        repo: str | None = None,
        base_commit: str | None = None,
        version: str | None = None,
    ) -> None:
        self.image = image
        self.instance_id = instance_id or image
        self.repo = repo
        self.base_commit = base_commit
        self.version = version

        # Generate a unique id up front so the workspace path is always unique,
        # even when n-samples-per-prompt > 1 creates concurrent sandboxes for
        # the same instance_id.
        self.sandbox_id = uuid.uuid4().hex[:8]

        base = os.environ.get("LOCAL_SANDBOX_WORKSPACE_ROOT", "/tmp/slime_sandbox")
        # {root}/{instance_id}-{sandbox_id} isolates parallel samples.
        self.workspace_root = Path(base) / f"{self.instance_id}-{self.sandbox_id}"
        self._clone_dir: Path | None = None  # set during __aenter__ if repo given

    # ---- path mapping ----------------------------------------------------

    def _resolve_path(self, sandbox_path: str) -> Path:
        """Map a sandbox-absolute path to a workspace-root-relative path.

        ``/opt/miniconda3/...`` is the one exception — it maps to the real
        host path because the conda installation is shared globally.
        """
        p = Path(sandbox_path)
        if not p.is_absolute():
            return self.workspace_root / p
        parts = p.parts
        if parts[:3] == ("/", "opt", "miniconda3"):
            return Path(sandbox_path)
        return self.workspace_root / Path(*parts[1:])

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    async def _run(cmd: str, timeout: int = 30) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return -1, "", "timeout"
        code = proc.returncode if proc.returncode is not None else -1
        return code, stdout.decode(errors="replace"), stderr.decode(errors="replace")

    def _get_base_dir(self) -> Path:
        """The workspace root's parent — holds _repo_clones/ and _bare_repos/."""
        base = os.environ.get("LOCAL_SANDBOX_WORKSPACE_ROOT", "/tmp/slime_sandbox")
        return Path(base)

    def _get_bare_repo_path(self) -> Path:
        slug = self.repo.replace("/", "_") if self.repo else ""
        return self._get_base_dir() / _BARE_SUBDIR / f"{slug}.git"

    def _get_clone_dir(self) -> Path:
        slug = self.repo.replace("/", "_") if self.repo else ""
        return self._get_base_dir() / _CLONES_SUBDIR / slug

    async def _ensure_shared_clone(self) -> Path:
        """Ensure the shared per-repo clone exists and has the base_commit."""
        clone_dir = self._get_clone_dir()
        bare_repo = self._get_bare_repo_path()

        # Create the clone once (shared across all instances of the same repo)
        if not (clone_dir / ".git").exists():
            clone_dir.parent.mkdir(parents=True, exist_ok=True)
            logger.info("[local_sandbox] cloning %s ...", self.repo)
            if bare_repo.exists():
                cmd = f"git clone {bare_repo} {clone_dir}"
            else:
                cmd = f"git clone https://github.com/{self.repo}.git {clone_dir}"
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=600)

        # Ensure base_commit is available
        check = await asyncio.create_subprocess_exec(
            "bash", "-c", f"cd {clone_dir} && git cat-file -t {self.base_commit}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await check.communicate()
        if check.returncode != 0:
            logger.info("[local_sandbox] fetching commit %s for %s ...", self.base_commit[:12], self.repo)
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", f"cd {clone_dir} && git fetch origin {self.base_commit}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=120)

        return clone_dir

    # ---- context manager -------------------------------------------------

    @staticmethod
    def _get_repo_lock(repo: str) -> asyncio.Lock:
        """Return a per-repo asyncio.Lock that serializes worktree operations."""
        if repo not in _repo_locks:
            _repo_locks[repo] = asyncio.Lock()
        return _repo_locks[repo]

    async def __aenter__(self) -> LocalSandbox:
        self.workspace_root.mkdir(parents=True, exist_ok=True)

        # Create subdirectories (testbed is handled by worktree if repo given)
        for d in _WORKSPACE_SUBDIRS:
            p = self.workspace_root / d
            p.mkdir(parents=True, exist_ok=True)
            # /tmp must be world-writable (like a real /tmp) so non-root users
            # (e.g. agent) can create done-marker and output files there.
            if d == "tmp":
                os.chmod(str(p), 0o1777)

        # Lazy workspace setup via git worktree
        if self.repo and self.base_commit:
            clone_dir = await self._ensure_shared_clone()
            self._clone_dir = clone_dir

            testbed = self.workspace_root / "testbed"

            # Prune stale worktrees and remove any prior registration for our
            # target path under the per-repo lock (these mutate shared .git/worktrees/).
            lock = self._get_repo_lock(self.repo)
            async with lock:
                await self._run(f"git -C {clone_dir} worktree prune")
                await self._run(
                    f"git -C {clone_dir} worktree remove {testbed} --force 2>/dev/null || true"
                )
                if testbed.exists():
                    await self._run(f"rm -rf {testbed}")

            # git worktree add with a unique target directory is safe to run
            # concurrently — each sandbox gets its own path via sandbox_id.
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c",
                f"git -C {clone_dir} worktree add --detach {testbed} {self.base_commit}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"worktree add failed for {self.instance_id}: {stderr.decode()[:400]}"
                )
        else:
            # No repo — just create empty testbed
            (self.workspace_root / "testbed").mkdir(parents=True, exist_ok=True)

        logger.info("[local_sandbox] %s: workspace=%s", self.sandbox_id, self.workspace_root)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        cleanup = os.environ.get("LOCAL_SANDBOX_CLEANUP_ON_EXIT", "").lower() in ("1", "true", "yes")
        if cleanup and self.workspace_root.exists():
            # rm -rf via subprocess: async (non-blocking) and faster than shutil.rmtree.
            # This removes the testbed directory, making the worktree stale;
            # the next __aenter__ will prune it via `git worktree prune`.
            await self._run(f"rm -rf {self.workspace_root}", timeout=120)
            logger.info("[local_sandbox] %s: cleaned up %s", self.sandbox_id, self.workspace_root)

    # ---- env building ----------------------------------------------------

    def _build_env(self, user: str = "root", extra: dict[str, str] | None = None) -> dict[str, str]:
        """Compute the effective environment for a subprocess."""
        env = dict(os.environ)
        home_dir = self.workspace_root / "home" / user
        env["HOME"] = str(home_dir)
        env["USER"] = user
        conda_bin = "/opt/miniconda3/bin"
        existing_path = env.get("PATH", "/usr/local/bin:/usr/bin:/bin")
        if conda_bin not in existing_path:
            env["PATH"] = f"{conda_bin}:{existing_path}"
        # PYTHONPATH pointing at the host Python's site-packages (e.g. 3.12)
        # poisons the conda Python (e.g. 3.9) — the mismatched bytecode causes
        # import errors like "module 'io' has no attribute 'open'".  Strip it so
        # that each command's Python resolves packages from its own sys.path.
        env.pop("PYTHONPATH", None)
        if extra:
            env.update(extra)
        return env

    # ---- exec ------------------------------------------------------------

    def _wrap_with_mount_ns(self, cmd: str, *, user: str = "root", env: dict[str, str] | None = None) -> str:
        """Wrap *cmd* so it runs inside an isolated mount namespace with
        bind mounts mapping sandbox-absolute paths to the workspace tree.

        Each ``exec()`` gets its own mount namespace via ``unshare --mount``
        so concurrent sandboxes can bind different directories to the same
        absolute path (e.g. ``/testbed``) without collision.

        When *user* is not ``"root"``, the launcher script drops privileges
        via ``su`` so the command runs as that user inside the namespace.

        Script files are written to disk by Python (not heredocs) so the
        returned command string works correctly inside ``bash -c '...'``.
        """
        if not _BIND_MOUNTS:
            return cmd
        launcher_id = uuid.uuid4().hex[:8]
        ns_dir = self.workspace_root / "tmp"
        ns_dir.mkdir(parents=True, exist_ok=True)

        # Entry script: bind mounts then exec the launcher.
        mount_lines = []
        for sandbox_path, subpath in _BIND_MOUNTS:
            real = self.workspace_root / subpath
            mount_lines.append(
                f'mkdir -p "{real}" "{sandbox_path}" && '
                f'mount --bind "{real}" "{sandbox_path}" 2>/dev/null || true'
            )
        launcher_name = f".slime-ns-{launcher_id}.sh"
        launcher_host = str(ns_dir / launcher_name)
        launcher_ns = f"/tmp/{launcher_name}"

        # When user != root, wrap with `su` to drop privileges.  Write a
        # separate "user-cmd" script owned by the target user so `su` can
        # execute it without shell-quoting issues.
        if user != "root":
            user_cmd_name = f".slime-ns-cmd-{launcher_id}.sh"
            user_cmd_host = str(ns_dir / user_cmd_name)
            user_cmd_ns = f"/tmp/{user_cmd_name}"
            user_cmd_body = f"#!/bin/bash\n"
            if env:
                user_cmd_body += "\n".join(f"export {k}={shlex.quote(v)}" for k, v in env.items()) + "\n"
            user_cmd_body += f"cd {shlex.quote(str(self.workspace_root / 'testbed' if (self.workspace_root / 'testbed').exists() else str(self.workspace_root)))}\n"
            user_cmd_body += f"exec bash -c {shlex.quote(cmd)}\n"
            Path(user_cmd_host).write_text(user_cmd_body)
            os.chmod(user_cmd_host, 0o755)
            # chown so the agent user can read/execute it
            # pwd already imported at module level
            try:
                pw = pwd.getpwnam(user)
                os.chown(user_cmd_host, pw.pw_uid, pw.pw_gid)
            except (KeyError, PermissionError):
                pass
            launcher_body = f"#!/bin/bash\nexec su -s /bin/bash {user} {user_cmd_ns}\n"
        else:
            launcher_body = f"#!/bin/bash\nexec bash -c {shlex.quote(cmd)}\n"
        entry_body = "#!/bin/bash\n" + "\n".join(mount_lines) + f"\nexec bash {launcher_ns}\n"
        entry_name = f".slime-ns-entry-{launcher_id}.sh"
        entry_host = str(ns_dir / entry_name)

        # Write scripts directly via Python — avoids heredocs in bash -c.
        Path(launcher_host).write_text(launcher_body)
        Path(entry_host).write_text(entry_body)
        os.chmod(launcher_host, 0o755)
        os.chmod(entry_host, 0o755)

        cleanup_files = f"{launcher_host} {entry_host}"
        if user != "root":
            cleanup_files += f" {user_cmd_host}"
        return (
            f"unshare --mount bash {entry_host}\n"
            f"rm -f {cleanup_files}"
        )

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,  # accepted but ignored
    ) -> ExecResult:
        effective_env = self._build_env(user, extra=env)
        wrapped_cmd = self._wrap_with_mount_ns(cmd, user=user, env=effective_env)
        proc = await asyncio.create_subprocess_exec(
            "bash",
            "-c",
            wrapped_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.workspace_root),
            env=effective_env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return (-1, "", "timeout")
        code = proc.returncode if proc.returncode is not None else -1
        out = stdout.decode(errors="replace")
        err = stderr.decode(errors="replace")
        if check and code != 0:
            raise RuntimeError(
                f"exec failed (exit={code}): {cmd[:120]}\n{err[:400]}"
            )
        return (code, out, err)

    # ---- file I/O --------------------------------------------------------

    async def write_file(
        self,
        sandbox_path: str,
        content: FileContent,
        *,
        user: str = "root",
    ) -> None:
        resolved = self._resolve_path(sandbox_path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, Path):
            shutil.copy2(str(content), str(resolved))
        elif isinstance(content, bytes):
            resolved.write_bytes(content)
        else:
            resolved.write_text(content)
        if user != "root":
            # pwd already imported at module level
            try:
                pw = pwd.getpwnam(user)
                os.chown(str(resolved), pw.pw_uid, pw.pw_gid)
            except (KeyError, PermissionError):
                pass

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        resolved = self._resolve_path(sandbox_path)
        try:
            return resolved.read_text(errors="replace")
        except (FileNotFoundError, IsADirectoryError):
            return ""