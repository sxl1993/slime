"""Tests for LocalSandbox — process-level sandbox with mount namespace isolation.

These are pytest-style tests (no ``__main__`` script entry), distinct from the
script-style ``test_*.py`` files the CI matrix runs via ``python tests/xxx.py``.
Run locally with ``pytest tests/test_local_sandbox.py``; async cases need
``pytest-asyncio`` (``pip install pytest-asyncio``), which is a local dev
dependency, not declared in the CI pip-install line because CI does not
collect this file.
"""

import asyncio
import os

import pytest

# Guard: skip when unshare --mount is not available (e.g. some CI containers).
_unshare_available = os.system("unshare --mount true 2>/dev/null") == 0


@pytest.fixture()
def workspace_root(tmp_path):
    """Provide a temporary workspace root and set the env var."""
    old = os.environ.get("LOCAL_SANDBOX_WORKSPACE_ROOT")
    os.environ["LOCAL_SANDBOX_WORKSPACE_ROOT"] = str(tmp_path)
    yield tmp_path
    if old is None:
        os.environ.pop("LOCAL_SANDBOX_WORKSPACE_ROOT", None)
    else:
        os.environ["LOCAL_SANDBOX_WORKSPACE_ROOT"] = old


@pytest.fixture()
def sandbox(workspace_root):
    """Create a LocalSandbox with a repo-less workspace (no git needed)."""
    from slime.agent.local_sandbox import LocalSandbox

    sb = LocalSandbox(
        image="test",
        instance_id="test_instance",
    )
    return sb


# ---------------------------------------------------------------------------
# RED: exec() must resolve sandbox-absolute paths via mount namespace
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _unshare_available, reason="unshare --mount not available")
@pytest.mark.asyncio
async def test_exec_resolves_testbed_absolute_path(sandbox, workspace_root):
    """When a command references /testbed (a sandbox-absolute path),
    exec() should resolve it to {workspace_root}/testbed so the command
    succeeds even though /testbed doesn't exist on the host filesystem."""
    async with sandbox:
        # Write a marker file into workspace_root/testbed
        testbed_dir = workspace_root / f"test_instance-{sandbox.sandbox_id}" / "testbed"
        testbed_dir.mkdir(parents=True, exist_ok=True)
        (testbed_dir / "marker.txt").write_text("hello")

        # `cat /testbed/marker.txt` uses the absolute path /testbed.
        # Without mount namespace isolation this would fail because
        # /testbed doesn't exist on the host.
        code, out, err = await sandbox.exec("cat /testbed/marker.txt")

        assert code == 0, f"exit={code}, stderr={err}"
        assert "hello" in out


@pytest.mark.skipif(not _unshare_available, reason="unshare --mount not available")
@pytest.mark.asyncio
async def test_exec_chown_testbed(sandbox, workspace_root):
    """ensure_agent_user does `chown -R agent:agent /home/agent /testbed`.
    With mount namespace isolation, /testbed must resolve to the workspace
    testbed directory so chown doesn't fail with 'No such file or directory'."""
    async with sandbox:
        testbed_dir = workspace_root / f"test_instance-{sandbox.sandbox_id}" / "testbed"
        testbed_dir.mkdir(parents=True, exist_ok=True)

        # Create the agent user idempotently, then chown /testbed
        code, out, err = await sandbox.exec(
            "id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent; " "chown -R agent:agent /testbed",
            user="root",
        )
        assert code == 0, f"exit={code}, stderr={err}"


@pytest.mark.skipif(not _unshare_available, reason="unshare --mount not available")
@pytest.mark.asyncio
async def test_exec_home_agent_absolute_path(sandbox, workspace_root):
    """Commands referencing /home/agent must resolve to the workspace copy."""
    async with sandbox:
        code, out, err = await sandbox.exec("ls /home/agent")
        assert code == 0, f"exit={code}, stderr={err}"


@pytest.mark.skipif(not _unshare_available, reason="unshare --mount not available")
@pytest.mark.asyncio
async def test_exec_workspace_absolute_path(sandbox, workspace_root):
    """/workspace paths used by swe.py must resolve correctly."""
    async with sandbox:
        code, out, err = await sandbox.exec(
            "mkdir -p /workspace/test && touch /workspace/test/x && cat /workspace/test/x"
        )
        assert code == 0, f"exit={code}, stderr={err}"


@pytest.mark.skipif(not _unshare_available, reason="unshare --mount not available")
@pytest.mark.asyncio
async def test_concurrent_sandboxes_isolated(workspace_root):
    """Two sandboxes with the same instance_id must not collide on /testbed."""
    from slime.agent.local_sandbox import LocalSandbox

    sb1 = LocalSandbox(image="test", instance_id="concurrent_test")
    sb2 = LocalSandbox(image="test", instance_id="concurrent_test")

    async with sb1, sb2:
        # Write different markers into each sandbox's testbed
        tb1 = workspace_root / f"concurrent_test-{sb1.sandbox_id}" / "testbed"
        tb2 = workspace_root / f"concurrent_test-{sb2.sandbox_id}" / "testbed"
        tb1.mkdir(parents=True, exist_ok=True)
        tb2.mkdir(parents=True, exist_ok=True)
        (tb1 / "id.txt").write_text("sandbox1")
        (tb2 / "id.txt").write_text("sandbox2")

        # Run concurrently — each should see only its own /testbed
        (c1, o1, e1), (c2, o2, e2) = await asyncio.gather(
            sb1.exec("cat /testbed/id.txt"),
            sb2.exec("cat /testbed/id.txt"),
        )
        assert c1 == 0, f"sb1 exit={c1}, stderr={e1}"
        assert c2 == 0, f"sb2 exit={c2}, stderr={e2}"
        assert "sandbox1" in o1
        assert "sandbox2" in o2


@pytest.mark.skipif(not _unshare_available, reason="unshare --mount not available")
@pytest.mark.asyncio
async def test_cleanup_removes_workspace(workspace_root):
    """__aexit__ with CLEANUP_ON_EXIT=1 must remove the workspace directory."""
    from slime.agent.local_sandbox import LocalSandbox

    os.environ["LOCAL_SANDBOX_CLEANUP_ON_EXIT"] = "1"
    try:
        sb = LocalSandbox(image="test", instance_id="cleanup_test")
        async with sb:
            ws = sb.workspace_root
            assert ws.exists(), "workspace must exist inside context"

        # After __aexit__, workspace should be gone
        assert not ws.exists(), f"workspace {ws} should have been cleaned up"
    finally:
        os.environ.pop("LOCAL_SANDBOX_CLEANUP_ON_EXIT", None)


@pytest.mark.skipif(not _unshare_available, reason="unshare --mount not available")
@pytest.mark.asyncio
async def test_no_cleanup_preserves_workspace(workspace_root):
    """Without CLEANUP_ON_EXIT, __aexit__ must not remove the workspace."""
    from slime.agent.local_sandbox import LocalSandbox

    os.environ.pop("LOCAL_SANDBOX_CLEANUP_ON_EXIT", None)
    sb = LocalSandbox(image="test", instance_id="no_cleanup_test")
    async with sb:
        ws = sb.workspace_root

    # Workspace must still exist after __aexit__
    assert ws.exists(), f"workspace {ws} should survive without CLEANUP_ON_EXIT"

    # Clean up manually to avoid leaking temp dirs
    import shutil

    shutil.rmtree(ws, ignore_errors=True)


# ---------------------------------------------------------------------------
# RED: per-instance conda env clone must isolate eval writes (concurrent
# `pip install -e .` writing the shared base env's site-packages corrupts
# dist-info → reward=0 batch-wide). Root cause + fix in ADR-0007.
# ---------------------------------------------------------------------------


def _make_fake_base_env(base_dir: str) -> None:
    """Create a fake conda env layout on disk that _clone_env_for_eval's
    file-system simulation can copy from: a bin/, a site-packages/, and a
    conda-meta/history (the canary we assert stays read-only)."""
    base = os.path.join(base_dir, "env")
    os.makedirs(os.path.join(base, "bin"), exist_ok=True)
    sp = os.path.join(base, "lib", "python3.9", "site-packages")
    os.makedirs(sp, exist_ok=True)
    # A pre-existing package marker in the base env.
    with open(os.path.join(sp, "base_pkg.txt"), "w") as f:
        f.write("base")
    meta = os.path.join(base, "conda-meta")
    os.makedirs(meta, exist_ok=True)
    with open(os.path.join(meta, "history"), "w") as f:
        f.write("BASE_HISTORY_CANARY")


@pytest.mark.asyncio
async def test_concurrent_env_clones_isolate_writes(tmp_path, monkeypatch):
    """Two eval samples concurrently clone the same read-only base conda env
    and each `pip install` (simulated) into their own clone. The clones must
    be independent — A's installed package must not appear in B, and the base
    env's conda-meta must not be mutated.

    This is the unit-level lock on the root cause fixed in ADR-0007: without
    per-instance clones, every concurrent eval wrote the *shared* base env's
    site-packages and tore dist-info apart (hypothesis METADATA deleted,
    numpy half-overwritten → reward=0 for the whole batch).

    Driven via async + pytest.mark.asyncio (pytest-asyncio is a local dev
    dependency for this file; the clone's conda invocation is monkeypatched to
    a file-system copy so the test runs without conda / miniconda / GPU, per
    ADR-0007's CPU-only validation requirement).
    """
    # Import lazily so a missing swebench import (tolerated by swe.py) does
    # not break collection of the rest of this file.
    from examples.coding_agent_rl import swe

    base_root = tmp_path / "base"
    base_root.mkdir()
    _make_fake_base_env(str(base_root))

    # Replace the real conda invocation with a pure file-system clone so the
    # test runs anywhere (no conda / no miniconda needed). We copy the env
    # tree the same way `conda create --clone` would — a recursive file copy.
    def fake_clone(src_env_dir: str, dest_env_dir: str) -> None:
        import shutil

        shutil.copytree(src_env_dir, dest_env_dir)

    monkeypatch.setattr(swe, "_run_env_clone_sync", fake_clone)

    # Each clone gets its own destination under a per-instance workspace.
    clone_a = tmp_path / "ws_a" / "env"
    clone_b = tmp_path / "ws_b" / "env"
    clone_a.parent.mkdir(parents=True)
    clone_b.parent.mkdir(parents=True)

    base_env_dir = str(base_root / "env")

    # Concurrent clone — the operation under test.
    await asyncio.gather(
        swe._clone_env_for_eval(base_env_dir, str(clone_a)),
        swe._clone_env_for_eval(base_env_dir, str(clone_b)),
    )

    def site_packages(env_dir: str) -> str:
        return os.path.join(env_dir, "lib", "python3.9", "site-packages")

    # Simulate each eval's `pip install -e .` writing into ITS OWN clone's
    # site-packages (the write that previously corrupted the shared base).
    (clone_a / "lib" / "python3.9" / "site-packages" / "a_pkg.txt").write_text("a")
    (clone_b / "lib" / "python3.9" / "site-packages" / "b_pkg.txt").write_text("b")

    sp_a = os.listdir(site_packages(str(clone_a)))
    sp_b = os.listdir(site_packages(str(clone_b)))

    # (1) Clones are independent: each wrote only into its own site-packages.
    assert "a_pkg.txt" in sp_a and "b_pkg.txt" not in sp_a
    assert "b_pkg.txt" in sp_b and "a_pkg.txt" not in sp_b
    # Both clones inherited the base's pre-existing package.
    assert "base_pkg.txt" in sp_a and "base_pkg.txt" in sp_b

    # (2) The base env was never mutated by the eval writes.
    base_sp = os.listdir(site_packages(base_env_dir))
    assert "a_pkg.txt" not in base_sp and "b_pkg.txt" not in base_sp
    with open(os.path.join(base_env_dir, "conda-meta", "history")) as f:
        assert f.read() == "BASE_HISTORY_CANARY", "base env conda-meta must stay read-only"


@pytest.mark.skipif(not _unshare_available, reason="unshare --mount not available")
@pytest.mark.asyncio
async def test_concurrent_bind_mounts_isolated(tmp_path):
    """Two sandboxes each add_bind_mount different host dirs to the same
    sandbox path — each must see only its own content via exec()."""
    from slime.agent.local_sandbox import LocalSandbox

    # Create two different host directories with different markers
    dir_a = tmp_path / "env_a"
    dir_b = tmp_path / "env_b"
    dir_a.mkdir()
    dir_b.mkdir()
    (dir_a / "marker.txt").write_text("alpha")
    (dir_b / "marker.txt").write_text("bravo")

    os.environ["LOCAL_SANDBOX_WORKSPACE_ROOT"] = str(tmp_path / "ws")
    try:
        sb_a = LocalSandbox(image="test", instance_id="bind_a")
        sb_b = LocalSandbox(image="test", instance_id="bind_b")
        async with sb_a, sb_b:
            await sb_a.add_bind_mount("/opt/test_env", str(dir_a))
            await sb_b.add_bind_mount("/opt/test_env", str(dir_b))

            # Concurrent exec — each sandbox must see its own bind mount
            (ca, oa, ea), (cb, ob, eb) = await asyncio.gather(
                sb_a.exec("cat /opt/test_env/marker.txt"),
                sb_b.exec("cat /opt/test_env/marker.txt"),
            )
            assert ca == 0, f"sb_a exit={ca}, stderr={ea}"
            assert cb == 0, f"sb_b exit={cb}, stderr={eb}"
            assert "alpha" in oa, f"sb_a should see 'alpha', got: {oa!r}"
            assert "bravo" in ob, f"sb_b should see 'bravo', got: {ob!r}"
    finally:
        os.environ.pop("LOCAL_SANDBOX_WORKSPACE_ROOT", None)


@pytest.mark.asyncio
async def test_add_bind_mount_before_enter_raises(workspace_root):
    """add_bind_mount called before __aenter__ must raise RuntimeError."""
    from slime.agent.local_sandbox import LocalSandbox
    import tempfile

    sb = LocalSandbox(image="test", instance_id="pre_enter_test")
    # workspace_root doesn't exist yet — add_bind_mount must raise
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(RuntimeError, match="sandbox not entered"):
            await sb.add_bind_mount("/opt/test_env", d)
