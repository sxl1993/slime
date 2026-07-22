"""Tests for LocalSandbox — process-level sandbox with mount namespace isolation."""

import asyncio
import os
import tempfile

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
            "id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent; "
            "chown -R agent:agent /testbed",
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
        code, out, err = await sandbox.exec("mkdir -p /workspace/test && touch /workspace/test/x && cat /workspace/test/x")
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