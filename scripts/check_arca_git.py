#!/usr/bin/env python3
"""Create one ARCA sandbox and verify Git is usable in its SWE worktree."""

from __future__ import annotations

import asyncio
import os

from slime.agent.sandbox import create_sandbox


async def main() -> None:
    image = os.environ.get("SLIME_AGENT_ARCA_GIT_TEST_IMAGE", "local/astropy__astropy-13398")
    workdir = os.environ.get("SLIME_AGENT_ARCA_GIT_TEST_WORKDIR", "/testbed")

    async with create_sandbox(image, metadata={"role": "git-smoke"}) as sandbox:
        _, version, _ = await sandbox.exec("git --version", user=sandbox.work_user, check=True, timeout=30)
        _, status, _ = await sandbox.exec(
            f"cd {workdir} && git status --short",
            user=sandbox.work_user,
            check=True,
            timeout=30,
        )

    print(f"ARCA_GIT_OK image={image!r} workdir={workdir!r} version={version.strip()!r} status={status.strip()!r}")


if __name__ == "__main__":
    asyncio.run(main())
