"""CPU-only contract tests for the optional ARCA sandbox backend."""

from __future__ import annotations

import asyncio
import logging
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.agent import sandbox as sandbox_mod  # noqa: E402

NUM_GPUS = 0


class FakeResourceSpecification:
    def __init__(self, cpu: int, memory: int, disk: float | None = None):
        self.cpu = cpu
        self.memory = memory
        self.disk = disk


class FakeTerminal:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    async def exec_command(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if self.responses:
            value = self.responses.pop(0)
            if isinstance(value, BaseException):
                raise value
            return value
        if cmd == "echo ready":
            return SimpleNamespace(exit_code=0, stdout="ready\n", stderr="")
        return SimpleNamespace(exit_code=0, stdout="", stderr="")


class FakeFilesystem:
    def __init__(self):
        self.writes = []
        self.uploads = []
        self.reads = []

    async def write(self, path, content):
        self.writes.append((path, content))
        return SimpleNamespace(file_path=path)

    async def upload(self, source, dest):
        self.uploads.append((source, dest))
        return SimpleNamespace(success=True)

    async def read(self, path):
        self.reads.append(path)
        return SimpleNamespace(content="file-body")


class FakeProviderSandbox:
    def __init__(self, *, sandbox_id="arca-1", terminal=None, destroy_success=True):
        self.id = sandbox_id
        self.terminal = terminal or FakeTerminal()
        self.filesystem = FakeFilesystem()
        self.destroy_success = destroy_success
        self.destroy_calls = 0

    async def destroy(self):
        self.destroy_calls += 1
        return SimpleNamespace(success=self.destroy_success)


class FakeFactory:
    created = []
    config_observations = []
    provider = FakeProviderSandbox()
    create_error = None

    def __init__(self, *, config_file):
        path = Path(config_file)
        self.__class__.config_observations.append(
            {
                "path": path,
                "mode": stat.S_IMODE(path.stat().st_mode),
                "text": path.read_text(),
            }
        )

    async def create_async_sandbox(self, **kwargs):
        self.__class__.created.append(kwargs)
        if self.__class__.create_error is not None:
            raise self.__class__.create_error
        return self.__class__.provider


@pytest.fixture(autouse=True)
def _reset_arca(monkeypatch):
    FakeFactory.created = []
    FakeFactory.config_observations = []
    FakeFactory.provider = FakeProviderSandbox()
    FakeFactory.create_error = None
    sandbox_mod._reset_arca_factory_for_tests()
    monkeypatch.setattr(
        sandbox_mod,
        "_load_arca_sdk",
        lambda: (FakeFactory, FakeResourceSpecification),
    )
    for key in tuple(os.environ):
        if key.startswith("SLIME_ARCA_") or key.startswith("SLIME_AGENT_ARCA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("SLIME_AGENT_SANDBOX_BACKEND", raising=False)
    monkeypatch.setenv("SLIME_ARCA_APP_NAME", "a3training")
    monkeypatch.setenv("SLIME_ARCA_BASE_URL", "https://arca.example.test")
    monkeypatch.setenv("SLIME_ARCA_API_KEY", "secret-do-not-log")
    monkeypatch.setenv("SLIME_AGENT_ARCA_TEMPLATE_ID", "ARCA-TEMPLATE-test")


def test_backend_defaults_to_e2b_and_arca_is_explicit(monkeypatch):
    assert isinstance(sandbox_mod.create_sandbox("image:tag"), sandbox_mod.E2BSandbox)

    monkeypatch.setenv("SLIME_AGENT_SANDBOX_BACKEND", "arca")
    sb = sandbox_mod.create_sandbox("image:tag", metadata={"role": "agent"})
    assert isinstance(sb, sandbox_mod.ArcaSandbox)
    assert sb.work_user == "admin"
    assert sb.privileged_user == "admin"
    assert sb.home_dir == "/home/admin"
    assert sb.cli_preinstalled is True


def test_factory_uses_0600_short_lived_yaml_without_logging_secret(caplog):
    caplog.set_level(logging.DEBUG)
    factory = sandbox_mod._get_arca_factory()

    assert isinstance(factory, FakeFactory)
    observation = FakeFactory.config_observations[0]
    assert observation["mode"] == 0o600
    assert "a3training" in observation["text"]
    assert "https://arca.example.test" in observation["text"]
    assert "secret-do-not-log" in observation["text"]
    assert not observation["path"].exists()
    assert "secret-do-not-log" not in caplog.text

    assert sandbox_mod._get_arca_factory() is factory
    assert len(FakeFactory.config_observations) == 1


def test_arca_create_is_async_once_and_polls_terminal_until_ready(monkeypatch):
    FakeFactory.provider = FakeProviderSandbox(
        terminal=FakeTerminal(
            [
                RuntimeError("502 connection refused"),
                SimpleNamespace(exit_code=0, stdout="ready\n", stderr=""),
            ]
        )
    )
    monkeypatch.setenv("SLIME_AGENT_ARCA_READY_POLL_INTERVAL_SEC", "0")
    metadata = {"instance_id": "astropy__astropy-12907", "role": "agent", "attempt": "1"}

    async def run_case():
        sb = sandbox_mod.ArcaSandbox("asr.example/swebench:instance-v1", metadata=metadata)
        entered = await sb.__aenter__()
        assert entered is sb
        assert sb.sandbox_id == "arca-1"
        await sb.__aexit__(None, None, None)

    asyncio.run(run_case())

    assert len(FakeFactory.created) == 1
    kwargs = FakeFactory.created[0]
    assert kwargs["template_id"] == "ARCA-TEMPLATE-test"
    assert kwargs["image"] == "asr.example/swebench:instance-v1"
    assert kwargs["ttl_in_minutes"] == 40
    assert (kwargs["resource_spec"].cpu, kwargs["resource_spec"].memory, kwargs["resource_spec"].disk) == (
        2,
        4,
        25,
    )
    assert kwargs["metadata"] == metadata
    assert "command" not in kwargs
    assert len(FakeFactory.provider.terminal.calls) == 2
    assert all(call[1]["user"] == "admin" for call in FakeFactory.provider.terminal.calls)
    assert FakeFactory.provider.destroy_calls == 1


def test_arca_image_map_hit_resolves_local_key_before_create(monkeypatch, tmp_path):
    image_map = tmp_path / "arca-images.json"
    image_map.write_text('{"astropy__astropy-14508": "asr.example/swebench:astropy__astropy-14508-v1"}')
    monkeypatch.setenv("SLIME_AGENT_ARCA_IMAGE_MAP", str(image_map))

    async def run_case():
        sb = sandbox_mod.ArcaSandbox("local/astropy__astropy-14508")
        await sb.__aenter__()
        await sb.__aexit__(None, None, None)

    asyncio.run(run_case())

    assert FakeFactory.created[0]["image"] == "asr.example/swebench:astropy__astropy-14508-v1"


def test_arca_image_map_miss_fails_before_create(monkeypatch, tmp_path):
    image_map = tmp_path / "arca-images.json"
    image_map.write_text("{}")
    monkeypatch.setenv("SLIME_AGENT_ARCA_IMAGE_MAP", str(image_map))

    async def run_case():
        sb = sandbox_mod.ArcaSandbox("local/astropy__astropy-unknown")
        with pytest.raises(RuntimeError, match="local/astropy__astropy-unknown") as exc_info:
            await sb.__aenter__()
        assert not isinstance(exc_info.value, sandbox_mod.AmbiguousCreate)

    asyncio.run(run_case())

    assert FakeFactory.created == []


def test_arca_image_map_local_key_requires_config(monkeypatch):
    monkeypatch.delenv("SLIME_AGENT_ARCA_IMAGE_MAP", raising=False)

    async def run_case():
        sb = sandbox_mod.ArcaSandbox("local/astropy__astropy-14508")
        with pytest.raises(RuntimeError, match="SLIME_AGENT_ARCA_IMAGE_MAP") as exc_info:
            await sb.__aenter__()
        assert not isinstance(exc_info.value, sandbox_mod.AmbiguousCreate)

    asyncio.run(run_case())

    assert FakeFactory.created == []


@pytest.mark.parametrize(
    "contents",
    [
        "[]",
        '{"astropy__astropy-14508": ""}',
        '{"astropy__astropy-14508": "local/astropy__astropy-14508"}',
        '{"astropy__astropy-14508": " asr.example/swebench:astropy-v1"}',
    ],
)
def test_arca_image_map_rejects_invalid_mapping_before_create(monkeypatch, tmp_path, contents):
    image_map = tmp_path / "arca-images.json"
    image_map.write_text(contents)
    monkeypatch.setenv("SLIME_AGENT_ARCA_IMAGE_MAP", str(image_map))

    async def run_case():
        sb = sandbox_mod.ArcaSandbox("local/astropy__astropy-14508")
        with pytest.raises(RuntimeError, match="ARCA image map") as exc_info:
            await sb.__aenter__()
        assert not isinstance(exc_info.value, sandbox_mod.AmbiguousCreate)

    asyncio.run(run_case())

    assert FakeFactory.created == []


def test_arca_image_map_rejects_non_utf8_file_before_create(monkeypatch, tmp_path):
    image_map = tmp_path / "arca-images.json"
    image_map.write_bytes(b"\xff")
    monkeypatch.setenv("SLIME_AGENT_ARCA_IMAGE_MAP", str(image_map))

    async def run_case():
        sb = sandbox_mod.ArcaSandbox("local/astropy__astropy-14508")
        with pytest.raises(RuntimeError, match="ARCA image map") as exc_info:
            await sb.__aenter__()
        assert not isinstance(exc_info.value, sandbox_mod.AmbiguousCreate)

    asyncio.run(run_case())

    assert FakeFactory.created == []


def test_arca_image_map_passthrough_preserves_complete_reference(monkeypatch):
    monkeypatch.delenv("SLIME_AGENT_ARCA_IMAGE_MAP", raising=False)
    image = "asr.example/swebench:astropy__astropy-14508-v1"

    async def run_case():
        sb = sandbox_mod.ArcaSandbox(image)
        await sb.__aenter__()
        await sb.__aexit__(None, None, None)

    asyncio.run(run_case())

    assert FakeFactory.created[0]["image"] == image


def test_arca_exec_and_filesystem_use_async_sdk_and_reject_non_admin(tmp_path):
    provider = FakeProviderSandbox(
        terminal=FakeTerminal(
            [
                SimpleNamespace(exit_code=0, stdout="ready\n", stderr=""),
                SimpleNamespace(exit_code=7, stdout="out", stderr="err"),
            ]
        )
    )
    FakeFactory.provider = provider
    host_file = tmp_path / "payload.bin"
    host_file.write_bytes(b"payload")

    async def run_case():
        sb = sandbox_mod.ArcaSandbox("image")
        await sb.__aenter__()
        assert await sb.exec("false", user="admin") == (7, "out", "err")
        await sb.write_file("/testbed/text", "hello", user="admin")
        await sb.write_file("/testbed/payload", host_file, user="admin")
        assert await sb.read_file("/testbed/text", user="admin") == "file-body"
        with pytest.raises(ValueError, match="admin"):
            await sb.exec("id", user="root")
        with pytest.raises(ValueError, match="admin"):
            await sb.write_file("/tmp/x", "x", user="root")
        await sb.__aexit__(None, None, None)

    asyncio.run(run_case())

    assert provider.terminal.calls[1][1]["timeout_in_millis"] == 120_000
    assert provider.filesystem.writes == [("/testbed/text", "hello")]
    assert provider.filesystem.uploads == [(str(host_file), "/testbed/payload")]


def test_ambiguous_create_is_recognizable_and_not_retried():
    FakeFactory.create_error = RuntimeError("request outcome unknown")

    async def run_case():
        sb = sandbox_mod.ArcaSandbox("image")
        with pytest.raises(sandbox_mod.AmbiguousCreate):
            await sb.__aenter__()

    asyncio.run(run_case())
    assert len(FakeFactory.created) == 1


def test_readiness_timeout_destroys_known_sandbox(monkeypatch):
    provider = FakeProviderSandbox(terminal=FakeTerminal([RuntimeError("not ready")] * 10))
    FakeFactory.provider = provider
    monkeypatch.setenv("SLIME_AGENT_ARCA_READY_TIMEOUT_SEC", "0")
    monkeypatch.setenv("SLIME_AGENT_ARCA_READY_POLL_INTERVAL_SEC", "0")

    async def run_case():
        sb = sandbox_mod.ArcaSandbox("image")
        with pytest.raises(RuntimeError, match="terminal readiness"):
            await sb.__aenter__()

    asyncio.run(run_case())
    assert provider.destroy_calls == 1


def test_readiness_deadline_bounds_long_poll_interval(monkeypatch):
    provider = FakeProviderSandbox(terminal=FakeTerminal([RuntimeError("not ready")] * 10))
    FakeFactory.provider = provider
    monkeypatch.setenv("SLIME_AGENT_ARCA_READY_TIMEOUT_SEC", "0.01")
    monkeypatch.setenv("SLIME_AGENT_ARCA_READY_POLL_INTERVAL_SEC", "60")

    async def run_case():
        sb = sandbox_mod.ArcaSandbox("image")
        with pytest.raises(RuntimeError, match="terminal readiness"):
            await asyncio.wait_for(sb.__aenter__(), timeout=0.25)

    asyncio.run(run_case())
    assert provider.destroy_calls == 1


def test_failed_cleanup_after_readiness_failure_is_recognizable(monkeypatch):
    provider = FakeProviderSandbox(
        terminal=FakeTerminal([RuntimeError("not ready")] * 10),
        destroy_success=False,
    )
    FakeFactory.provider = provider
    monkeypatch.setenv("SLIME_AGENT_ARCA_READY_TIMEOUT_SEC", "0")
    monkeypatch.setenv("SLIME_AGENT_ARCA_READY_POLL_INTERVAL_SEC", "0")

    async def run_case():
        sb = sandbox_mod.ArcaSandbox("image")
        with pytest.raises(sandbox_mod.UnreleasedSandbox, match="arca-1"):
            await sb.__aenter__()

    asyncio.run(run_case())
    assert provider.destroy_calls == 1


def test_destroy_failure_is_reported_with_bounded_diagnostics(caplog):
    provider = FakeProviderSandbox(destroy_success=False)
    FakeFactory.provider = provider
    caplog.set_level(logging.WARNING)

    async def run_case():
        sb = sandbox_mod.ArcaSandbox("image")
        await sb.__aenter__()
        assert await sb.destroy() is False

    asyncio.run(run_case())
    assert provider.destroy_calls == 1
    assert "arca-1" in caplog.text
    assert "destroy" in caplog.text.lower()


def test_optional_sdk_import_fails_only_when_arca_is_selected(monkeypatch):
    monkeypatch.setattr(
        sandbox_mod,
        "_load_arca_sdk",
        lambda: (_ for _ in ()).throw(RuntimeError("install arca-sandbox==1.1.0")),
    )
    sandbox_mod._reset_arca_factory_for_tests()

    assert isinstance(sandbox_mod.create_sandbox("image"), sandbox_mod.E2BSandbox)
    monkeypatch.setenv("SLIME_AGENT_SANDBOX_BACKEND", "arca")

    async def run_case():
        with pytest.raises(RuntimeError, match="arca-sandbox"):
            async with sandbox_mod.create_sandbox("image"):
                pass

    asyncio.run(run_case())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
