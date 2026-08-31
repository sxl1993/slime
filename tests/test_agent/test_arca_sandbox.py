"""CPU-only contract tests for the optional ARCA sandbox backend."""

from __future__ import annotations

import asyncio
import logging
import os
import stat
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

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
        self.downloads = []

    async def write(self, path, content):
        self.writes.append((path, content))
        return SimpleNamespace(file_path=path)

    async def upload(self, source, dest):
        self.uploads.append((source, dest))
        return SimpleNamespace(success=True)

    async def read(self, path, *, raw=False):
        self.reads.append((path, raw))
        return SimpleNamespace(content=b"file-body")

    async def download(self, source, dest, *, timeout_in_millis):
        self.downloads.append((source, dest, timeout_in_millis))
        Path(dest).write_bytes(b"downloaded-body")


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
    instances = []
    provider = FakeProviderSandbox()
    create_error = None

    def __init__(self, *, config_file):
        self.closed = False
        self.__class__.instances.append(self)
        path = Path(config_file)
        self.__class__.config_observations.append(
            {
                "path": path,
                "mode": stat.S_IMODE(path.stat().st_mode),
                "text": path.read_text(),
            }
        )

    def close(self):
        self.closed = True

    async def create_async_sandbox(self, **kwargs):
        self.__class__.created.append(kwargs)
        if self.__class__.create_error is not None:
            raise self.__class__.create_error
        return self.__class__.provider


def _sdk_create_error(message: str) -> RuntimeError:
    try:
        raise RuntimeError(message)
    except RuntimeError as cause:
        try:
            raise RuntimeError("Failed to create async sandbox") from cause
        except RuntimeError as error:
            return error


@pytest.fixture(autouse=True)
def _reset_arca(monkeypatch):
    FakeFactory.created = []
    FakeFactory.config_observations = []
    FakeFactory.instances = []
    FakeFactory.provider = FakeProviderSandbox()
    FakeFactory.create_error = None
    arca_module = types.ModuleType("arca")
    arca_module.SandboxFactory = FakeFactory
    arca_model_module = types.ModuleType("arca.model")
    arca_sandbox_module = types.ModuleType("arca.model.sandbox")
    arca_sandbox_module.ResourceSpecification = FakeResourceSpecification
    arca_module.model = arca_model_module
    arca_model_module.sandbox = arca_sandbox_module
    monkeypatch.setitem(sys.modules, "arca", arca_module)
    monkeypatch.setitem(sys.modules, "arca.model", arca_model_module)
    monkeypatch.setitem(sys.modules, "arca.model.sandbox", arca_sandbox_module)
    for key in tuple(os.environ):
        if key.startswith("SLIME_AGENT_ARCA_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv("SLIME_AGENT_SANDBOX_BACKEND", raising=False)
    monkeypatch.setenv("SLIME_AGENT_ARCA_APP_NAME", "a3training")
    monkeypatch.setenv("SLIME_AGENT_ARCA_BASE_URL", "https://arca.example.test")
    monkeypatch.setenv("SLIME_AGENT_ARCA_API_KEY", "secret-do-not-log")
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


def test_arca_factory_uses_0600_short_lived_yaml_without_logging_secret(caplog):
    caplog.set_level(logging.DEBUG)
    sb = sandbox_mod.ArcaSandbox("image")

    async def run_case():
        await sb.__aenter__()
        await sb.__aexit__(None, None, None)

    asyncio.run(run_case())

    factory = FakeFactory.instances[0]
    observation = FakeFactory.config_observations[0]
    config = yaml.safe_load(observation["text"])
    assert observation["mode"] == 0o600
    assert config == {
        "app_name": "a3training",
        "sandbox": {
            "base_url": "https://arca.example.test",
            "api_key": "secret-do-not-log",
        },
    }
    assert not observation["path"].exists()
    assert "secret-do-not-log" not in caplog.text
    assert len(FakeFactory.config_observations) == 1
    assert factory.closed is True


def test_arca_create_is_async_once_and_returns_ready_sandbox():
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
    assert FakeFactory.provider.terminal.calls == []
    assert FakeFactory.provider.destroy_calls == 1
    assert FakeFactory.instances[0].closed is True


def test_arca_local_key_resolves_canonical_tag_before_create(monkeypatch):
    monkeypatch.delenv("SLIME_AGENT_ARCA_IMAGE_REGISTRY", raising=False)
    monkeypatch.delenv("SLIME_AGENT_ARCA_IMAGE_TAG_SUFFIX", raising=False)

    async def run_case():
        sb = sandbox_mod.ArcaSandbox("local/astropy__astropy-14508")
        await sb.__aenter__()
        await sb.__aexit__(None, None, None)

    asyncio.run(run_case())

    assert FakeFactory.created[0]["image"] == (
        "asr.antgroup-inc.cn/arcaslimeagentrl/sweb.instance:astropy__astropy-14508-claude-code-2.1.220-latest"
    )


def test_arca_local_key_resolution_honors_env_overrides(monkeypatch):
    monkeypatch.setenv("SLIME_AGENT_ARCA_IMAGE_REGISTRY", "asr.example/custom")
    monkeypatch.setenv("SLIME_AGENT_ARCA_IMAGE_TAG_SUFFIX", "claude-code-9.9.9-v7")

    async def run_case():
        sb = sandbox_mod.ArcaSandbox("local/astropy__astropy-14508")
        await sb.__aenter__()
        await sb.__aexit__(None, None, None)

    asyncio.run(run_case())

    assert FakeFactory.created[0]["image"] == "asr.example/custom:astropy__astropy-14508-claude-code-9.9.9-v7"


def test_arca_local_key_empty_instance_id_fails_before_create():
    async def run_case():
        sb = sandbox_mod.ArcaSandbox("local/")
        with pytest.raises(RuntimeError, match="empty instance ID") as exc_info:
            await sb.__aenter__()
        assert not isinstance(exc_info.value, sandbox_mod.SandboxLeaseError)

    asyncio.run(run_case())

    assert FakeFactory.created == []


def test_arca_image_passthrough_preserves_complete_reference():
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
                SimpleNamespace(exit_code=7, stdout="out", stderr="err"),
            ]
        )
    )
    FakeFactory.provider = provider
    host_file = tmp_path / "payload.bin"
    host_file.write_bytes(b"payload")
    downloaded_file = tmp_path / "downloaded.bin"

    async def run_case():
        sb = sandbox_mod.ArcaSandbox("image")
        await sb.__aenter__()
        assert await sb.exec("false", user="admin") == (7, "out", "err")
        await sb.write_file("/testbed/text", "hello", user="admin")
        await sb.write_file("/testbed/payload", host_file, user="admin")
        assert await sb.read_file("/testbed/text", user="admin") == "file-body"
        await sb.download_file("/testbed/large", downloaded_file, user="admin")
        with pytest.raises(ValueError, match="admin"):
            await sb.exec("id", user="root")
        with pytest.raises(ValueError, match="admin"):
            await sb.write_file("/tmp/x", "x", user="root")
        await sb.__aexit__(None, None, None)

    asyncio.run(run_case())

    assert provider.terminal.calls[0][1]["timeout_in_millis"] == 120_000
    assert provider.filesystem.writes == [("/testbed/text", "hello")]
    assert provider.filesystem.uploads == [(str(host_file), "/testbed/payload")]
    assert provider.filesystem.reads == [("/testbed/text", True)]
    assert provider.filesystem.downloads == [("/testbed/large", str(downloaded_file), 600_000)]
    assert downloaded_file.read_bytes() == b"downloaded-body"


def test_ambiguous_create_is_recognizable_and_not_retried():
    FakeFactory.create_error = RuntimeError("request outcome unknown")

    async def run_case():
        sb = sandbox_mod.ArcaSandbox("image")
        with pytest.raises(sandbox_mod.SandboxLeaseError):
            await sb.__aenter__()

    asyncio.run(run_case())
    assert len(FakeFactory.created) == 1
    assert FakeFactory.instances[0].closed is True


def test_lifecycle_rate_limit_is_recognizable_as_safe_to_retry():
    FakeFactory.create_error = _sdk_create_error(
        '429: {"code":42911,"message":"Rate limit exceeded","success":false,'
        '"data":{"limitType":"LIFECYCLE","limit":250,"retryAfter":1}}'
    )

    async def run_case():
        sb = sandbox_mod.ArcaSandbox("image")
        with pytest.raises(sandbox_mod.SandboxCreateRateLimitError) as exc_info:
            await sb.__aenter__()
        assert exc_info.value.retry_after == 1.0

    asyncio.run(run_case())
    assert len(FakeFactory.created) == 1
    assert FakeFactory.instances[0].closed is True


def test_non_lifecycle_rate_limit_remains_ambiguous():
    FakeFactory.create_error = _sdk_create_error(
        '429: {"code":42911,"message":"Rate limit exceeded","success":false,'
        '"data":{"limitType":"QUERY","limit":250,"retryAfter":1}}'
    )

    async def run_case():
        sb = sandbox_mod.ArcaSandbox("image")
        with pytest.raises(sandbox_mod.SandboxLeaseError):
            await sb.__aenter__()

    asyncio.run(run_case())
    assert len(FakeFactory.created) == 1
    assert FakeFactory.instances[0].closed is True


def test_ambiguous_create_without_id_closes_factory():
    FakeFactory.provider = FakeProviderSandbox(sandbox_id="")

    async def run_case():
        sb = sandbox_mod.ArcaSandbox("image")
        with pytest.raises(sandbox_mod.SandboxLeaseError, match="without a sandbox ID"):
            await sb.__aenter__()

    asyncio.run(run_case())
    assert FakeFactory.instances[0].closed is True


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
    assert isinstance(sandbox_mod.create_sandbox("image"), sandbox_mod.E2BSandbox)
    monkeypatch.setenv("SLIME_AGENT_SANDBOX_BACKEND", "arca")
    monkeypatch.setitem(sys.modules, "arca", None)
    monkeypatch.setitem(sys.modules, "arca.model.sandbox", None)

    async def run_case():
        with pytest.raises(RuntimeError, match="arca-sandbox"):
            async with sandbox_mod.create_sandbox("image"):
                pass

    asyncio.run(run_case())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
