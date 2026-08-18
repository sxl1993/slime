"""Lightweight routing tests for coding-agent ARCA integration.

Heavy adapter/model dependencies are stubbed because these tests exercise only
the sandbox boot/eval orchestration boundary.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.test_agent._fakes import FakeSandbox  # noqa: E402

NUM_GPUS = 0


@contextlib.contextmanager
def _coding_modules():
    names = (
        "slime.agent.adapters",
        "slime.agent.adapters.common",
        "slime.agent.aiohttp_threaded",
        "slime.utils.processing_utils",
        "slime.utils.types",
        "examples.coding_agent_rl.generate",
        "examples.coding_agent_rl.swe",
    )
    saved = {name: sys.modules.get(name) for name in names}

    adapters = types.ModuleType("slime.agent.adapters")
    adapters.__path__ = []
    adapters.AnthropicAdapter = type("AnthropicAdapter", (), {})
    adapters.OpenAIAdapter = type("OpenAIAdapter", (), {})
    adapters_common = types.ModuleType("slime.agent.adapters.common")
    adapters_common.flatten_content = lambda value: str(value or "")
    aiohttp_threaded = types.ModuleType("slime.agent.aiohttp_threaded")
    aiohttp_threaded.FilteredAccessLogger = type("FilteredAccessLogger", (), {})
    aiohttp_threaded.run_app_in_thread = lambda *args, **kwargs: None
    processing = types.ModuleType("slime.utils.processing_utils")
    processing.load_tokenizer = lambda *args, **kwargs: None
    types_mod = types.ModuleType("slime.utils.types")
    types_mod.Sample = type("Sample", (), {})

    try:
        sys.modules.update(
            {
                "slime.agent.adapters": adapters,
                "slime.agent.adapters.common": adapters_common,
                "slime.agent.aiohttp_threaded": aiohttp_threaded,
                "slime.utils.processing_utils": processing,
                "slime.utils.types": types_mod,
            }
        )
        sys.modules.pop("examples.coding_agent_rl.generate", None)
        sys.modules.pop("examples.coding_agent_rl.swe", None)
        generate = importlib.import_module("examples.coding_agent_rl.generate")
        swe = importlib.import_module("examples.coding_agent_rl.swe")
        yield generate, swe
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_agent_and_eval_both_use_selected_sandbox_factory():
    with _coding_modules() as (generate, swe):
        created = []

        def factory(image, *, metadata=None):
            created.append((image, dict(metadata or {})))
            return FakeSandbox(image)

        class Harness:
            async def prepare_cli(self, sb):
                return None

        generate.create_sandbox = factory
        generate.HARNESS_CLS = Harness
        swe.create_sandbox = factory

        async def run_case():
            async with generate.boot_agent_sandbox("image", "instance-1", "session-1"):
                pass
            result = await swe._grade_scaleswe(
                {
                    "instance_id": "instance-1",
                    "image": "image",
                    "workdir": "/workspace/repo",
                    "grading": {"eval_cmd": "true"},
                },
                "",
                30,
            )
            assert result.reward == 1.0

        asyncio.run(run_case())
        assert [metadata["role"] for _, metadata in created] == ["agent", "eval"]
        assert created[0][1]["session_id"] == "session-1"
        assert all(metadata["instance_id"] == "instance-1" for _, metadata in created)


def test_ambiguous_create_does_not_enter_outer_boot_retry():
    with _coding_modules() as (generate, _swe):
        calls = 0

        class AmbiguousSandbox(FakeSandbox):
            async def __aenter__(self):
                raise generate.SandboxLeaseError("unknown create outcome")

        def factory(image, *, metadata=None):
            nonlocal calls
            calls += 1
            return AmbiguousSandbox(image)

        class Harness:
            async def prepare_cli(self, sb):
                raise AssertionError("CLI preparation must not run")

        generate.create_sandbox = factory
        generate.HARNESS_CLS = Harness
        generate.CONFIG = generate.SweConfig(
            eval_protocol="scaleswe",
            train_protocol="scaleswe",
            adapter_bind_host="0.0.0.0",
            adapter_port=18001,
            theta_base_url="https://theta.example/api/anthropic",
            theta_service_name="test-service",
            theta_api_key="test-key",
            fork_merge_threshold=None,
            agent_time_budget_sec=30,
            eval_timeout_sec=30,
            rollout_guard_sec=60,
            boot_concurrency=1,
            boot_retries=3,
        )

        async def run_case():
            with pytest.raises(generate.SandboxLeaseError):
                async with generate.boot_agent_sandbox("image", "instance-1", "session-1"):
                    pass

        asyncio.run(run_case())
        assert calls == 1


def test_unreleased_sandbox_does_not_enter_outer_boot_retry():
    with _coding_modules() as (generate, _swe):
        calls = 0

        class Unreleased(FakeSandbox):
            async def __aenter__(self):
                raise generate.SandboxLeaseError("known sandbox was not destroyed")

        def factory(image, *, metadata=None):
            nonlocal calls
            calls += 1
            return Unreleased(image)

        generate.create_sandbox = factory
        generate.CONFIG = generate.SweConfig(
            eval_protocol="scaleswe",
            train_protocol="scaleswe",
            adapter_bind_host="0.0.0.0",
            adapter_port=18001,
            theta_base_url="https://theta.example/api/anthropic",
            theta_service_name="test-service",
            theta_api_key="test-key",
            fork_merge_threshold=None,
            agent_time_budget_sec=30,
            eval_timeout_sec=30,
            rollout_guard_sec=60,
            boot_concurrency=1,
            boot_retries=3,
        )

        async def run_case():
            with pytest.raises(generate.SandboxLeaseError):
                async with generate.boot_agent_sandbox("image", "instance-1", "session-1"):
                    pass

        asyncio.run(run_case())
        assert calls == 1


def test_multinode_launcher_keeps_arca_api_key_out_of_ray_cli_arguments():
    launcher = (REPO_ROOT / "examples/coding_agent_rl/run_qwen36_35b_a3b_swe_8nodes.sh").read_text()

    assert "SLIME_AGENT_ARCA_API_KEY" in launcher
    assert 'chmod 600 "${RUNTIME_ENV_FILE}"' in launcher
    assert '--runtime-env="${RUNTIME_ENV_FILE}"' in launcher
    assert '--runtime-env-json="${RUNTIME_ENV_JSON}"' not in launcher


def test_arca_8gpu_launcher_enables_core_attention_activation_offloading():
    launcher = (REPO_ROOT / "examples/coding_agent_rl/run_qwen36_27b_swe_8gpu_arca.sh").read_text()

    assert "--fine-grained-activation-offloading" in launcher
    assert "--offload-modules core_attn" in launcher
    assert "export NVTE_CPU_OFFLOAD_V1=1" in launcher
    assert '"NVTE_CPU_OFFLOAD_V1",' in launcher


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
