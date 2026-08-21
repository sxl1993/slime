"""Lightweight routing tests for coding-agent ARCA integration.

Heavy adapter/model dependencies are stubbed because these tests exercise only
the sandbox boot/eval orchestration boundary.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
import subprocess
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


def test_lifecycle_rate_limit_enters_outer_boot_retry(monkeypatch):
    with _coding_modules() as (generate, _swe):
        calls = []
        sleeps = []
        rate_limit_error = generate.SandboxCreateRateLimitError

        class RateLimitedSandbox(FakeSandbox):
            async def __aenter__(self):
                raise rate_limit_error(retry_after=5)

        def factory(image, *, metadata=None):
            calls.append(dict(metadata or {}))
            return RateLimitedSandbox(image) if len(calls) == 1 else FakeSandbox(image)

        class Harness:
            async def prepare_cli(self, sb):
                return None

        async def record_sleep(delay):
            sleeps.append(delay)

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
            boot_retries=2,
        )
        monkeypatch.setattr(generate.asyncio, "sleep", record_sleep)
        monkeypatch.setattr(generate.random, "random", lambda: 0.25)

        async def run_case():
            async with generate.boot_agent_sandbox("image", "instance-1", "session-1"):
                pass

        asyncio.run(run_case())
        assert [metadata["attempt"] for metadata in calls] == ["1", "2"]
        assert sleeps == [5.25]


def test_lifecycle_rate_limit_does_not_sleep_after_final_attempt(monkeypatch):
    with _coding_modules() as (generate, _swe):
        calls = []
        sleeps = []
        rate_limit_error = generate.SandboxCreateRateLimitError

        class RateLimitedSandbox(FakeSandbox):
            async def __aenter__(self):
                raise rate_limit_error(retry_after=5)

        def factory(image, *, metadata=None):
            calls.append(dict(metadata or {}))
            return RateLimitedSandbox(image)

        async def record_sleep(delay):
            sleeps.append(delay)

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
            boot_retries=2,
        )
        monkeypatch.setattr(generate.asyncio, "sleep", record_sleep)
        monkeypatch.setattr(generate.random, "random", lambda: 0.25)

        async def run_case():
            with pytest.raises(rate_limit_error):
                async with generate.boot_agent_sandbox("image", "instance-1", "session-1"):
                    pass

        asyncio.run(run_case())
        assert [metadata["attempt"] for metadata in calls] == ["1", "2"]
        assert sleeps == [5.25]


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


def test_arca_32gpu_launcher_configures_claude_context_budget():
    launcher = (REPO_ROOT / "examples/coding_agent_rl/run_qwen36_27b_swe_32gpu_arca.sh").read_text()

    assert '--rollout-max-response-len "${MAX_GEN_LEN:-16384}"' in launcher
    assert 'export CLAUDE_CODE_AUTO_COMPACT_WINDOW="${CLAUDE_CODE_AUTO_COMPACT_WINDOW:-100000}"' in launcher
    assert 'export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE="${CLAUDE_AUTOCOMPACT_PCT_OVERRIDE:-45}"' in launcher
    assert 'export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-32768}"' in launcher
    assert 'export SLIME_ADAPTER_MAX_TOOL_RESULT_CHARS="${SLIME_ADAPTER_MAX_TOOL_RESULT_CHARS:-10000}"' in launcher
    assert '"CLAUDE_CODE_AUTO_COMPACT_WINDOW",' in launcher
    assert '"CLAUDE_AUTOCOMPACT_PCT_OVERRIDE",' in launcher
    assert '"CLAUDE_CODE_MAX_OUTPUT_TOKENS",' in launcher
    assert '"SLIME_ADAPTER_MAX_TOOL_RESULT_CHARS",' in launcher
    assert '"autoCompactWindow":20000' not in launcher


def test_arca_32gpu_launcher_disables_generate_replay_and_uses_testbed_python():
    launcher = (REPO_ROOT / "examples/coding_agent_rl/run_qwen36_27b_swe_32gpu_arca.sh").read_text()

    assert "--router-disable-retries" in launcher
    assert "/opt/miniconda3/envs/testbed/bin" in launcher
    assert "export SLIME_AGENT_CC_EXTRA_ENVS=" in launcher
    assert '"SLIME_AGENT_CC_EXTRA_ENVS",' in launcher


def test_arca_32gpu_launcher_uses_32gpu_names():
    launcher = (REPO_ROOT / "examples/coding_agent_rl/run_qwen36_27b_swe_32gpu_arca.sh").read_text()

    assert 'EXP_TAG_DEFAULT="arca-sandbox-32gpu-27b"' in launcher
    assert 'EXP_TAG_DEFAULT="arca-sandbox-32gpu-non-colocate-27b"' in launcher
    assert 'THETA_SERVICE_NAME="${THETA_SERVICE_NAME:-slime_qwen36_27b_32gpu_${STAMP}}"' in launcher


def test_arca_32gpu_launcher_configures_trajectory_directory():
    launcher = (REPO_ROOT / "examples/coding_agent_rl/run_qwen36_27b_swe_32gpu_arca.sh").read_text()

    assert 'export SLIME_AGENT_TRAJECTORY_SAVE="${SLIME_AGENT_TRAJECTORY_SAVE:-all}"' in launcher
    assert 'export SLIME_AGENT_TRAJECTORY_DIR="${SLIME_AGENT_TRAJECTORY_DIR:-${RUN_ROOT}/trajectories}"' in launcher
    assert "export SLIME_AGENT_TRAJECTORY_WRITE_CONCURRENCY=" in launcher
    assert 'install -d -m 700 "${SLIME_AGENT_TRAJECTORY_DIR}"' in launcher
    assert '"SLIME_AGENT_TRAJECTORY_SAVE", "SLIME_AGENT_TRAJECTORY_DIR",' in launcher
    assert '"SLIME_AGENT_TRAJECTORY_WRITE_CONCURRENCY",' in launcher


def test_arca_32gpu_launcher_supports_explicit_placement_modes():
    launcher = (REPO_ROOT / "examples/coding_agent_rl/run_qwen36_27b_swe_32gpu_arca.sh").read_text()

    assert 'PLACEMENT_MODE="${PLACEMENT_MODE:-colocate}"' in launcher
    assert 'MISC_ARGS+=(--rollout-num-gpus "${ROLLOUT_NUM_GPUS}")' in launcher
    assert "MISC_ARGS+=(--colocate)" in launcher


def test_arca_32gpu_launcher_saves_training_checkpoints_every_100_steps():
    launcher = (REPO_ROOT / "examples/coding_agent_rl/run_qwen36_27b_swe_32gpu_arca.sh").read_text()

    assert 'SAVE_INTERVAL="${SAVE_INTERVAL:-100}"' in launcher
    assert 'SAVE_PATH="${SAVE_PATH:-${RUN_ROOT}/checkpoints}"' in launcher
    assert '--save "${SAVE_PATH}"' in launcher
    assert '--save-interval "${SAVE_INTERVAL}"' in launcher
    assert 'if [[ "${ROLLOUT_ONLY}" != "1" ]]; then' in launcher


def test_arca_32gpu_launcher_rejects_invalid_save_interval():
    launcher = REPO_ROOT / "examples/coding_agent_rl/run_qwen36_27b_swe_32gpu_arca.sh"
    env = os.environ.copy()
    env.update({"SAVE_INTERVAL": "0", "SLIME_DIR": str(REPO_ROOT)})

    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "SAVE_INTERVAL must be a positive integer" in result.stderr


def test_arca_32gpu_launcher_rejects_existing_run_root(tmp_path):
    launcher = REPO_ROOT / "examples/coding_agent_rl/run_qwen36_27b_swe_32gpu_arca.sh"
    env = os.environ.copy()
    env.update({"RUN_ROOT": str(tmp_path), "SLIME_DIR": str(REPO_ROOT)})

    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert "RUN_ROOT already exists or cannot be created" in result.stderr


def test_qwen36_27b_arca_launchers_allocate_fresh_run_roots():
    for name in (
        "run_qwen36_27b_swe_8gpu_arca.sh",
        "run_qwen36_27b_swe_32gpu_arca.sh",
    ):
        launcher = (REPO_ROOT / "examples/coding_agent_rl" / name).read_text()

        assert 'RUN_ID="${RUN_ID:-${STAMP}_$$}"' in launcher
        assert 'if ! mkdir -m 700 "${RUN_ROOT}"; then' in launcher
        assert 'mkdir -p "${RUN_ROOT}"' not in launcher


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
