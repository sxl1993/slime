import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "examples/coding_agent_rl/qwen38_27b_swe_32gpu_arca_fully_async.yaml"
LAUNCHER = REPO_ROOT / "examples/coding_agent_rl/run_qwen38_27b_swe_32gpu_arca_fully_async.sh"


def test_qwen38_fully_async_sao_config_enables_critic_attention_freeze():
    config = CONFIG.read_text()

    assert "sao-critic-freeze-attention: true" in config
    assert "sao-critic-update-ratio: 2" in config
    assert "sao-critic-warmup-steps: 10" in config


def test_qwen38_fully_async_sao_launcher_enables_critic_attention_freeze():
    launcher = LAUNCHER.read_text()

    assert "--sao-critic-freeze-attention" in launcher
    assert "--sao-critic-update-ratio 2" in launcher
    assert "--sao-critic-warmup-steps 10" in launcher


def test_qwen38_fully_async_sao_launcher_has_valid_shell_syntax():
    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
