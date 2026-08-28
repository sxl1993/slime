from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "examples/coding_agent_rl/run_qwen38_27b_critic_pretrain.sh"


def test_critic_pretrain_launcher_contract():
    launcher = LAUNCHER.read_text()
    assert "critic_pretrain.train" in launcher
    assert "--debug-train-only" in launcher
    assert "--load" in launcher
    assert "--global-batch-size 128" in launcher
    assert "--lr 5e-6" in launcher
    assert "--lr-warmup-fraction 0.01" in launcher
    assert "--lr-decay-style cosine" in launcher
    assert "--min-lr 5e-7" in launcher
    assert "rollout-function-path" not in launcher
    assert "sglang" not in launcher.lower()


def test_critic_pretrain_launcher_has_valid_shell_syntax():
    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
