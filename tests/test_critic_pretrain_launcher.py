from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "examples/coding_agent_rl/run_qwen38_27b_critic_pretrain.sh"
LAUNCHER_64 = REPO_ROOT / "examples/coding_agent_rl/run_qwen38_27b_critic_pretrain_64gpu.sh"


def test_critic_pretrain_launcher_contract():
    launcher = LAUNCHER.read_text()
    assert "critic_pretrain.train" in launcher
    assert "--debug-train-only" in launcher
    assert "--load" in launcher
    assert "--global-batch-size 128" in launcher
    assert "--seq-length 98304" in launcher
    assert "--critic-pretrain-canary" in launcher
    assert "CRITIC_PRETRAIN_MODE=train" in launcher
    assert 'INITIAL_LOAD="${CRITIC_SAVE_PATH}"' in launcher
    assert "--lr 5e-6" in launcher
    assert "--lr-warmup-fraction 0.01" in launcher
    assert "--lr-decay-style cosine" in launcher
    assert "--min-lr 5e-7" in launcher
    assert '--save-interval "${CRITIC_EVAL_INTERVAL}"' in launcher
    assert 'CRITIC_LOAD_PATH="${CRITIC_LOAD_PATH:-${HF_CHECKPOINT}}"' in launcher
    assert 'HF_CHECKPOINT="${HF_CHECKPOINT:-/mnt/amedelastic-et117-aidc/common/ckpt/AQInfra/Qwen3.8-27B}"' in launcher
    assert 'ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"' in launcher
    assert 'ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"' in launcher
    assert 'TP_SIZE="${TP_SIZE:-4}"' in launcher
    assert 'PP_SIZE="${PP_SIZE:-1}"' in launcher
    assert 'CP_SIZE="${CP_SIZE:-2}"' in launcher
    assert 'RAY_ADDRESS="http://127.0.0.1:8265"' in launcher
    assert 'CRITIC_LOG_FILE="${CRITIC_LOG_FILE:-${CRITIC_SAVE_PATH}.log}"' in launcher
    assert 'mkdir -p "$(dirname -- "${CRITIC_LOG_FILE}")"' in launcher
    assert 'tee "${CRITIC_LOG_FILE}"' in launcher
    assert "ray stop --force || true" in launcher
    assert launcher.count("pkill -9 ray || true") == 2
    assert "ray start --head" in launcher
    assert "--dashboard-host=0.0.0.0" in launcher
    assert "--dashboard-port=8265" in launcher
    assert "sleep 10" in launcher
    assert "ray status" in launcher
    assert "START_RAY" not in launcher
    assert "/api/version" not in launcher
    assert "ps aux | grep dashboard" not in launcher
    assert "/mnt/amedelastic-m/" not in launcher
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


def test_critic_pretrain_64gpu_launcher_contract():
    launcher = LAUNCHER_64.read_text()
    assert "critic_pretrain.train" in launcher
    assert 'ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-8}"' in launcher
    assert 'ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"' in launcher
    assert 'TP_SIZE="${TP_SIZE:-4}"' in launcher
    assert 'PP_SIZE="${PP_SIZE:-1}"' in launcher
    assert 'CP_SIZE="${CP_SIZE:-2}"' in launcher
    assert 'CRITIC_LOAD_PATH="${CRITIC_LOAD_PATH:-${HF_CHECKPOINT}}"' in launcher
    assert 'CRITIC_EVAL_INTERVAL="${CRITIC_EVAL_INTERVAL:-50}"' in launcher
    assert "world_size=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))" in launcher
    assert "model_parallel_size=$((TP_SIZE * PP_SIZE * CP_SIZE))" in launcher
    assert "if (( world_size % model_parallel_size != 0 )); then" in launcher
    assert "Detected Ray Head IP" in launcher
    assert 'export RAY_ADDRESS="http://${HEAD_NODE_ADDRESS}:${DASHBOARD_PORT}"' in launcher
    assert 'ray job submit --address="${RAY_ADDRESS}"' in launcher
    assert 'env "PYTHONPATH=${SLIME_DIR}:${SLIME_DIR}/third_party:${MEGATRON_PATH}" \\' in launcher
    assert "CUDA_DEVICE_MAX_CONNECTIONS=1 \\" in launcher
    assert "python3 -u -m examples.coding_agent_rl.critic_pretrain.train" in launcher
    assert (
        'os.environ.get("SLIME_DIR"),\n        os.path.join(os.environ["SLIME_DIR"], "third_party"),\n        os.environ.get("MEGATRON_PATH")'
        in launcher
    )
    assert 'env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"' in launcher
    assert "CRITIC_LOCAL_LOG_FILE=" in launcher
    assert 'tee "${CRITIC_LOCAL_LOG_FILE}"' in launcher
    assert "ray start" not in launcher
    assert "ray stop" not in launcher
    assert "pkill -9 ray" not in launcher
    assert "rollout-function-path" not in launcher
    assert "sglang" not in launcher.lower()


def test_critic_pretrain_64gpu_launcher_has_valid_shell_syntax():
    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER_64)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
