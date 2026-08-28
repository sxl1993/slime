#!/usr/bin/env bash
# Slime-native offline critic pretraining for Qwen3.8-27B.

set -euo pipefail
export PYTHONUNBUFFERED=1

: "${CRITIC_DATA_DIR:?CRITIC_DATA_DIR must point to the prepared Orchard artifact}"
: "${CRITIC_SAVE_PATH:?CRITIC_SAVE_PATH must point to the critic checkpoint root}"

SLIME_DIR="${SLIME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
export MEGATRON_PATH

CRITIC_MODE="${CRITIC_MODE:-canary}"
case "${CRITIC_MODE}" in
   canary|full|eval) ;;
   *) echo "ERROR: CRITIC_MODE must be canary, full, or eval" >&2; exit 2 ;;
esac

HF_CHECKPOINT="${HF_CHECKPOINT:-/mnt/amedelastic-m/common/ckpt/muchen/Qwen3.8-27B}"
CRITIC_TRAIN_LIMIT="${CRITIC_TRAIN_LIMIT:-4096}"
CRITIC_SELECTION_JSON="${CRITIC_SELECTION_JSON:-${CRITIC_SAVE_PATH}/selection.json}"
if [[ "${CRITIC_MODE}" == "full" ]]; then
   unset CRITIC_TRAIN_LIMIT
fi
if [[ "${CRITIC_MODE}" == "eval" ]]; then
   : "${CRITIC_CKPT_STEP:?CRITIC_CKPT_STEP is required for eval mode}"
   if [[ ! "${CRITIC_CKPT_STEP}" =~ ^[1-9][0-9]*$ ]]; then
      echo "ERROR: CRITIC_CKPT_STEP must be a positive integer" >&2
      exit 2
   fi
fi

source "${SLIME_DIR}/scripts/models/qwen3.5-27B.sh"

MODEL_ARGS=(
   "${MODEL_ARGS[@]}"
   --hf-checkpoint "${HF_CHECKPOINT}"
   --load "${HF_CHECKPOINT}"
   --save "${CRITIC_SAVE_PATH}"
   --debug-train-only
   --global-batch-size 128
   --rollout-batch-size 128
   --num-rollout 0
   --critic-pretrain-data "${CRITIC_DATA_DIR}"
   --critic-pretrain-mode "${CRITIC_MODE}"
   --critic-pretrain-selection-json "${CRITIC_SELECTION_JSON}"
   --critic-pretrain-eval-batch-size 128
   --lr 5e-6
   --lr-warmup-fraction 0.01
   --lr-decay-style cosine
   --min-lr 5e-7
   --optimizer adam
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   --actor-num-nodes 8
   --actor-num-gpus-per-node 8
   --num-gpus-per-node 8
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 4
   --context-parallel-size 4
   --sequence-parallel
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 4
   --max-tokens-per-gpu 4096
   --use-dynamic-batch-size
   --qkv-format thd
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-softmax-in-fp32
   --attention-backend flash
   --accumulate-allreduce-grads-in-fp32
)
if [[ -n "${CRITIC_TRAIN_LIMIT:-}" ]]; then
   MODEL_ARGS+=(--critic-pretrain-train-limit "${CRITIC_TRAIN_LIMIT}")
fi
if [[ "${CRITIC_MODE}" == "eval" ]]; then
   MODEL_ARGS+=(--ckpt-step "${CRITIC_CKPT_STEP}")
fi

export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(hostname -I | awk '{print $1}')}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-6379}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export SLIME_DIR
export CUDA_DEVICE_MAX_CONNECTIONS=1

ip=$(ps aux | grep dashboard | grep -oP '(?<=--node-ip-address=)[0-9\.]+')
port=$(ps aux | grep dashboard | grep -oP '(?<=--dashboard-port=)[0-9]+')
export RAY_ADDRESS="http://${ip}:${port}"

RUNTIME_ENV_FILE="$(mktemp "${TMPDIR:-/tmp}/slime-critic-runtime.XXXXXX")"
chmod 600 "${RUNTIME_ENV_FILE}"
trap 'rm -f "${RUNTIME_ENV_FILE}"' EXIT
python3 - "${RUNTIME_ENV_FILE}" <<'PY'
import json
import os
import sys

path = sys.argv[1]
env = {
    key: os.environ[key]
    for key in ("MASTER_ADDR", "MASTER_PORT", "GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME", "SLIME_DIR")
    if key in os.environ
}
env["PYTHONPATH"] = ":".join(
    entry for entry in (os.environ.get("MEGATRON_PATH"), os.environ.get("SLIME_DIR")) if entry
)
json.dump({"env_vars": env}, open(path, "w", encoding="utf-8"))
PY

ray job submit --address="${RAY_ADDRESS}" \
   --runtime-env="${RUNTIME_ENV_FILE}" \
   -- python3 -u -m examples.coding_agent_rl.critic_pretrain.train \
   "${MODEL_ARGS[@]}"
