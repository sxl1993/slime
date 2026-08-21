#!/usr/bin/env bash
# SWE coding-agent RL with Qwen3.6-27B + ARCA sandbox, single-node 8 GPUs.
# Model arch sourced from scripts/models/qwen3.5-27B.sh (qwen3_5 hybrid 64L).

set -euo pipefail

SLIME_DIR="${SLIME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"

# ============ ARCA sandbox pre-flight ============
export SLIME_AGENT_SANDBOX_BACKEND=arca
export SLIME_AGENT_ARCA_APP_NAME="${SLIME_AGENT_ARCA_APP_NAME:-arcaslimeagentrl}"
export SLIME_AGENT_ARCA_BASE_URL="${SLIME_AGENT_ARCA_BASE_URL:-http://arca-sandbox.global.alipay.com:8080}"
: "${SLIME_AGENT_ARCA_API_KEY:?Set SLIME_AGENT_ARCA_API_KEY for the ARCA backend}"
export SLIME_AGENT_ARCA_API_KEY
export SLIME_AGENT_ARCA_TEMPLATE_ID="${SLIME_AGENT_ARCA_TEMPLATE_ID:-ARCA-TEMPLATE-000000004480168f}"
export SLIME_AGENT_ARCA_IMAGE_REGISTRY="${SLIME_AGENT_ARCA_IMAGE_REGISTRY:-asr.antgroup-inc.cn/arcaslimeagentrl/sweb.instance}"
export SLIME_AGENT_ARCA_IMAGE_TAG_SUFFIX="${SLIME_AGENT_ARCA_IMAGE_TAG_SUFFIX:-claude-code-2.1.220-v1}"

python3 -c 'import arca' || {
  echo "ERROR: arca-sandbox SDK is not importable" >&2
  exit 1
}

EXP_TAG="${EXP_TAG:-arca-sandbox-8gpu-27b}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-${STAMP}_$$}"
RUN_ROOT="${RUN_ROOT:-${SLIME_DIR}/runs/${EXP_TAG}_${RUN_ID}}"
mkdir -p "$(dirname -- "${RUN_ROOT}")"
if ! mkdir -m 700 "${RUN_ROOT}"; then
   echo "ERROR: RUN_ROOT already exists or cannot be created: ${RUN_ROOT}" >&2
   exit 2
fi
LOG_FILE="${RUN_ROOT}/run.log"
echo "Training log: ${LOG_FILE}"
echo "RUN_ROOT=${RUN_ROOT} | backend=${SLIME_AGENT_SANDBOX_BACKEND}"

# ============ Cleanup ============
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
sleep 3
pkill -9 ray || true

# ============ Model spec (qwen3_5 hybrid 27B from scripts/models/qwen3.5-27B.sh) ============
source "${SLIME_DIR}/scripts/models/qwen3.5-27B.sh"

# Keep enough headroom for torch_memory_saver.pause() after the actor update.
# 32768 tokens/GPU left only ~7 GB free and caused the native offload path to die.

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT:-/mnt/amedelastic-m/common/ckpt/muchen/Qwen3.6-27B}"
   --ref-load "${REF_MODEL_PATH:-/mnt/amedelastic-m/common/ckpt/muchen/Qwen3.6-27B-tdst}"
)

ROLLOUT_ARGS=(
   --custom-generate-function-path examples.coding_agent_rl.generate.generate
   --prompt-data "${PROMPT_DATA:-/personal/muchen/code_agent_data/swe_verified_v5.jsonl}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --num-rollout "${NUM_ROLLOUT:-1}"
   --rollout-batch-size 1
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-1}"
   --rollout-max-context-len "${MAX_CONTEXT_LEN:-65536}"
   --rollout-max-response-len "${MAX_GEN_LEN:-8192}"
   --rollout-temperature 1.0
   --rollout-stop-token-ids 248046 248044
   --num-steps-per-rollout 1
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE:-4}"
   --sequence-parallel
   --context-parallel-size "${CP_SIZE:-2}"
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --fine-grained-activation-offloading
   --offload-modules core_attn
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
   --use-dynamic-batch-size
   --qkv-format thd
)

# Transformer Engine v2.10+ otherwise offloads weights as well as activations.
export NVTE_CPU_OFFLOAD_V1=1

ALGO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   # Each rollout performs one optimizer step. StatelessAdam implements the
   # same zero-moment update without allocating persistent exp_avg/exp_avg_sq,
   # leaving activation headroom after an offload/wake-up cycle.
   --use-stateless-adam
   --no-save-optim
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus 8
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static "${ROLLOUT_MEM_UTILIZATION:-0.80}"
   --sglang-context-length "${MAX_CONTEXT_LEN:-65536}"
   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --colocate
   --no-tms-cpu-backup
)

# ============ Network ============
export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(hostname -I | awk '{print $1}')}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export SLIME_DESTROY_WORLD_PROCESS_GROUP="0"

# ============ SWE agent knobs ============
export SWE_TRAIN_PROTOCOL="${SWE_TRAIN_PROTOCOL:-swebench}"
export SWE_EVAL_PROTOCOL="${SWE_EVAL_PROTOCOL:-swebench}"

# The single-node RolloutManager owns one Adapter and registers it to Theta in
# code. Sandbox Claude Code always calls antchat at THETA_BASE_URL, addressing
# the model as ckpt:<THETA_SERVICE_NAME>; Theta routes it to the registered
# Adapter. Session routing rides on Claude Code's metadata.user_id because the
# gateway bearer token is no longer a session identifier.
: "${THETA_API_KEY:?Set THETA_API_KEY for the Theta gateway}"
export THETA_API_KEY
export THETA_SERVICE_NAME="${THETA_SERVICE_NAME:-slime_qwen36_27b_${STAMP}}"
export THETA_BASE_URL="${THETA_BASE_URL:-https://antchat.alipay.com/api/anthropic}"
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18001}"

# autoCompactWindow (20k) < context length (65536): compact early so the CLI
# has more room before the 65536 training-side cap.
SETTINGS_JSON='{"permissions":{"defaultMode":"bypassPermissions"},"autoCompactEnabled":true,"autoCompactWindow":20000}'
export SLIME_AGENT_CC_EXTRA_ARGS="--settings '${SETTINGS_JSON}' --disable-slash-commands --disallowedTools WebFetch WebSearch Task Agent EnterWorktree ExitWorktree"

# Allow SGLang to extend context beyond model's max_position_embeddings.
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

export no_proxy="127.0.0.1,${MASTER_ADDR}"
export NO_PROXY="${no_proxy}"

cd "${SLIME_DIR}"

# ============ Ray (single node) ============
# --num-cpus omitted: let Ray autodetect machine cores. Placement-group bundles
# reserve 1 CPU per GPU, and CPU-only actors (RolloutManager/lock) need room.
NUM_GPUS=8
ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${NUM_GPUS}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "Waiting for Ray..."
sleep 10
ray status

# ============ Runtime env ============
export SLIME_DIR
RUNTIME_ENV_FILE="$(mktemp "${TMPDIR:-/tmp}/slime-runtime-env.json.XXXXXX")"
chmod 600 "${RUNTIME_ENV_FILE}"
export RUNTIME_ENV_FILE
trap 'rm -f "${RUNTIME_ENV_FILE}"' EXIT
python3 - <<'PY'
import json
import os

keys = (
    "no_proxy", "NO_PROXY",
    "NVTE_CPU_OFFLOAD_V1",
    "ADAPTER_BIND_HOST", "ADAPTER_PORT",
    "THETA_API_KEY", "THETA_SERVICE_NAME", "THETA_BASE_URL",
    "POD_IP", "SYSTEM_API_JWT_TAG", "DV_ENDPOINT_ADDR",
    "SLIME_AGENT_CC_EXTRA_ARGS",
    "SLIME_AGENT_CC_EXTRA_ENVS",
    "SWE_CC_PROMPT",
    "SWE_TRAIN_PROTOCOL", "SWE_EVAL_PROTOCOL",
    "SLIME_AGENT_SANDBOX_BACKEND",
    "SLIME_AGENT_ARCA_APP_NAME", "SLIME_AGENT_ARCA_BASE_URL", "SLIME_AGENT_ARCA_API_KEY",
    "SLIME_AGENT_ARCA_TEMPLATE_ID",
    "SLIME_AGENT_ARCA_IMAGE_REGISTRY", "SLIME_AGENT_ARCA_IMAGE_TAG_SUFFIX",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN",
    "SLIME_DESTROY_WORLD_PROCESS_GROUP",
)
env = {key: os.environ[key] for key in keys if key in os.environ}
env["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
env["MASTER_PORT"] = os.environ.get("MASTER_PORT", "")
env["GLOO_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["TP_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["NCCL_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]
env["PYTHONPATH"] = f"/root/Megatron-LM/:{os.environ['SLIME_DIR']}:{os.environ['SLIME_DIR']}/third_party"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
with open(os.environ["RUNTIME_ENV_FILE"], "w", encoding="utf-8") as fp:
    json.dump({"env_vars": env}, fp)
PY

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env="${RUNTIME_ENV_FILE}" \
   -- python3 -u train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${ALGO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   2>&1 | tee "${LOG_FILE}"

echo "RUN_ROOT=${RUN_ROOT}"
