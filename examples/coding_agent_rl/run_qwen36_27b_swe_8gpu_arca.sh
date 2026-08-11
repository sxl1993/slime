#!/usr/bin/env bash
# SWE coding-agent RL with Qwen3.6-27B + ARCA sandbox, single-node 8 GPUs.
# Model arch sourced from scripts/models/qwen3.5-27B.sh (qwen3_5 hybrid 64L).

set -euo pipefail

SLIME_DIR="${SLIME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"

# ============ ARCA sandbox pre-flight ============
export SLIME_AGENT_SANDBOX_BACKEND=arca
export SLIME_ARCA_APP_NAME="${SLIME_ARCA_APP_NAME:-a3training}"
export SLIME_ARCA_BASE_URL="${SLIME_ARCA_BASE_URL:-http://arca-sandbox.global.alipay.com:8080}"
: "${SLIME_ARCA_API_KEY:?Set SLIME_ARCA_API_KEY for the ARCA backend}"
export SLIME_ARCA_API_KEY
export SLIME_AGENT_ARCA_TEMPLATE_ID="${SLIME_AGENT_ARCA_TEMPLATE_ID:-ARCA-TEMPLATE-000000004480168f}"
export SLIME_AGENT_ARCA_TTL_MINUTES="${SLIME_AGENT_ARCA_TTL_MINUTES:-40}"
export SLIME_AGENT_ARCA_CPU="${SLIME_AGENT_ARCA_CPU:-2}"
export SLIME_AGENT_ARCA_MEMORY="${SLIME_AGENT_ARCA_MEMORY:-4}"
export SLIME_AGENT_ARCA_DISK="${SLIME_AGENT_ARCA_DISK:-25}"
export SLIME_AGENT_ARCA_CREATE_TIMEOUT_SEC="${SLIME_AGENT_ARCA_CREATE_TIMEOUT_SEC:-150}"
export SLIME_AGENT_ARCA_READY_TIMEOUT_SEC="${SLIME_AGENT_ARCA_READY_TIMEOUT_SEC:-120}"
export SLIME_AGENT_ARCA_READY_POLL_INTERVAL_SEC="${SLIME_AGENT_ARCA_READY_POLL_INTERVAL_SEC:-2}"

python3 -c 'import arca' || {
  echo "ERROR: arca-sandbox SDK is not importable" >&2
  exit 1
}

# ============ Cleanup ============
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
sleep 3
pkill -9 ray || true

# ============ Model spec (qwen3_5 hybrid 27B from scripts/models/qwen3.5-27B.sh) ============
source "${SLIME_DIR}/scripts/models/qwen3.5-27B.sh"

# ============ context length ============
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-65536}"
MAX_GEN_LEN="${MAX_GEN_LEN:-8192}"
CP_SIZE="${CP_SIZE:-2}"
# Keep enough headroom for torch_memory_saver.pause() after the actor update.
# 32768 tokens/GPU left only ~7 GB free and caused the native offload path to die.

# ============ Paths ============
HF_CHECKPOINT="${HF_CHECKPOINT:-/mnt/amed-s1/common/ckpt/muchen/Qwen3.6-27B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/mnt/amed-s1/common/ckpt/muchen/Qwen3.6-27B-tdst/}"
PROMPT_DATA="${PROMPT_DATA:-/personal/muchen/code_agent_data/swe_django.jsonl}"

EXP_TAG="${EXP_TAG:-arca-sandbox-8gpu-27b}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${SLIME_DIR}/runs/${EXP_TAG}_${STAMP}}"
LOG_FILE="${RUN_ROOT}/run.log"
mkdir -p "${RUN_ROOT}"
echo "======================================================================"
echo "Training log: ${LOG_FILE}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "SLIME_AGENT_SANDBOX_BACKEND=${SLIME_AGENT_SANDBOX_BACKEND}"
echo "======================================================================"

# ============ Phase 0 observability ============
PHASE0_ONLY="${PHASE0_ONLY:-0}"
NUM_ROLLOUT="${NUM_ROLLOUT:-4}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
export SWE_ROLLOUT_METRICS_PATH="${SWE_ROLLOUT_METRICS_PATH:-}"
export SWE_ROLLOUT_RUN_ID="${SWE_ROLLOUT_RUN_ID:-unknown}"
export SWE_ROLLOUT_SEED="${SWE_ROLLOUT_SEED:-}"

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT}"
   --ref-load "${REF_MODEL_PATH}"
)

ROLLOUT_ARGS=(
   --custom-generate-function-path examples.coding_agent_rl.generate.generate
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size 4
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-context-len "${MAX_CONTEXT_LEN}"
   --rollout-max-response-len "${MAX_GEN_LEN}"
   --rollout-temperature 1.0
   --rollout-stop-token-ids 248046 248044
   --num-steps-per-rollout 1
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE:-4}"
   --sequence-parallel
   --context-parallel-size "${CP_SIZE}"
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-24576}"
   --use-dynamic-batch-size
   --qkv-format thd
)

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
   --sglang-mem-fraction-static "${ROLLOUT_MEM_UTILIZATION:-0.60}"
   --sglang-context-length "${MAX_CONTEXT_LEN}"
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
)
# --no-tms-cpu-backup cuts host memory during the colocated train<->rollout
# offload; it needs --offload-train, which --debug-rollout-only (PHASE0_ONLY)
# force-disables. Gate it to normal training so PHASE0_ONLY runs don't trip
# slime_validate_args ("--no-tms-cpu-backup requires --offload-train").
if [[ "${PHASE0_ONLY}" != "1" ]]; then
   MISC_ARGS+=(--no-tms-cpu-backup)
fi

# ============ Network ============
export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(hostname -I | awk '{print $1}')}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"

# ============ SWE agent knobs ============
export SWE_AGENT="${SWE_AGENT:-claude_code}"
export SWE_TRAIN_PROTOCOL="${SWE_TRAIN_PROTOCOL:-swebench}"
export SWE_EVAL_PROTOCOL="${SWE_EVAL_PROTOCOL:-swebench}"

# The Adapter starts lazily in the RolloutManager process on the first SWE sample.
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-${MASTER_ADDR}}"
export ADAPTER_PUBLIC_URL="${ADAPTER_PUBLIC_URL:-}"
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18001}"

export SWE_AGENT_TIME_BUDGET_SEC="${SWE_AGENT_TIME_BUDGET_SEC:-1500}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-600}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-8}"

# Adapter context compression
export SLIME_ADAPTER_SYSTEM_PROMPT="${SLIME_ADAPTER_SYSTEM_PROMPT:-1}"
export SLIME_ADAPTER_TOOL_WHITELIST="${SLIME_ADAPTER_TOOL_WHITELIST:-Bash,Read,Edit,Write}"
export SLIME_ADAPTER_MAX_TOOL_RESULT_CHARS="${SLIME_ADAPTER_MAX_TOOL_RESULT_CHARS:-10000}"

# autoCompactWindow (20k) < MAX_CONTEXT_LEN (65536): compact early so the CLI
# has more room before the 65536 training-side cap.
SETTINGS_JSON='{"permissions":{"defaultMode":"bypassPermissions"},"autoCompactEnabled":true,"autoCompactWindow":20000}'
export SLIME_AGENT_CC_EXTRA_ARGS="--settings '${SETTINGS_JSON}' --disable-slash-commands --disallowedTools WebFetch WebSearch Task Agent EnterWorktree ExitWorktree"

# Raise CLI session output cap to match rollout context length.
# Without this the CLI defaults to 32000 tokens and aborts multi-turn sessions.
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-${MAX_CONTEXT_LEN}}"

# Allow SGLang to extend context beyond model's max_position_embeddings.
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

export no_proxy="127.0.0.1,${MASTER_ADDR},${ADAPTER_PUBLIC_HOST}"
export NO_PROXY="${no_proxy}"

cd "${SLIME_DIR}"

# ============ Phase 0 debug-rollout-only mode ============
DEBUG_ARGS=()
if [[ "${PHASE0_ONLY}" == "1" ]]; then
   mkdir -p "${RUN_ROOT}/rollout_dumps"
   DEBUG_ARGS=(
      --debug-rollout-only
      --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
   )
   export SWE_ROLLOUT_METRICS_PATH="${SWE_ROLLOUT_METRICS_PATH:-${RUN_ROOT}/rollout_metrics.jsonl}"
   echo "PHASE0_ONLY=1: debug-rollout-only mode enabled"
   echo "  rollout_dumps: ${RUN_ROOT}/rollout_dumps/"
   echo "  metrics_path:  ${SWE_ROLLOUT_METRICS_PATH}"
   echo "  rollout_seed:  ${SWE_ROLLOUT_SEED:-<unset>} (prompt-selection reproducibility only; does not fix sampling)"
fi

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
    "SWE_AGENT",
    "ADAPTER_PUBLIC_HOST", "ADAPTER_PUBLIC_URL", "ADAPTER_BIND_HOST", "ADAPTER_PORT",
    "SWE_AGENT_TIME_BUDGET_SEC", "SWE_EVAL_TIMEOUT_SEC", "SWE_BOOT_CONCURRENCY",
    "SLIME_AGENT_CC_EXTRA_ARGS",
    "SLIME_AGENT_CC_EXTRA_ENVS",
    "SWE_CC_PROMPT",
    "SWE_TRAIN_PROTOCOL", "SWE_EVAL_PROTOCOL",
    "SLIME_AGENT_SANDBOX_BACKEND",
    "SLIME_ARCA_APP_NAME", "SLIME_ARCA_BASE_URL", "SLIME_ARCA_API_KEY",
    "SLIME_AGENT_ARCA_TEMPLATE_ID", "SLIME_AGENT_ARCA_IMAGE_MAP",
    "SLIME_AGENT_ARCA_TTL_MINUTES",
    "SLIME_AGENT_ARCA_CPU", "SLIME_AGENT_ARCA_MEMORY", "SLIME_AGENT_ARCA_DISK",
    "SLIME_AGENT_ARCA_CREATE_TIMEOUT_SEC", "SLIME_AGENT_ARCA_READY_TIMEOUT_SEC",
    "SLIME_AGENT_ARCA_READY_POLL_INTERVAL_SEC",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN",
    "SLIME_ADAPTER_SYSTEM_PROMPT",
    "SLIME_ADAPTER_TOOL_WHITELIST",
    "SLIME_ADAPTER_MAX_TOOL_RESULT_CHARS",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    "SWE_ROLLOUT_METRICS_PATH",
    "SWE_ROLLOUT_RUN_ID",
    "SWE_ROLLOUT_SEED",
    "SLIME_AGENT_TOOL_TIMING",
)
env = {key: os.environ[key] for key in keys if key in os.environ}
env["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
env["MASTER_PORT"] = os.environ.get("MASTER_PORT", "")
env["GLOO_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["TP_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["NCCL_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]
env["PYTHONPATH"] = f"/root/Megatron-LM/:{os.environ['SLIME_DIR']}:{os.environ['SLIME_DIR']}/third_party"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env["NCCL_NVLS_ENABLE"] = "0"
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
   "${DEBUG_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${ALGO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   2>&1 | tee "${LOG_FILE}"

echo "RUN_ROOT=${RUN_ROOT}"
