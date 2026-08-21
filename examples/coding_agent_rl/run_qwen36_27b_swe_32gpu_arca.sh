#!/usr/bin/env bash
# SWE coding-agent RL with Qwen3.6-27B + ARCA sandbox, 32 GPUs.
# Model arch sourced from scripts/models/qwen3.5-27B.sh (qwen3_5 hybrid 64L).
#
# PLACEMENT_MODE=colocate uses all 32 GPUs for colocated training and rollout.
# PLACEMENT_MODE=non_colocate splits them into 16 training and 16 rollout GPUs.

set -euo pipefail
export PYTHONUNBUFFERED=1

# Run the regular rollout + train loop by default. Set ROLLOUT_ONLY=1 to use
# the rollout-only A baseline for inference profiling.
ROLLOUT_ONLY="${ROLLOUT_ONLY:-0}"
if [[ "${ROLLOUT_ONLY}" != "0" && "${ROLLOUT_ONLY}" != "1" ]]; then
   echo "ERROR: ROLLOUT_ONLY must be 0 or 1" >&2
   exit 2
fi
PLACEMENT_MODE="${PLACEMENT_MODE:-colocate}"
if [[ "${PLACEMENT_MODE}" != "colocate" && "${PLACEMENT_MODE}" != "non_colocate" ]]; then
   echo "ERROR: PLACEMENT_MODE must be colocate or non_colocate" >&2
   exit 2
fi
NUM_ROLLOUT_DEFAULT=300
if [[ "${ROLLOUT_ONLY}" == "1" ]]; then
   NUM_ROLLOUT_DEFAULT=3
fi
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
if [[ "${ROLLOUT_ONLY}" != "1" && ! "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
   echo "ERROR: SAVE_INTERVAL must be a positive integer" >&2
   exit 2
fi

SLIME_DIR="${SLIME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
SGLANG_PATH="${SGLANG_PATH:-}"
export MEGATRON_PATH
export SGLANG_PATH

# ============ ARCA sandbox pre-flight ============
export SLIME_AGENT_SANDBOX_BACKEND=arca
export SLIME_AGENT_ARCA_APP_NAME="${SLIME_AGENT_ARCA_APP_NAME:-arcaslimeagentrl}"
export SLIME_AGENT_ARCA_BASE_URL="${SLIME_AGENT_ARCA_BASE_URL:-http://arca-sandbox.global.alipay.com:8080}"
export SLIME_AGENT_ARCA_API_KEY="665934ee53b64b0f83c1e8115f6e0dd5"
export SLIME_AGENT_ARCA_TEMPLATE_ID="${SLIME_AGENT_ARCA_TEMPLATE_ID:-ARCA-TEMPLATE-000000004480168f}"
export SLIME_AGENT_ARCA_IMAGE_REGISTRY="${SLIME_AGENT_ARCA_IMAGE_REGISTRY:-asr.antgroup-inc.cn/arcaslimeagentrl/sweb.instance}"
export SLIME_AGENT_ARCA_IMAGE_TAG_SUFFIX="${SLIME_AGENT_ARCA_IMAGE_TAG_SUFFIX:-claude-code-2.1.220-v1}"

# ============ Cluster and model parallelism ============
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
TP_SIZE="${TP_SIZE:-4}"
CP_SIZE="${CP_SIZE:-2}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-4}"
ROLLOUT_MEM_UTILIZATION="${ROLLOUT_MEM_UTILIZATION:-0.80}"

if [[ "${PLACEMENT_MODE}" == "non_colocate" ]]; then
   ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-2}"
   PP_SIZE="${PP_SIZE:-2}"
   ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-16}"
   EXP_TAG_DEFAULT="arca-sandbox-32gpu-non-colocate-27b"
else
   ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-4}"
   PP_SIZE="${PP_SIZE:-4}"
   EXP_TAG_DEFAULT="arca-sandbox-32gpu-27b"
fi

# ============ Model spec (qwen3_5 hybrid 27B) ============
source "${SLIME_DIR}/scripts/models/qwen3.5-27B.sh"

EXP_TAG="${EXP_TAG:-${EXP_TAG_DEFAULT}}"
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
if [[ "${PLACEMENT_MODE}" == "non_colocate" ]]; then
   echo "RUN_ROOT=${RUN_ROOT} | backend=${SLIME_AGENT_SANDBOX_BACKEND} | placement=${PLACEMENT_MODE} | train=${ACTOR_NUM_NODES}x${ACTOR_NUM_GPUS_PER_NODE} | rollout=${ROLLOUT_NUM_GPUS} | parallelism=TP${TP_SIZE}xPP${PP_SIZE}xCP${CP_SIZE}"
else
   echo "RUN_ROOT=${RUN_ROOT} | backend=${SLIME_AGENT_SANDBOX_BACKEND} | placement=${PLACEMENT_MODE} | actors=${ACTOR_NUM_NODES}x${ACTOR_NUM_GPUS_PER_NODE} | parallelism=TP${TP_SIZE}xPP${PP_SIZE}xCP${CP_SIZE}"
fi

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT:-/mnt/amedelastic-m/common/ckpt/muchen/Qwen3.6-27B}"
   --ref-load "${REF_MODEL_PATH:-/mnt/amedelastic-m/common/ckpt/muchen/Qwen3.6-27B-tdst}"
)
if [[ "${ROLLOUT_ONLY}" != "1" ]]; then
   SAVE_PATH="${SAVE_PATH:-${RUN_ROOT}/checkpoints}"
   install -d -m 700 "${SAVE_PATH}"
   CKPT_ARGS+=(
      --save "${SAVE_PATH}"
      --save-interval "${SAVE_INTERVAL}"
   )
   echo "Checkpoint path: ${SAVE_PATH} | interval=${SAVE_INTERVAL} steps"
fi

ROLLOUT_ARGS=(
   --custom-generate-function-path examples.coding_agent_rl.generate.generate
   --custom-reward-post-process-path slime.rollout._fanout_test_helpers.grpo_normalize_by_group_index
   --prompt-data "${PROMPT_DATA:-/personal/muchen/code_agent_data/swe_verified_v5.jsonl}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --num-rollout "${NUM_ROLLOUT:-${NUM_ROLLOUT_DEFAULT}}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-4}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-4}"
   --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std_with_fallback
   --rollout-max-context-len "${MAX_CONTEXT_LEN:-65536}"
   --rollout-max-response-len "${MAX_GEN_LEN:-16384}"
   --rollout-temperature 1.0
   --rollout-stop-token-ids 248046 248044
   --num-steps-per-rollout 1
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PP_SIZE}"
   --context-parallel-size "${CP_SIZE}"
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
   --use-dynamic-batch-size
   --qkv-format thd
)

# Transformer Engine v2.10+ otherwise offloads weights as well as activations.

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
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   # Slime derives SGLang TP from rollout-num-gpus-per-engine.
   --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE}"
   --sglang-mem-fraction-static "${ROLLOUT_MEM_UTILIZATION}"
   --sglang-context-length "${MAX_CONTEXT_LEN:-65536}"
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-1024}"
   --sglang-page-size 256
   --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-128}"
   --router-policy cache_aware
   --router-balance-abs-threshold "${ROUTER_BALANCE_ABS_THRESHOLD:-10}"
   --router-balance-rel-threshold "${ROUTER_BALANCE_REL_THRESHOLD:-1.2}"
   # /generate is not idempotent once a worker has accepted its rid. Router
   # retries can replay the same rid while the first generation is still live.
   --router-disable-retries
   --sglang-disable-custom-all-reduce
   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --log-interval 10
   --log-memory-interval 10
   --log-device-memory-used
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --actor-num-nodes "${ACTOR_NUM_NODES}"
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
   --num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
)
if [[ "${PLACEMENT_MODE}" == "non_colocate" ]]; then
   MISC_ARGS+=(--rollout-num-gpus "${ROLLOUT_NUM_GPUS}")
else
   MISC_ARGS+=(--colocate)
   export SLIME_DESTROY_WORLD_PROCESS_GROUP="${SLIME_DESTROY_WORLD_PROCESS_GROUP:-0}"
   if [[ "${ROLLOUT_ONLY}" != "1" ]]; then
      MISC_ARGS+=(--no-tms-cpu-backup)
   fi
fi

# ============ Network ============
export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(hostname -I | awk '{print $1}')}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-6379}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export NCCL_DEBUG=WARN

# ============ SWE agent knobs ============
export SWE_TRAIN_PROTOCOL="${SWE_TRAIN_PROTOCOL:-swebench}"
export SWE_EVAL_PROTOCOL="${SWE_EVAL_PROTOCOL:-swebench}"
export THETA_API_KEY="UlRvc3YoQBg0lQjeDxwen7OTPSTUd9Xh"
export THETA_SERVICE_NAME="${THETA_SERVICE_NAME:-slime_qwen36_27b_32gpu_${STAMP}}"
export THETA_BASE_URL="${THETA_BASE_URL:-https://antchat.alipay.com/api/anthropic}"
export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18001}"

export SWE_AGENT_TIME_BUDGET_SEC="${SWE_AGENT_TIME_BUDGET_SEC:-1800}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-600}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-16}"

export SLIME_AGENT_TRAJECTORY_SAVE="${SLIME_AGENT_TRAJECTORY_SAVE:-all}"
export SLIME_AGENT_TRAJECTORY_DIR="${SLIME_AGENT_TRAJECTORY_DIR:-${RUN_ROOT}/trajectories}"
export SLIME_AGENT_TRAJECTORY_WRITE_CONCURRENCY="${SLIME_AGENT_TRAJECTORY_WRITE_CONCURRENCY:-4}"
install -d -m 700 "${SLIME_AGENT_TRAJECTORY_DIR}"

export SLIME_ADAPTER_MAX_TOOL_RESULT_CHARS="${SLIME_ADAPTER_MAX_TOOL_RESULT_CHARS:-10000}"
export CLAUDE_CODE_AUTO_COMPACT_WINDOW="${CLAUDE_CODE_AUTO_COMPACT_WINDOW:-100000}"
export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE="${CLAUDE_AUTOCOMPACT_PCT_OVERRIDE:-45}"
export CLAUDE_CODE_MAX_OUTPUT_TOKENS="${CLAUDE_CODE_MAX_OUTPUT_TOKENS:-32768}"
# ARCA preserves the image's base Conda PATH, while SWE-bench dependencies live
# in the per-instance testbed environment. Claude and every tool it spawns must
# resolve python/pip from that environment too.
if [[ -z "${SLIME_AGENT_CC_EXTRA_ENVS:-}" ]]; then
   export SLIME_AGENT_CC_EXTRA_ENVS='{"PATH":"/opt/miniconda3/envs/testbed/bin:/opt/python/bin:/opt/miniconda3/bin:/usr/local/bin:/usr/bin:/bin","CONDA_PREFIX":"/opt/miniconda3/envs/testbed","CONDA_DEFAULT_ENV":"testbed"}'
fi
SETTINGS_JSON='{"permissions":{"defaultMode":"bypassPermissions"},"autoCompactEnabled":true}'
export SLIME_AGENT_CC_EXTRA_ARGS="--settings '${SETTINGS_JSON}' --disable-slash-commands --disallowedTools WebFetch WebSearch Task Agent EnterWorktree ExitWorktree"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

export no_proxy="127.0.0.1,${MASTER_ADDR}"
export NO_PROXY="${no_proxy}"

DEBUG_ARGS=()
if [[ "${ROLLOUT_ONLY}" == "1" ]]; then
   install -d -m 700 "${RUN_ROOT}/rollout_dumps"
   DEBUG_ARGS=(
      --debug-rollout-only
      --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
   )
   echo "ROLLOUT_ONLY=1: debug-rollout-only A baseline enabled"
   echo "  rollout_dumps: ${RUN_ROOT}/rollout_dumps/"
   echo "  num_rollout: ${NUM_ROLLOUT:-${NUM_ROLLOUT_DEFAULT}}"
fi

ip=$(ps aux | grep dashboard | grep -oP '(?<=--node-ip-address=)[0-9\.]+' | head -1)
port=$(ps aux | grep dashboard | grep -oP '(?<=--port=)\d+' | head -1)
export HEAD_NODE_ADDRESS="${ip}"
export DASHBOARD_PORT="${port}"
export RAY_ADDRESS="http://${HEAD_NODE_ADDRESS}:${DASHBOARD_PORT}"
echo "Detected Ray Head IP: ${HEAD_NODE_ADDRESS}, Port: ${DASHBOARD_PORT}"

# ============ Runtime env propagated to Ray workers ============
export SLIME_DIR
RUNTIME_ENV_FILE="$(mktemp "${TMPDIR:-/tmp}/slime-runtime-env.yaml.XXXXXX")"
chmod 600 "${RUNTIME_ENV_FILE}"
export RUNTIME_ENV_FILE
trap 'rm -f "${RUNTIME_ENV_FILE}"' EXIT
python3 - <<'PY'
import json
import os

keys = (
    "no_proxy", "NO_PROXY",
    "ADAPTER_BIND_HOST", "ADAPTER_PORT",
    "THETA_API_KEY", "THETA_SERVICE_NAME", "THETA_BASE_URL",
    "POD_IP", "SYSTEM_API_JWT_TAG", "DV_ENDPOINT_ADDR",
    "SLIME_AGENT_CC_EXTRA_ARGS", "SLIME_AGENT_CC_EXTRA_ENVS",
    "SLIME_AGENT_TRAJECTORY_SAVE", "SLIME_AGENT_TRAJECTORY_DIR",
    "SLIME_AGENT_TRAJECTORY_WRITE_CONCURRENCY",
    "SLIME_ADAPTER_MAX_TOOL_RESULT_CHARS",
    "CLAUDE_CODE_AUTO_COMPACT_WINDOW",
    "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    "SWE_CC_PROMPT", "SWE_TRAIN_PROTOCOL", "SWE_EVAL_PROTOCOL",
    "SWE_AGENT_TIME_BUDGET_SEC", "SWE_EVAL_TIMEOUT_SEC", "SWE_ROLLOUT_GUARD_SEC",
    "SWE_BOOT_CONCURRENCY",
    "SLIME_AGENT_SANDBOX_BACKEND",
    "SLIME_AGENT_ARCA_APP_NAME", "SLIME_AGENT_ARCA_BASE_URL", "SLIME_AGENT_ARCA_API_KEY",
    "SLIME_AGENT_ARCA_TEMPLATE_ID",
    "SLIME_AGENT_ARCA_IMAGE_REGISTRY", "SLIME_AGENT_ARCA_IMAGE_TAG_SUFFIX",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN",
    "NCCL_DEBUG",
    # A-approach: trainers keep the WORLD process group alive across the
    # colocated train<->rollout flips; propagate the override to every Ray worker.
    "SLIME_DESTROY_WORLD_PROCESS_GROUP",
)
env = {key: os.environ[key] for key in keys if key in os.environ}
env["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
env["MASTER_PORT"] = os.environ.get("MASTER_PORT", "")
env["GLOO_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["TP_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["NCCL_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]
pythonpath_entries = [
    os.environ.get("SGLANG_PATH"),
    os.environ["MEGATRON_PATH"],
    os.environ["SLIME_DIR"],
    f"{os.environ['SLIME_DIR']}/third_party",
]
env["PYTHONPATH"] = ":".join(entry for entry in pythonpath_entries if entry)
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
with open(os.environ["RUNTIME_ENV_FILE"], "w", encoding="utf-8") as fp:
    json.dump({"env_vars": env}, fp)
PY

ray job submit --address="${RAY_ADDRESS}" \
   --runtime-env="${RUNTIME_ENV_FILE}" \
   -- python3 -u "${SLIME_DIR}/train.py" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${ALGO_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${DEBUG_ARGS[@]}" \
   2>&1 | tee "${LOG_FILE}"

echo "RUN_ROOT=${RUN_ROOT}"
