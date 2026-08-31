#!/usr/bin/env bash
# SWE coding-agent RL with Qwen3.8-27B + ARCA sandbox, 64 GPUs.
# Fully-async mode keeps a background rollout pool warm across rollout
# boundaries and returns completed groups as they become available.
# This script uses 32 training GPUs and 32 rollout GPUs; train_async.py does
# not support colocated training and rollout.
#
# Model arch sourced from scripts/models/qwen3.5-27B.sh (qwen3_5 hybrid 64L).
# The fully-async collector requeues ABORTED groups from scratch; it does not
# provide partial-rollout session resume.

set -euo pipefail
export PYTHONUNBUFFERED=1

# Run the regular rollout + train loop by default. Set ROLLOUT_ONLY=1 to use
# the rollout-only A baseline for inference profiling.
ROLLOUT_ONLY="${ROLLOUT_ONLY:-0}"
if [[ "${ROLLOUT_ONLY}" != "0" && "${ROLLOUT_ONLY}" != "1" ]]; then
   echo "ERROR: ROLLOUT_ONLY must be 0 or 1" >&2
   exit 2
fi
NUM_ROLLOUT_DEFAULT=3
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
if [[ "${ROLLOUT_ONLY}" != "1" && ! "${SAVE_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
   echo "ERROR: SAVE_INTERVAL must be a positive integer" >&2
   exit 2
fi
UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-2}"
if [[ ! "${UPDATE_WEIGHTS_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
   echo "ERROR: UPDATE_WEIGHTS_INTERVAL must be a positive integer" >&2
   exit 2
fi
CRITIC_LOAD="${CRITIC_LOAD:-/mnt/amedelastic-m/common/ckpt/muchen/Qwen3.8-27B-Critic}"
# Set CRITIC_LOAD to an independent HuggingFace critic.

ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
SAO_BATCH_SIZE="${SAO_BATCH_SIZE:-8}"

N_SAMPLES_PER_PROMPT=1
if [[ ! "${SAO_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
   echo "ERROR: SAO_BATCH_SIZE must be a positive integer" >&2
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
export THETA_API_KEY="UlRvc3YoQBg0lQjeDxwen7OTPSTUd9Xh"
export SLIME_AGENT_ARCA_API_KEY="665934ee53b64b0f83c1e8115f6e0dd5"
export SLIME_AGENT_ARCA_TEMPLATE_ID="${SLIME_AGENT_ARCA_TEMPLATE_ID:-ARCA-TEMPLATE-000000004480168f}"
export SLIME_AGENT_ARCA_IMAGE_REGISTRY="${SLIME_AGENT_ARCA_IMAGE_REGISTRY:-asr.antgroup-inc.cn/arcaslimeagentrl/sweb.instance}"
export SLIME_AGENT_ARCA_IMAGE_TAG_SUFFIX="${SLIME_AGENT_ARCA_IMAGE_TAG_SUFFIX:-claude-code-2.1.220-v1}"

# ============ Cluster and model parallelism ==========
# Fixed 64-GPU non-colocated layout: 4 nodes x 8 actor GPUs + 32 rollout GPUs.
ACTOR_NUM_NODES=8
ACTOR_NUM_GPUS_PER_NODE=8
ROLLOUT_NUM_GPUS=16
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE=4
CP_SIZE="${CP_SIZE:-4}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-8}"
ROLLOUT_MEM_UTILIZATION="${ROLLOUT_MEM_UTILIZATION:-0.80}"
EXP_TAG_DEFAULT="arca-sandbox-64gpu-fully-async-27b"

# ============ Model spec (qwen3_5 hybrid 27B) =========
source "${SLIME_DIR}/scripts/models/qwen3.5-27B.sh"

TRAIN_PROMPT_DATA="${PROMPT_DATA:-/personal/muchen/code_agent_data/swe_verified_v5.jsonl}"

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
echo "RUN_ROOT=${RUN_ROOT} | backend=${SLIME_AGENT_SANDBOX_BACKEND} | mode=fully_async | train=${ACTOR_NUM_NODES}x${ACTOR_NUM_GPUS_PER_NODE} | rollout=${ROLLOUT_NUM_GPUS} | parallelism=TP${TP_SIZE}xPP${PP_SIZE}xCP${CP_SIZE} | update_weights_interval=${UPDATE_WEIGHTS_INTERVAL}"

PROFILE="${PROFILE:-0}"
if [[ "${PROFILE}" != "0" && "${PROFILE}" != "1" ]]; then
   echo "ERROR: PROFILE must be 0 or 1" >&2
   exit 2
fi
PROFILE_ARGS=()
if [[ "${PROFILE}" == "1" ]]; then
   PROFILE_STEP_START="${PROFILE_STEP_START:-3}"
   PROFILE_STEP_END="${PROFILE_STEP_END:-4}"
   if [[ ! "${PROFILE_STEP_START}" =~ ^[0-9]+$ || ! "${PROFILE_STEP_END}" =~ ^[1-9][0-9]*$ ]] || (( PROFILE_STEP_END <= PROFILE_STEP_START )); then
      echo "ERROR: PROFILE_STEP_END must be greater than PROFILE_STEP_START >= 0" >&2
      exit 2
   fi
   PROFILE_DIR="${PROFILE_DIR:-${RUN_ROOT}/profiles}"
   install -d -m 700 "${PROFILE_DIR}"
   read -r -a PROFILE_RANK_ARGS <<< "${PROFILE_RANKS:-0 8 16 24}"
   if (( ${#PROFILE_RANK_ARGS[@]} == 0 )); then
      echo "ERROR: PROFILE_RANKS must contain at least one training rank" >&2
      exit 2
   fi
   PROFILE_ARGS=(
      --use-pytorch-profiler
      --profile-step-start "${PROFILE_STEP_START}"
      --profile-step-end "${PROFILE_STEP_END}"
      --profile-ranks "${PROFILE_RANK_ARGS[@]}"
      --tensorboard-dir "${PROFILE_DIR}"
   )
   echo "PyTorch profiler: dir=${PROFILE_DIR} | steps=[${PROFILE_STEP_START},${PROFILE_STEP_END}) | ranks=${PROFILE_RANK_ARGS[*]}"
fi

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT:-/mnt/amedelastic-m/common/ckpt/muchen/Qwen3.8-27B}"
   --ref-load "${REF_MODEL_PATH:-/mnt/amedelastic-m/common/ckpt/muchen/Qwen3.8-27B_torch_dist}"
)
ROLLOUT_COUNT_ARGS=()
DEBUG_ARGS=()
if [[ "${ROLLOUT_ONLY}" == "1" ]]; then
   ROLLOUT_COUNT_ARGS=(--num-rollout "${NUM_ROLLOUT:-${NUM_ROLLOUT_DEFAULT}}")
   install -d -m 700 "${RUN_ROOT}/rollout_dumps"
   DEBUG_ARGS=(
      --debug-rollout-only
      --save-debug-rollout-data "${RUN_ROOT}/rollout_dumps/rollout_{rollout_id}.pt"
   )
   echo "ROLLOUT_ONLY=1: debug-rollout-only A baseline enabled"
   echo "  rollout_dumps: ${RUN_ROOT}/rollout_dumps/"
   echo "  num_rollout: ${NUM_ROLLOUT:-${NUM_ROLLOUT_DEFAULT}}"
else
   ROLLOUT_COUNT_ARGS=(--num-epoch "${NUM_EPOCH:-3}")
   SAVE_PATH="${SAVE_PATH:-${RUN_ROOT}/checkpoints}"
   install -d -m 700 "${SAVE_PATH}"
   CKPT_ARGS+=(
      --save "${SAVE_PATH}"
      --save-interval "${SAVE_INTERVAL}"
   )
   echo "Checkpoint path: ${SAVE_PATH} | interval=${SAVE_INTERVAL} steps"
fi

if [[ -n "${CRITIC_LOAD}" ]]; then
   if [[ "${ROLLOUT_ONLY}" == "1" ]]; then
      echo "ERROR: CRITIC_LOAD cannot be used with ROLLOUT_ONLY=1" >&2
      exit 2
   fi
   if [[ ! -d "${CRITIC_LOAD}" ]]; then
      echo "ERROR: CRITIC_LOAD must be an existing model directory: ${CRITIC_LOAD}" >&2
      exit 2
   fi
   ROLE_CONFIG_TEMPLATE="${SLIME_DIR}/examples/coding_agent_rl/qwen38_27b_sao_roles.yaml"
   ACTOR_LOAD_PATH="${REF_MODEL_PATH:-/mnt/amedelastic-m/common/ckpt/muchen/Qwen3.8-27B_torch_dist}"
   ACTOR_SAVE_PATH="${SAVE_PATH}/actor"
   CRITIC_SAVE_PATH="${SAVE_PATH}/critic"
   install -d -m 700 "${ACTOR_SAVE_PATH}" "${CRITIC_SAVE_PATH}"
   ROLE_CONFIG_PATH="${RUN_ROOT}/megatron_roles.yaml"
   sed \
      -e "s|__ACTOR_LOAD_PATH__|${ACTOR_LOAD_PATH}|g" \
      -e "s|__CRITIC_LOAD_PATH__|${CRITIC_LOAD}|g" \
      -e "s|__ACTOR_SAVE_PATH__|${ACTOR_SAVE_PATH}|g" \
      -e "s|__CRITIC_SAVE_PATH__|${CRITIC_SAVE_PATH}|g" \
      "${ROLE_CONFIG_TEMPLATE}" > "${ROLE_CONFIG_PATH}"
   chmod 600 "${ROLE_CONFIG_PATH}"
   CKPT_ARGS+=(--megatron-config-path "${ROLE_CONFIG_PATH}")
   echo "Separate critic: HF=${CRITIC_LOAD} | actor=${ACTOR_LOAD_PATH}"
   echo "Role config: ${ROLE_CONFIG_PATH} | actor save=${ACTOR_SAVE_PATH} | critic save=${CRITIC_SAVE_PATH}"
fi

MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-131072}"
MAX_GEN_LEN="${MAX_GEN_LEN:-8192}"
AUTO_COMPACT_WINDOW="${AUTO_COMPACT_WINDOW:-100000}"
if [[ ! "${MAX_CONTEXT_LEN}" =~ ^[1-9][0-9]*$ || ! "${MAX_GEN_LEN}" =~ ^[1-9][0-9]*$ || ! "${AUTO_COMPACT_WINDOW}" =~ ^[1-9][0-9]*$ ]]; then
   echo "ERROR: MAX_CONTEXT_LEN, MAX_GEN_LEN, and AUTO_COMPACT_WINDOW must be positive integers" >&2
   exit 2
fi
if (( AUTO_COMPACT_WINDOW + MAX_GEN_LEN >= MAX_CONTEXT_LEN )); then
   echo "ERROR: AUTO_COMPACT_WINDOW + MAX_GEN_LEN must be less than MAX_CONTEXT_LEN" >&2
   exit 2
fi

ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
   --custom-generate-function-path examples.coding_agent_rl.generate.generate
   --prompt-data "${TRAIN_PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   "${ROLLOUT_COUNT_ARGS[@]}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-context-len "${MAX_CONTEXT_LEN}"
   --rollout-max-response-len "${MAX_GEN_LEN}"
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
   --recompute-num-layers 4
   # --recompute-granularity selective
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-4096}"
   --use-dynamic-batch-size
   --qkv-format thd
)

ALGO_ARGS=(
   --advantage-estimator sao
   --sao-batch-size "${SAO_BATCH_SIZE}"
   --critic-lr 5e-6
   --sao-critic-freeze-attention
   --sao-critic-update-ratio 2
   --sao-critic-warmup-steps 4
   --sao-gae-alpha 1.5
   --sao-dis-clip-low 0.8
   --sao-dis-clip-high 3.0
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
   # fully_async_rollout scales this per-engine concurrency by the number of
   # rollout engines. Keep the per-engine default bounded; 1024 is the
   # semi-async launcher default and would overfill the persistent pool here.
   --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE}"
   --sglang-mem-fraction-static "${ROLLOUT_MEM_UTILIZATION}"
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-16}"
   --sglang-incremental-streaming-output
   --sglang-page-size 256
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
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
   --update-weights-interval "${UPDATE_WEIGHTS_INTERVAL}"
)
# ============ Network ==========
export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(hostname -I | awk '{print $1}')}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-6379}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export NCCL_DEBUG=WARN
export SLIME_DESTROY_WORLD_PROCESS_GROUP="${SLIME_DESTROY_WORLD_PROCESS_GROUP:-0}"
export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}}"

# ============ SWE agent knobs ==========
export SWE_TRAIN_PROTOCOL="${SWE_TRAIN_PROTOCOL:-swebench}"
export SWE_EVAL_PROTOCOL="${SWE_EVAL_PROTOCOL:-swebench}"
export THETA_SERVICE_NAME="${THETA_SERVICE_NAME:-slime_qwen38_27b_64gpu_fully_async_${STAMP}}"
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
if [[ -z "${SLIME_AGENT_CC_EXTRA_ENVS:-}" ]]; then
   export SLIME_AGENT_CC_EXTRA_ENVS='{"PATH":"/opt/miniconda3/envs/testbed/bin:/opt/python/bin:/opt/miniconda3/bin:/usr/local/bin:/usr/bin:/bin","CONDA_PREFIX":"/opt/miniconda3/envs/testbed","CONDA_DEFAULT_ENV":"testbed"}'
fi
SETTINGS_JSON="{\"permissions\":{\"defaultMode\":\"bypassPermissions\"},\"autoCompactEnabled\":true,\"autoCompactWindow\":${AUTO_COMPACT_WINDOW}}"
export SLIME_AGENT_CC_EXTRA_ARGS="--settings '${SETTINGS_JSON}' --disable-slash-commands --disallowedTools WebFetch WebSearch Task Agent EnterWorktree ExitWorktree"
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_REQUEST_LIFECYCLE_LOG="${SGLANG_REQUEST_LIFECYCLE_LOG:-0}"

export no_proxy="127.0.0.1,${MASTER_ADDR}"
export NO_PROXY="${no_proxy}"

: "${RAY_DASHBOARD_ADDRESS:?RAY_DASHBOARD_ADDRESS is required}"
export RAY_ADDRESS="${RAY_DASHBOARD_ADDRESS%/}"
curl --noproxy '*' --fail --silent --show-error --max-time 5 \
   "${RAY_ADDRESS}/api/version" >/dev/null || {
   echo "ERROR: cannot reach Ray Dashboard: ${RAY_ADDRESS}" >&2
   exit 2
}
echo "Ray Dashboard: ${RAY_ADDRESS}"

# ============ Runtime env propagated to Ray workers ==========
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
    "ADAPTER_PUBLIC_HOST",
    "ADAPTER_BIND_HOST", "ADAPTER_PORT",
    "THETA_API_KEY", "THETA_SERVICE_NAME", "THETA_BASE_URL",
    "POD_IP", "SYSTEM_API_JWT_TAG", "DV_ENDPOINT_ADDR",
    "SLIME_AGENT_CC_EXTRA_ARGS", "SLIME_AGENT_CC_EXTRA_ENVS",
    "SLIME_AGENT_TRAJECTORY_SAVE", "SLIME_AGENT_TRAJECTORY_DIR",
    "SLIME_AGENT_TRAJECTORY_WRITE_CONCURRENCY",
    "SLIME_ADAPTER_MAX_TOOL_RESULT_CHARS",
    "SWE_CC_PROMPT", "SWE_TRAIN_PROTOCOL", "SWE_EVAL_PROTOCOL",
    "SWE_AGENT_TIME_BUDGET_SEC", "SWE_EVAL_TIMEOUT_SEC", "SWE_ROLLOUT_GUARD_SEC",
    "SWE_BOOT_CONCURRENCY",
    "SLIME_AGENT_SANDBOX_BACKEND",
    "SLIME_AGENT_ARCA_APP_NAME", "SLIME_AGENT_ARCA_BASE_URL", "SLIME_AGENT_ARCA_API_KEY",
    "SLIME_AGENT_ARCA_TEMPLATE_ID", "SLIME_AGENT_ARCA_IMAGE_REGISTRY",
    "SLIME_AGENT_ARCA_IMAGE_TAG_SUFFIX",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "SGLANG_REQUEST_LIFECYCLE_LOG",
    "NCCL_DEBUG", "SLIME_DESTROY_WORLD_PROCESS_GROUP",
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
    f"{os.environ['SLIME_DIR']}/tests",
    f"{os.environ['SLIME_DIR']}/third_party",
]
env["PYTHONPATH"] = ":".join(entry for entry in pythonpath_entries if entry)
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
with open(os.environ["RUNTIME_ENV_FILE"], "w", encoding="utf-8") as fp:
    json.dump({"env_vars": env}, fp)
PY

TRAIN_ARGS=(
   --
   python3 -u "${SLIME_DIR}/train_async.py"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${ALGO_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${MISC_ARGS[@]}"
)
if (( ${#PROFILE_ARGS[@]} )); then
   TRAIN_ARGS+=("${PROFILE_ARGS[@]}")
fi
if (( ${#DEBUG_ARGS[@]} )); then
   TRAIN_ARGS+=("${DEBUG_ARGS[@]}")
fi
ray job submit --no-wait --address="${RAY_ADDRESS}" \
   --runtime-env="${RUNTIME_ENV_FILE}" \
   "${TRAIN_ARGS[@]}" \
   2>&1 | tee "${LOG_FILE}"

echo "RUN_ROOT=${RUN_ROOT}"
