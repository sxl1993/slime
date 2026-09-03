#!/usr/bin/env bash
# SWE coding-agent RL with Qwen3.8-27B + ARCA sandbox, 144 logical GPUs.
# Fully-async mode keeps a background rollout pool warm across rollout
# boundaries and returns completed groups as they become available.
# Actor and critic each use 64 logical GPUs; rollout uses 16 logical GPUs.
# Actor and critic share a placement group, so physical reservation is 80 GPUs.
#
# Model arch sourced from scripts/models/qwen3.5-27B.sh (qwen3_5 hybrid 64L).
# The fully-async collector requeues ABORTED groups from scratch; it does not
# provide partial-rollout session resume.

set -euo pipefail
export PYTHONUNBUFFERED=1

# ============ Training configuration ============
SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
UPDATE_WEIGHTS_INTERVAL="${UPDATE_WEIGHTS_INTERVAL:-2}"
ACTOR_LOAD_PATH="${REF_MODEL_PATH:-/mnt/amedelastic-m/common/ckpt/muchen/Qwen3.8-27B_torch_dist}"
CRITIC_LOAD="${CRITIC_LOAD:-/mnt/amedelastic-m/common/ckpt/muchen/Qwen3.8-27B-Critic/}"
SAO_BATCH_SIZE="${SAO_BATCH_SIZE:-8}"
SAO_CRITIC_UPDATE_RATIO="${SAO_CRITIC_UPDATE_RATIO:-2}"
SAO_CRITIC_WARMUP_STEPS="${SAO_CRITIC_WARMUP_STEPS:-5}"

SLIME_DIR="${SLIME_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." &>/dev/null && pwd)}"
export MEGATRON_PATH="${MEGATRON_PATH:-/root/Megatron-LM}"
export SGLANG_PATH="${SGLANG_PATH:-}"

# ============ ARCA sandbox ============
export SLIME_AGENT_SANDBOX_BACKEND=arca
export SLIME_AGENT_ARCA_APP_NAME="${SLIME_AGENT_ARCA_APP_NAME:-arcaslimeagentrl}"
export SLIME_AGENT_ARCA_BASE_URL="${SLIME_AGENT_ARCA_BASE_URL:-http://arca-sandbox.global.alipay.com:8080}"
export THETA_API_KEY="UlRvc3YoQBg0lQjeDxwen7OTPSTUd9Xh"
export SLIME_AGENT_ARCA_API_KEY="665934ee53b64b0f83c1e8115f6e0dd5"
export SLIME_AGENT_ARCA_TEMPLATE_ID="${SLIME_AGENT_ARCA_TEMPLATE_ID:-ARCA-TEMPLATE-000000004480168f}"
export SLIME_AGENT_ARCA_IMAGE_REGISTRY="${SLIME_AGENT_ARCA_IMAGE_REGISTRY:-asr.antgroup-inc.cn/arcaslimeagentrl/sweb.instance}"
export SLIME_AGENT_ARCA_IMAGE_TAG_SUFFIX="${SLIME_AGENT_ARCA_IMAGE_TAG_SUFFIX:-claude-code-2.1.220-v1}"

# ============ Parallelism ============
# Fixed layout: actor logical = 64, critic logical = 64, rollout = 16,
# logical role footprint = 144; actor and critic share placement group, so
# physical reservation = 80 GPUs.
ACTOR_NUM_NODES=8
ACTOR_NUM_GPUS_PER_NODE=8
ROLLOUT_NUM_GPUS=16
TP_SIZE="${TP_SIZE:-4}"
PP_SIZE=4
CP_SIZE="${CP_SIZE:-4}"
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-8}"

# ============ Model ============
source "${SLIME_DIR}/scripts/models/qwen3.5-27B.sh"

TRAIN_PROMPT_DATA="${PROMPT_DATA:-/personal/muchen/code_agent_data/swe_verified_v5.jsonl}"

# ============ Run directory ============
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${SLIME_DIR}/runs/${EXP_TAG:-arca-sandbox-64gpu-fully-async-27b}_${RUN_ID:-${STAMP}_$$}}"
mkdir -p "$(dirname -- "${RUN_ROOT}")"
if ! mkdir -m 700 "${RUN_ROOT}"; then
   echo "ERROR: RUN_ROOT already exists or cannot be created: ${RUN_ROOT}" >&2
   exit 2
fi
echo "Training log: ${RUN_ROOT}/run.log"
echo "RUN_ROOT=${RUN_ROOT} | backend=${SLIME_AGENT_SANDBOX_BACKEND} | mode=fully_async | actor logical=64 (${ACTOR_NUM_NODES}x${ACTOR_NUM_GPUS_PER_NODE}) | critic logical=64 | rollout logical=${ROLLOUT_NUM_GPUS} | logical role footprint=144 | physical reservation=80 | parallelism=TP${TP_SIZE}xPP${PP_SIZE}xCP${CP_SIZE} | update_weights_interval=${UPDATE_WEIGHTS_INTERVAL}"

# ============ Profiler ============
PROFILE="${PROFILE:-0}"
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

# ============ Checkpoint and roles ============
SAVE_PATH="${SAVE_PATH:-${RUN_ROOT}/checkpoints}"
CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT:-/mnt/amedelastic-m/common/ckpt/muchen/Qwen3.8-27B}"
   --ref-load "${REF_MODEL_PATH:-/mnt/amedelastic-m/common/ckpt/muchen/Qwen3.8-27B_torch_dist}"
   --save "${SAVE_PATH}"
   --save-interval "${SAVE_INTERVAL}"
)
echo "Checkpoint path: ${SAVE_PATH} | interval=${SAVE_INTERVAL} steps"

ACTOR_SAVE_PATH="${SAVE_PATH}/actor"
CRITIC_SAVE_PATH="${SAVE_PATH}/critic"
install -d -m 700 "${ACTOR_SAVE_PATH}" "${CRITIC_SAVE_PATH}"
ROLE_CONFIG_PATH="${RUN_ROOT}/megatron_roles.yaml"
cat > "${ROLE_CONFIG_PATH}" <<EOF
megatron:
  - name: default
    role: actor
    overrides:
      load: ${ACTOR_LOAD_PATH}
      save: ${ACTOR_SAVE_PATH}
  - name: default
    role: critic
    overrides:
      load: ${CRITIC_LOAD}
      save: ${CRITIC_SAVE_PATH}
EOF
chmod 600 "${ROLE_CONFIG_PATH}"
CKPT_ARGS+=(--megatron-config-path "${ROLE_CONFIG_PATH}")
echo "Separate critic role: load=${CRITIC_LOAD} | actor load=${ACTOR_LOAD_PATH}"
echo "Role config: ${ROLE_CONFIG_PATH} | actor save=${ACTOR_SAVE_PATH} | critic save=${CRITIC_SAVE_PATH}"

MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-131072}"
MAX_GEN_LEN="${MAX_GEN_LEN:-8192}"
AUTO_COMPACT_WINDOW="${AUTO_COMPACT_WINDOW:-100000}"
if (( AUTO_COMPACT_WINDOW + MAX_GEN_LEN >= MAX_CONTEXT_LEN )); then
   echo "ERROR: AUTO_COMPACT_WINDOW + MAX_GEN_LEN must be less than MAX_CONTEXT_LEN" >&2
   exit 2
fi

# ============ Rollout ============
ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
   --custom-generate-function-path examples.coding_agent_rl.generate.generate
   --prompt-data "${TRAIN_PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --apply-chat-template
   --num-epoch "${NUM_EPOCH:-3}"
   --rollout-batch-size "${SAO_BATCH_SIZE}"
   --n-samples-per-prompt 1
   --rollout-max-context-len "${MAX_CONTEXT_LEN}"
   --rollout-max-response-len "${MAX_GEN_LEN}"
   --rollout-temperature 1.0
   --rollout-stop-token-ids 248046 248044
   --num-steps-per-rollout 1
)

# ============ Performance ============
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

# ============ Algorithm ============
ALGO_ARGS=(
   --advantage-estimator sao
   --sao-batch-size "${SAO_BATCH_SIZE}"
   --critic-lr 5e-6
   --sao-critic-freeze-attention
   --sao-critic-update-ratio "${SAO_CRITIC_UPDATE_RATIO}"
   --sao-critic-warmup-steps "${SAO_CRITIC_WARMUP_STEPS}"
   --sao-policy-gae-alpha 1.5
   --sao-dis-clip-low 0.8
   --sao-dis-clip-high 3.0
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

# ============ Optimizer ============
OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# ============ SGLang ============
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_TP_SIZE}"
   --sglang-mem-fraction-static 0.80
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-16}"
   --sglang-incremental-streaming-output
   --sglang-page-size 256
   --router-disable-retries
   --sglang-disable-custom-all-reduce
   --sglang-tool-call-parser qwen3_coder
   --sglang-reasoning-parser qwen3
)

# ============ Miscellaneous ============
MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --actor-num-nodes "${ACTOR_NUM_NODES}"
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
   --num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}"
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
   --update-weights-interval "${UPDATE_WEIGHTS_INTERVAL}"
)
# ============ Network ============
export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(hostname -I | awk '{print $1}')}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-6379}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export SLIME_DESTROY_WORLD_PROCESS_GROUP="${SLIME_DESTROY_WORLD_PROCESS_GROUP:-0}"

# ============ SWE agent ============
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
export SLIME_AGENT_CC_EXTRA_ENVS='{"PATH":"/opt/miniconda3/envs/testbed/bin:/opt/python/bin:/opt/miniconda3/bin:/usr/local/bin:/usr/bin:/bin","CONDA_PREFIX":"/opt/miniconda3/envs/testbed","CONDA_DEFAULT_ENV":"testbed"}'
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

# ============ Runtime environment propagated to Ray workers ============
export SLIME_DIR
RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

keys = (
    "no_proxy",
    "NO_PROXY",
    "ADAPTER_BIND_HOST",
    "ADAPTER_PORT",
    "THETA_API_KEY",
    "THETA_SERVICE_NAME",
    "THETA_BASE_URL",
    "POD_IP",
    "SYSTEM_API_JWT_TAG",
    "DV_ENDPOINT_ADDR",
    "SLIME_AGENT_CC_EXTRA_ARGS",
    "SLIME_AGENT_CC_EXTRA_ENVS",
    "SLIME_AGENT_TRAJECTORY_SAVE",
    "SLIME_AGENT_TRAJECTORY_DIR",
    "SLIME_AGENT_TRAJECTORY_WRITE_CONCURRENCY",
    "SLIME_ADAPTER_MAX_TOOL_RESULT_CHARS",
    "SWE_CC_PROMPT",
    "SWE_TRAIN_PROTOCOL",
    "SWE_EVAL_PROTOCOL",
    "SWE_AGENT_TIME_BUDGET_SEC",
    "SWE_EVAL_TIMEOUT_SEC",
    "SWE_ROLLOUT_GUARD_SEC",
    "SWE_BOOT_CONCURRENCY",
    "SLIME_AGENT_SANDBOX_BACKEND",
    "SLIME_AGENT_ARCA_APP_NAME",
    "SLIME_AGENT_ARCA_BASE_URL",
    "SLIME_AGENT_ARCA_API_KEY",
    "SLIME_AGENT_ARCA_TEMPLATE_ID",
    "SLIME_AGENT_ARCA_IMAGE_REGISTRY",
    "SLIME_AGENT_ARCA_IMAGE_TAG_SUFFIX",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN",
    "SGLANG_REQUEST_LIFECYCLE_LOG",
    "NCCL_DEBUG",
    "SLIME_DESTROY_WORLD_PROCESS_GROUP",
)
env = {key: os.environ[key] for key in keys if key in os.environ}
env["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
env["MASTER_PORT"] = os.environ.get("MASTER_PORT", "")
env["GLOO_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["TP_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["NCCL_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]
env["PYTHONPATH"] = f"{os.environ['MEGATRON_PATH']}:{os.environ['SLIME_DIR']}"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
print(json.dumps({"env_vars": env}))
PY
)"

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
ray job submit --no-wait --address="${RAY_ADDRESS}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   "${TRAIN_ARGS[@]}" \
   2>&1 | tee "${RUN_ROOT}/run.log"

echo "RUN_ROOT=${RUN_ROOT}"
