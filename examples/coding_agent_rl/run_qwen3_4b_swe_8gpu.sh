#!/usr/bin/env bash
# End-to-end SWE coding-agent RL on a single 8-GPU node (Qwen3-4B dense).
# See README.md for the dataset schema, env vars, and fan-out semantics.
# Run from a long-lived shell / tmux session on the Ray head node (a
# short-lived nohup launcher gets its Ray child processes cleaned up with it).

# Best-effort cleanup so a rerun does not collide with stale workers.
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
sleep 3
pkill -9 ray || true

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# model arch — sourced from scripts/models/qwen3-4B.sh (dense Qwen3-4B)
source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"

# ============ context length ============
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-65536}"
MAX_GEN_LEN="${MAX_GEN_LEN:-16384}"
CP_SIZE="${CP_SIZE:-2}"

# ============ paths — override before launching ============
HF_CHECKPOINT="${HF_CHECKPOINT:-/mnt/amed-s1/common/ckpt/gaochang/Qwen3-4B}"
REF_MODEL_PATH="${REF_MODEL_PATH:-/mnt/amed-s1/common/ckpt/gaochang/Qwen3-4B-tdst/}"
PROMPT_DATA="${PROMPT_DATA:-/personal/muchen/code_agent_data/swe_verified.jsonl}"

EXP_TAG="${EXP_TAG:-agent_only}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-${SLIME_DIR}/runs/${EXP_TAG}_${STAMP}}"

# ============ logging ============
LOG_DIR="${RUN_ROOT}"
mkdir -p "${LOG_DIR}/rollout_dumps"
LOG_FILE="${LOG_DIR}/run.log"
echo "======================================================================"
echo "Training log: ${LOG_FILE}"
echo "RUN_ROOT=${RUN_ROOT}"
echo "======================================================================"

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
   --num-rollout 10
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --rollout-max-context-len ${MAX_CONTEXT_LEN}
   --rollout-max-response-len ${MAX_GEN_LEN}
   --rollout-temperature 1.0
   --rollout-stop-token-ids 248046 248044
   --num-steps-per-rollout 1
)

PERF_ARGS=(
   --tensor-model-parallel-size ${TP_SIZE:-1}
   --sequence-parallel
   --pipeline-model-parallel-size ${PP_SIZE:-1}
   --context-parallel-size ${CP_SIZE}
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --max-tokens-per-gpu $((MAX_CONTEXT_LEN / CP_SIZE))
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
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus 8
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static ${ROLLOUT_MEM_UTILIZATION:-0.7}
   --sglang-context-length ${MAX_CONTEXT_LEN}
   --sglang-tool-call-parser qwen25
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

# Allow SGLang to extend context beyond model's max_position_embeddings
# (Qwen3-4B defaults to 40960 but RoPE scaling supports longer contexts).
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

# Adapter context compression: replace CLI's ~70k system prompt with a compact
# version, filter tools to Bash/Read/Edit/Write, truncate long tool_results.
export SLIME_ADAPTER_SYSTEM_PROMPT="${SLIME_ADAPTER_SYSTEM_PROMPT:-1}"
export SLIME_ADAPTER_TOOL_WHITELIST="${SLIME_ADAPTER_TOOL_WHITELIST:-Bash,Read,Edit,Write}"
export SLIME_ADAPTER_MAX_TOOL_RESULT_CHARS="${SLIME_ADAPTER_MAX_TOOL_RESULT_CHARS:-10000}"

# ============ ray cluster network ============
export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-$(hostname -I | awk '{print $1}')}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-6379}}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${MLP_SOCKET_IFNAME:-eth0}}"

# ============ SWE / claude-code rollout knobs ============

export SWE_AGENT="${SWE_AGENT:-claude_code}"
export SWE_TRAIN_PROTOCOL="${SWE_TRAIN_PROTOCOL:-swebench}"
export SWE_EVAL_PROTOCOL="${SWE_EVAL_PROTOCOL:-swebench}"

# ============ sandbox provider ============
# Set SLIME_SANDBOX_PROVIDER=local to use LocalSandbox (no E2B/Docker).
# Default is e2b for backward compatibility.
export SLIME_SANDBOX_PROVIDER="${SLIME_SANDBOX_PROVIDER:-local}"

if [[ "${SLIME_SANDBOX_PROVIDER}" == "local" ]]; then
  export LOCAL_SANDBOX_WORKSPACE_ROOT="${LOCAL_SANDBOX_WORKSPACE_ROOT:-/personal/muchen/swe_workspaces}"
  export LOCAL_SANDBOX_CLEANUP_ON_EXIT="${LOCAL_SANDBOX_CLEANUP_ON_EXIT:-1}"
  # Local sandbox: adapter reachable at localhost
  export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-127.0.0.1}"
else
  export E2B_API_KEY="${E2B_API_KEY:-e2b_0000000000000000000000000000000000000000}"
  # Metadata key your gateway routes images by; `image` is the neutral default.
  export SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY="${SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY:-image}"
  # ADAPTER_PUBLIC_HOST must be routable from inside the sandbox (not 127.0.0.1).
  export ADAPTER_PUBLIC_HOST="${ADAPTER_PUBLIC_HOST:-${MASTER_ADDR:-${MLP_WORKER_0_HOST:-127.0.0.1}}}"
fi

export SLIME_AGENT_NODE_TARBALL="${SLIME_AGENT_NODE_TARBALL:-}"
export SLIME_AGENT_CC_TARBALL="${SLIME_AGENT_CC_TARBALL:-}"

export ADAPTER_BIND_HOST="${ADAPTER_BIND_HOST:-0.0.0.0}"
export ADAPTER_PORT="${ADAPTER_PORT:-18001}"

export SWE_AGENT_TIME_BUDGET_SEC="${SWE_AGENT_TIME_BUDGET_SEC:-1800}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-600}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-32}"

# autoCompactWindow (20k) < MAX_CONTEXT_LEN (65536): compact early so the CLI
# has more room before the 65536 training-side cap, avoiding the overflow loop
# where prompt exceeds max_context_tokens and the adapter returns empty responses.
# `investigator` is a read-only sub-agent (a concrete dispatch target).
# WebFetch/WebSearch off (no outbound internet).
SETTINGS_JSON='{"permissions":{"defaultMode":"bypassPermissions"},"autoCompactEnabled":true,"autoCompactWindow":20000}'
AGENTS_JSON='{"investigator":{"description":"Searches the repo for relevant files before any edit","prompt":"You are an investigator sub-agent. Use Grep/Read/Glob to find every file relevant to the user task, then return a short bulleted summary. Do NOT edit anything.","tools":["Grep","Read","Glob"]}}'
export SLIME_AGENT_CC_EXTRA_ARGS="--settings '${SETTINGS_JSON}' --disable-slash-commands --agents '${AGENTS_JSON}' --disallowedTools WebFetch WebSearch"

# Optional: require dispatching the investigator before any edit, to maximize sub-agent fan-out.
# export SWE_CC_PROMPT="Read PROBLEM_STATEMENT.md. BEFORE editing any file, dispatch the 'investigator' sub-agent (via the Agent tool with subagent_type=investigator) to locate every file relevant to the issue. Then fix the issue and run the tests."

# ============ proxy bypass for in-cluster traffic ============
export no_proxy="127.0.0.1,${MASTER_ADDR},${ADAPTER_PUBLIC_HOST}"
export NO_PROXY="${no_proxy}"

cd "${SLIME_DIR}"

# ============ bring up ray cluster (single node, 8 GPUs) ============
ACTOR_NUM_NODES=1
ACTOR_NUM_GPUS_PER_NODE=8

ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${ACTOR_NUM_GPUS_PER_NODE}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

echo "Waiting for Ray cluster to stabilize..."
sleep 10
ray status

# ============ runtime env propagated to ray workers ============
export SLIME_DIR
RUNTIME_ENV_JSON=$(python3 - <<PY
import json, os
keys = (
    "no_proxy", "NO_PROXY",
    "SWE_AGENT",
    "E2B_API_KEY", "ADAPTER_PUBLIC_HOST",
    "SLIME_AGENT_NODE_TARBALL", "SLIME_AGENT_CC_TARBALL",
    "SWE_AGENT_TIME_BUDGET_SEC", "SWE_EVAL_TIMEOUT_SEC", "SWE_BOOT_CONCURRENCY",
    "ADAPTER_BIND_HOST", "ADAPTER_PORT",
    "SLIME_AGENT_CC_EXTRA_ARGS",
    "SLIME_AGENT_CC_EXTRA_ENVS",
    "SWE_CC_PROMPT",
    "SWE_TRAIN_PROTOCOL", "SWE_EVAL_PROTOCOL",
    "SLIME_AGENT_SANDBOX_IMAGE_METADATA_KEY",
    "SLIME_SANDBOX_PROVIDER",
    "LOCAL_SANDBOX_WORKSPACE_ROOT",
    "LOCAL_SANDBOX_CLEANUP_ON_EXIT",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN",
    "SLIME_ADAPTER_SYSTEM_PROMPT",
    "SLIME_ADAPTER_TOOL_WHITELIST",
    "SLIME_ADAPTER_MAX_TOOL_RESULT_CHARS",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
env["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
env["MASTER_PORT"] = os.environ.get("MASTER_PORT", "")
env["GLOO_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["TP_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["NCCL_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"]
env["PYTHONPATH"] = f"/root/Megatron-LM/:{os.environ['SLIME_DIR']}"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env["NCCL_NVLS_ENABLE"] = "0"
print(json.dumps({"env_vars": env}))
PY
)

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u train.py \
   --actor-num-nodes "${ACTOR_NUM_NODES}" \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
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