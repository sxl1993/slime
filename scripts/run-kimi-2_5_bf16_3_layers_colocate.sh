#!/bin/bash

NUM_GPUS=${SLIME_SCRIPT_NUM_GPUS:-8}
# DATASET_LOCAL_NAME=$(basename "$DATASET_NAME")

pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex
export PYTHONBUFFERED=16

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/kimi-k2-thinking.sh"

# Common args
CKPT_ARGS=(
   --hf-checkpoint /mnt/amed-s1/common/ckpt/gaochang/Kimi-K2.5-bb16
   # --ref-load /mnt/amed-s1/common/ckpt/gaochang/Kimi-VL-A3B-Thinking-2506_tdst
  #  --save /amed/share/s1-amed-spfs-ckpt/muchen/save/Kimi-VL-A3B-Instruct-slime
  #  --save-interval 100
  --rotary-base 50000
)

ROLLOUT_ARGS=(
   --prompt-data /personal/train_base64.parquet
   --input-key problem
   --label-key answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 100
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 0.8
   --global-batch-size 128
   --rollout-num-gpus 4
   # --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
)

# required for vlm datasets
MULTIMODAL_KEYS='{"image": "images"}'

# EVAL_ARGS=(
#    --eval-interval 1
#    --eval-prompt-data /amed/share/s1-amed-spfs-ckpt/muchen/datasets/geo3k_imgurl/test.parquet
#    --n-samples-per-eval-prompt 1
#    --eval-max-response-len 4096
# )

GRPO_ARGS=(
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
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
   # --sglang-disable-cuda-graph
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256
)

# Wandb args (only if WANDB_API_KEY is set)
WANDB_ARGS=()

MISC_ARGS=(
   # --colocate
)


BACKEND_ARGS=(
   --train-backend megatron
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
   --megatron-to-hf-mode bridge
   --model-name kimi_k25
   # --data-pad-size-multiplier 1
)


# Start Ray if not using external Ray
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus ${NUM_GPUS} --disable-usage-stats


# Build runtime env
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 2 \
   --actor-num-gpus-per-node 4 \
   --colocate \
   --multimodal-keys "${MULTIMODAL_KEYS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${BACKEND_ARGS[@]} \
   ${MISC_ARGS[@]}