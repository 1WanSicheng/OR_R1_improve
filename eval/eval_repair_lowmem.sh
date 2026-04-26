#!/bin/bash
set -euo pipefail

MODEL_PATH=$1
DATASET_PATH=$2
SAVE_DIR=$3

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

TP_SIZE=${TP_SIZE:-4}
MAX_TOKENS=${MAX_TOKENS:-2048}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.65}
MAX_SAMPLES=${MAX_SAMPLES:-}

ARGS=()
if [ -n "$MAX_SAMPLES" ]; then
  ARGS+=(--max_samples "$MAX_SAMPLES")
fi

python -m eval.generate_repair_lowmem \
  --model_name_or_path "$MODEL_PATH" \
  --dataset_name "$DATASET_PATH" \
  --dataset_split test \
  --save_dir "$SAVE_DIR" \
  --tensor_parallel_size ${TP_SIZE} \
  --max_tokens ${MAX_TOKENS} \
  --max_model_len ${MAX_MODEL_LEN} \
  --max_num_seqs ${MAX_NUM_SEQS} \
  --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
  --enforce_eager \
  --overwrite \
  "${ARGS[@]}"
