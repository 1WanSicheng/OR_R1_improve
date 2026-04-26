#!/bin/bash
set -euo pipefail

MODEL_PATH=$1
DATASET_PATH=$2
SAVE_DIR=$3

export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

TP_SIZE=${TP_SIZE:-4}
MAX_TOKENS=${MAX_TOKENS:-3072}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.65}

python -m eval.generate_lowmem \
  --model_name_or_path "$MODEL_PATH" \
  --dataset_name "$DATASET_PATH" \
  --dataset_split test \
  --tensor_parallel_size ${TP_SIZE} \
  --save_dir "$SAVE_DIR" \
  --topk 1 \
  --decoding greedy \
  --max_tokens ${MAX_TOKENS} \
  --max_model_len ${MAX_MODEL_LEN} \
  --max_num_seqs ${MAX_NUM_SEQS} \
  --gpu_memory_utilization ${GPU_MEMORY_UTILIZATION} \
  --enforce_eager

python -m eval.execute \
  --input_file "${SAVE_DIR}/generated.jsonl" \
  --output_file "${SAVE_DIR}/executed.jsonl" \
  --question_field question \
  --answer_field answer \
  --timeout 600 \
  --max_workers 16 \
  --verbose
