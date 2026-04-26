#!/bin/bash
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$(pwd)/.torch_extensions}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

DATA_PATH=$1
SAVE_TAG=${SAVE_TAG:-self_repair_sft_lora_v2}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-"./output/sft_qwen3_8b_dir_3Ksample_1epoch"}
SAVE_PATH="./output/${SAVE_TAG}"
TRAIN_MODULE=${TRAIN_MODULE:-01_sft_train_self_repair}

NUM_GPUS=${NUM_GPUS:-4}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-4096}
MAX_STEPS=${MAX_STEPS:-20}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}

python -m torch.distributed.run \
    --nproc_per_node $NUM_GPUS \
    -m $TRAIN_MODULE \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --train_dataset_name_or_path $DATA_PATH \
    --output_dir $SAVE_PATH \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_strategy "no" \
    --preprocessing_num_workers 0 \
    --ddp_timeout 14400 \
    --max_seq_length $MAX_SEQ_LENGTH \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --num_train_epochs 1 \
    --max_steps $MAX_STEPS \
    --logging_steps 1 \
    --report_to tensorboard \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --deepspeed config/sft_config.json \
    --overwrite_output_dir \
    --bf16 True \
    --use_lora True \
    --lora_rank $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_target_modules "[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]"
