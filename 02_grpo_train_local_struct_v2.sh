#!/bin/bash
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$(pwd)/.torch_extensions}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false

MODEL_NAME=$1
MODEL_NAME_OR_PATH="./output/${MODEL_NAME}"

SAVE_TAG=${SAVE_TAG:-struct_v2_20step}
SAVE_PATH="./output/lora_grpo_${MODEL_NAME}_${SAVE_TAG}"
DS_CONFIG_PATH=${DS_CONFIG_PATH:-"./config/grpo_config.json"}

GPUS_PER_NODE=${GPUS_PER_NODE:-$(python -c 'import torch; print(torch.cuda.device_count())')}
NNODES=1
NODE_RANK=0
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6012}

DATASET_PATH=${DATASET_PATH:-./datasets/trainset/train_100.jsonl}
NUM_GENERATIONS=${NUM_GENERATIONS:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1536}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-2048}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
SAVE_STEPS=${SAVE_STEPS:-20}
MAX_STEPS=${MAX_STEPS:-20}
LEARNING_RATE=${LEARNING_RATE:-5e-5}

LORA_R=${LORA_R:-32}
LORA_ALPHA=${LORA_ALPHA:-64}

LAMBDA_STRUCT=${LAMBDA_STRUCT:-0.35}
ALPHA_OBJ=${ALPHA_OBJ:-1.0}
ALPHA_VAR=${ALPHA_VAR:-1.0}
ALPHA_CON=${ALPHA_CON:-1.0}
ALPHA_ALIGN=${ALPHA_ALIGN:-0.0}
STRUCT_LOG_DIR=${STRUCT_LOG_DIR:-./logs_struct_${SAVE_TAG}}

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

python -m torch.distributed.run $DISTRIBUTED_ARGS ./02_grpo_train_struct.py \
    --deepspeed ${DS_CONFIG_PATH} \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $SAVE_PATH \
    --num_generations ${NUM_GENERATIONS} \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --num_train_epochs 1 \
    --max_steps ${MAX_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler_type cosine \
    --per_device_eval_batch_size 1 \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit 20 \
    --logging_dir ./logs_v0_${SAVE_TAG} \
    --logging_strategy steps \
    --logging_steps 1 \
    --warmup_steps 5 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --report_to tensorboard \
    --bf16 True \
    --logging_first_step \
    --use_peft \
    --lora_target_modules all-linear \
    --lora_r ${LORA_R} \
    --lora_alpha ${LORA_ALPHA} \
    --save_only_model \
    --max_prompt_length ${MAX_PROMPT_LENGTH} \
    --max_completion_length ${MAX_COMPLETION_LENGTH} \
    --dataset_path ${DATASET_PATH} \
    --lambda_struct ${LAMBDA_STRUCT} \
    --alpha_obj ${ALPHA_OBJ} \
    --alpha_var ${ALPHA_VAR} \
    --alpha_con ${ALPHA_CON} \
    --alpha_align ${ALPHA_ALIGN} \
    --log_dir ${STRUCT_LOG_DIR}
