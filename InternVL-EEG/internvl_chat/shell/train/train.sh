#!/usr/bin/env bash
# Training script for UniMind: CTQFormer v2 with 2 routers, 10 datasets, zscore norm.
#
# Usage:
#   [GPUS=8] [BATCH_SIZE=128] [OUTPUT_DIR=/path/to/out] bash <this_script>
#
# Required environment variables (must be set before running):
#   MODEL_PATH   : path to InternVL2-8B pretrained weights directory
#   OUTPUT_DIR   : directory to save checkpoints and logs
#   DATA_CONFIG  : path to training dataset JSON (default: shell/data/train_datasets.json)
#
# Optional overrides:
#   GPUS                 (default 8)
#   BATCH_SIZE           (default 128 — total across all GPUs)
#   PER_DEVICE_BATCH_SIZE (default 16 — per-GPU, gradient accumulation computed automatically)

set -euo pipefail
set -x

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

MODEL_PATH=${MODEL_PATH:-"pretrained/InternVL2-8B"}
OUTPUT_DIR=${OUTPUT_DIR:-"work_dirs/unfreezelabram_finetune_10data_ctqformer_v2_2router_merge_zscore_4e5_bs128"}
DATA_CONFIG=${DATA_CONFIG:-"shell/data/train_datasets.json"}

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export MASTER_PORT=${MASTER_PORT:-36319}
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

mkdir -p "$OUTPUT_DIR"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "${MODEL_PATH}" \
  --resample_method "ctqformer_v2_2router_merge" \
  --num_cquery_tokens 2 \
  --num_tquery_tokens 2 \
  --num_query_pool 16 \
  --conv_style "internlm2-chat" \
  --output_dir "${OUTPUT_DIR}" \
  --train_meta_path "${DATA_CONFIG}" \
  --normalize_type "zscore" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone False \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 10 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 1000 \
  --save_total_limit 40 \
  --learning_rate 4e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size False \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
