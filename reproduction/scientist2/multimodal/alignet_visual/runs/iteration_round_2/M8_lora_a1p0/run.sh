#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
export DATA_DIR=/data/zhenqian/data
export MODEL_DIR=/data/zhenqian/models
cd /data/zhenqian/Reproduction1/mechanica/multimodal/alignet_visual
conda run -n alignet_visual --live-stream python code/eval_downstream.py \
    --milestone M8 \
    --aligned_ckpt runs/iteration_round_2/M3_lora_a1p0/checkpoint.pt \
    --unaligned_ckpt "${MODEL_DIR}/dinov2-base" \
    --output_dir runs/iteration_round_2/M8_lora_a1p0 \
    --seed 42 --batch_size 64 \
    2>&1 | tee runs/iteration_round_2/M8_lora_a1p0/log.txt
