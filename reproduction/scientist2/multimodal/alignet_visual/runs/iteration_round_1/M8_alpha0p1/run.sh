#!/bin/bash
# Iteration 1 — M8 OOD sweep with alpha=0.1 aligned checkpoint.
# Uses the SAME datasets as the original M8 run so deltas are directly comparable.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export DATA_DIR=/data/zhenqian/data
export MODEL_DIR=/data/zhenqian/models

cd /data/zhenqian/Reproduction1/mechanica/multimodal/alignet_visual

conda run -n alignet_visual --live-stream python code/eval_downstream.py \
    --milestone M8 \
    --aligned_ckpt runs/iteration_round_1/M3_alpha0p1/checkpoint.pt \
    --unaligned_ckpt "${MODEL_DIR}/dinov2-base" \
    --output_dir runs/iteration_round_1/M8_alpha0p1 \
    --seed 42 --batch_size 64 \
    2>&1 | tee runs/iteration_round_1/M8_alpha0p1/log.txt
