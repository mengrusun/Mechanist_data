#!/bin/bash
# Iteration 1 — type ② fix for C4a/C4b:
# Rerun M3 (aligned finetune) with reduced alignment weight alpha=0.1 (vs original 1.0).
# All other hyperparameters identical to original M3.
# Purpose: preserve pretrained representation more strongly, reduce general-ability degradation
# observed in M7 downstream one-shot (mean Δ=-0.284) and M8 in-distribution slices.
# Hypothesis: BREEDS OOD gain (C4b) survives while downstream loss (C4a) shrinks.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export DATA_DIR=/data/zhenqian/data
export MODEL_DIR=/data/zhenqian/models

cd /data/zhenqian/Reproduction1/mechanica/multimodal/alignet_visual

conda run -n alignet_visual --live-stream python code/align_student.py \
    --student_ckpt "${MODEL_DIR}/dinov2-base" \
    --teacher_cache "${DATA_DIR}/things_ooo_cache/teacher_feats_v1.h5" \
    --teacher_source head \
    --align_loss triplet_kl --temperature 1.0 \
    --alpha 0.1 \
    --tune_scope full \
    --imagenet_subset_size 40000 \
    --epochs 1 --lr 5e-5 --batch_size 64 --seed 42 \
    --output_dir runs/iteration_round_1/M3_alpha0p1 \
    2>&1 | tee runs/iteration_round_1/M3_alpha0p1/log.txt
