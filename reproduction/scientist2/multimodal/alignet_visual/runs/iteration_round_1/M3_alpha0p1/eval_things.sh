#!/bin/bash
# Iteration 1 — M3-eval THINGS multi-level Spearman under alpha=0.1 aligned checkpoint.
# Purpose: confirm that C2a (aggregate Spearman gain) still holds at alpha=0.1.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1
export DATA_DIR=/data/zhenqian/data
export MODEL_DIR=/data/zhenqian/models

cd /data/zhenqian/Reproduction1/mechanica/multimodal/alignet_visual

conda run -n alignet_visual --live-stream python code/eval_student_similarity.py \
    --student_ckpt runs/iteration_round_1/M3_alpha0p1/checkpoint.pt \
    --split heldout \
    --levels_construction things_metadata \
    --output_json runs/iteration_round_1/M3_alpha0p1/eval_things_multilevel.json \
    --seed 42 \
    2>&1 | tee runs/iteration_round_1/M3_alpha0p1/eval_things.log
