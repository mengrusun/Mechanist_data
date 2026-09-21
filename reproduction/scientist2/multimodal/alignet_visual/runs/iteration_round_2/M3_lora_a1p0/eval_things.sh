#!/bin/bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2
export DATA_DIR=/data/zhenqian/data
export MODEL_DIR=/data/zhenqian/models

cd /data/zhenqian/Reproduction1/mechanica/multimodal/alignet_visual

conda run -n alignet_visual --live-stream python code/eval_student_similarity.py \
    --student_ckpt runs/iteration_round_2/M3_lora_a1p0/checkpoint.pt \
    --split heldout \
    --levels_construction things_metadata \
    --output_json runs/iteration_round_2/M3_lora_a1p0/eval_things_multilevel.json \
    --seed 42 \
    2>&1 | tee runs/iteration_round_2/M3_lora_a1p0/eval_things.log
