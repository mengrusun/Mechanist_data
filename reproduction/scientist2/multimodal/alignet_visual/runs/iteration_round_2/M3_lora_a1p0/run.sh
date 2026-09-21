#!/bin/bash
# Iteration 2 — type ② fix for C4a/C4b: LoRA-scope alignment finetune (freeze backbone; train only ~590K adapter params).
# Rationale: iteration 1's alpha=0.1 rerun did NOT recover downstream utility, indicating that full-backbone parameter drift
# (not KD-loss weight) is the root cause of general-ability degradation. LoRA is the reviewer's own "safer variant" — it
# freezes the pretrained DINOv2 backbone entirely and adapts only low-rank q/k/v/output projections.
# Alpha kept at 1.0 for strict comparability to original M3 (per reviewer suggestion).
set -euo pipefail

export CUDA_VISIBLE_DEVICES=2
export DATA_DIR=/data/zhenqian/data
export MODEL_DIR=/data/zhenqian/models

cd /data/zhenqian/Reproduction1/mechanica/multimodal/alignet_visual

conda run -n alignet_visual --live-stream python code/align_student.py \
    --student_ckpt "${MODEL_DIR}/dinov2-base" \
    --teacher_cache "${DATA_DIR}/things_ooo_cache/teacher_feats_v1.h5" \
    --teacher_source head \
    --align_loss triplet_kl --temperature 1.0 \
    --alpha 1.0 \
    --tune_scope lora --lora_r 16 --lora_alpha 32 \
    --imagenet_subset_size 40000 \
    --epochs 1 --lr 5e-5 --batch_size 64 --seed 42 \
    --output_dir runs/iteration_round_2/M3_lora_a1p0 \
    2>&1 | tee runs/iteration_round_2/M3_lora_a1p0/log.txt
