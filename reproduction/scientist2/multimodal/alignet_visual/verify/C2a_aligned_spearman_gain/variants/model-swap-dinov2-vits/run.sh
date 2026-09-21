#!/usr/bin/env bash
# Variant: model-swap-dinov2-vits
# Claim: C2a — Aligned DINOv2 ViT-B improves aggregate Spearman with human THINGS-triplet similarity
# Student swap: DINOv2 ViT-S (dinov2-small, 384-dim, 21M params) instead of DINOv2 ViT-B (768-dim, 86M params)
# Teacher: SigLIP-So400m (FIXED — never swapped per task.md hard constraint)
# All other hyperparameters identical to M3 (aligned finetune) and M4 (unaligned eval)
#
# GPU pinning: CUDA_VISIBLE_DEVICES=0,1,2,3 (task.md hard constraint)
# Code review status: PASS (no CRITICAL issues)
#   - Minor: eval script updated to use dinov2-small architecture (main eval_student_similarity.py
#     hardcodes dinov2-base arch; variant uses scripts/eval_student_similarity_vits.py)
#   - align_student.py is generic: loads any AutoModel from student_ckpt path, handles ViT-S fine

set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

DATA_DIR=/data/zhenqian/data
MODEL_DIR=/data/zhenqian/models
VARIANT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "${VARIANT_DIR}/../../../../" && pwd)"

ALIGNED_OUT="${WORK_DIR}/runs/verify/C2a_model_swap_dinov2_vits/aligned"
UNALIGNED_OUT="${WORK_DIR}/runs/verify/C2a_model_swap_dinov2_vits/unaligned"
mkdir -p "$ALIGNED_OUT" "$UNALIGNED_OUT"

cd "$WORK_DIR"

echo "=== Step 1: Align DINOv2 ViT-S (student model swap variant for C2a) ==="
echo "   Student: ${MODEL_DIR}/dinov2-small (384-dim, 21M params)"
echo "   Teacher cache: ${DATA_DIR}/things_ooo_cache/teacher_feats_v1.h5 (feats_head column)"
echo "   Hyperparams: lr=5e-5, alpha=1.0, T=1.0, batch=64, epochs=1, seed=42, subset=40k"
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n alignet_visual python code/align_student.py \
    --student_ckpt "${MODEL_DIR}/dinov2-small" \
    --teacher_cache "${DATA_DIR}/things_ooo_cache/teacher_feats_v1.h5" \
    --teacher_source head \
    --align_loss triplet_kl \
    --temperature 1.0 \
    --alpha 1.0 \
    --tune_scope full \
    --imagenet_subset_size 40000 \
    --epochs 1 \
    --lr 5e-5 \
    --batch_size 64 \
    --seed 42 \
    --output_dir "$ALIGNED_OUT"

echo "=== Step 2: Eval aligned DINOv2 ViT-S on THINGS held-out ==="
CUDA_VISIBLE_DEVICES=0 conda run -n alignet_visual python "${VARIANT_DIR}/scripts/eval_student_similarity_vits.py" \
    --student_ckpt "${ALIGNED_OUT}/checkpoint.pt" \
    --split heldout \
    --levels_construction things_metadata \
    --output_json "${ALIGNED_OUT}/eval_things_multilevel.json"

echo "=== Step 3: Eval UNALIGNED DINOv2 ViT-S on THINGS held-out (variant baseline) ==="
CUDA_VISIBLE_DEVICES=0 conda run -n alignet_visual python "${VARIANT_DIR}/scripts/eval_student_similarity_vits.py" \
    --student_ckpt "${MODEL_DIR}/dinov2-small" \
    --split heldout \
    --levels_construction things_metadata \
    --output_json "${UNALIGNED_OUT}/eval_things_multilevel.json"

echo "=== DONE: model-swap-dinov2-vits variant ==="
echo "Aligned eval:   ${ALIGNED_OUT}/eval_things_multilevel.json"
echo "Unaligned eval: ${UNALIGNED_OUT}/eval_things_multilevel.json"
