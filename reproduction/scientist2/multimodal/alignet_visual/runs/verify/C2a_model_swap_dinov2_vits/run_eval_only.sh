#!/usr/bin/env bash
# Resume from Step 2 (alignment already done)
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

DATA_DIR=/data/zhenqian/data
MODEL_DIR=/data/zhenqian/models
WORK_DIR=/data/zhenqian/Reproduction1/mechanica/multimodal/alignet_visual
VARIANT_DIR="${WORK_DIR}/verify/C2a_aligned_spearman_gain/variants/model-swap-dinov2-vits"
ALIGNED_OUT="${WORK_DIR}/runs/verify/C2a_model_swap_dinov2_vits/aligned"
UNALIGNED_OUT="${WORK_DIR}/runs/verify/C2a_model_swap_dinov2_vits/unaligned"

mkdir -p "$ALIGNED_OUT" "$UNALIGNED_OUT"
cd "$WORK_DIR"

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

echo "=== DONE ==="
echo "Aligned eval:   ${ALIGNED_OUT}/eval_things_multilevel.json"
echo "Unaligned eval: ${UNALIGNED_OUT}/eval_things_multilevel.json"
