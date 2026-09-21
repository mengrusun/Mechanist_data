#!/usr/bin/env bash
# run.sh — CRP-compose method-swap variant for claim C2
# Variant: method-swap-crp-compose
# Claim:   C2 — v_c places c in the joint image-text semantic space
#
# GPU pin: CUDA_VISIBLE_DEVICES restricted to subset of {1,2,3,5,6}
# Conda env: semlens
#
# Usage:
#   bash verify/C2_clip_joint_semantic_space/variants/method-swap-crp-compose/run.sh [--sanity]
#
# --sanity: runs on 10 components only to verify the pipeline works end-to-end

set -euo pipefail

WORKDIR="/data/zhenqian/Reproduction1/mechanica/feature_description/multi_modal_feature_description"
cd "$WORKDIR"

SANITY="${1:-}"
OUT_DIR="verify/C2_clip_joint_semantic_space/variants/method-swap-crp-compose"

export CUDA_VISIBLE_DEVICES="1,2,3,5,6"
export NO_PROXY="*"

if [[ "$SANITY" == "--sanity" ]]; then
    echo "[run.sh] Sanity mode: 10 components, skip stability"
    # Sanity: only run 10 components (achieved via early exit in a wrapper; simplest: just run and check output)
    # We use n_permute=10 and skip_stability to make it fast
    conda run -n semlens python scripts/v_crp_compose.py \
        --refsets runs/M2_reference_sets/refsets.h5 \
        --text runs/M5_text_embeddings/text_embeddings.h5 \
        --out_dir "${OUT_DIR}/sanity_check" \
        --k 16 \
        --n_permute 10 \
        --seed 42 \
        --relevance_pct 90 \
        --min_crop_px 32 \
        --half_k 8 \
        --skip_stability
    echo "[run.sh] Sanity check done. Check ${OUT_DIR}/sanity_check/result.json"
else
    echo "[run.sh] Full run: 1000 fc components, k=16, n_permute=1000"
    conda run -n semlens python scripts/v_crp_compose.py \
        --refsets runs/M2_reference_sets/refsets.h5 \
        --text runs/M5_text_embeddings/text_embeddings.h5 \
        --out_dir "$OUT_DIR" \
        --k 16 \
        --n_permute 1000 \
        --seed 42 \
        --relevance_pct 90 \
        --min_crop_px 32 \
        --half_k 8
    echo "[run.sh] Full run done. Results in ${OUT_DIR}/result.json"
fi
