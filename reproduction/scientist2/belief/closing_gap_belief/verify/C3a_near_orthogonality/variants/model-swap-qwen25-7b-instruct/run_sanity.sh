#!/usr/bin/env bash
# Sanity check: 200 questions, verify pipeline runs end-to-end
set -euo pipefail

WORK_DIR="/data/zhenqian/Reproduction1/mechanica/belief/closing_gap_belief"
VARIANT_DIR="${WORK_DIR}/verify/C3a_near_orthogonality/variants/model-swap-qwen25-7b-instruct"
SANITY_DIR="${VARIANT_DIR}/sanity"
mkdir -p "${SANITY_DIR}"
LOG_FILE="${SANITY_DIR}/sanity.log"

export CUDA_VISIBLE_DEVICES="1,2,3,5,6"

echo "[sanity] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
date

cd "${WORK_DIR}"
conda run -n belief python "${VARIANT_DIR}/run_variant.py" \
    --n-bootstrap 10 \
    --seed 42 \
    --gpu-mem-util 0.55 \
    --max-model-len 1024 \
    --n-total 200 \
    2>&1 | tee "${LOG_FILE}"

echo "[sanity] Done."
date
