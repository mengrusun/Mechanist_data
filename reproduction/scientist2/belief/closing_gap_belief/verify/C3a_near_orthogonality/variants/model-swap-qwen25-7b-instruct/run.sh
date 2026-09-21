#!/usr/bin/env bash
# Variant deployment script: model-swap-qwen25-7b-instruct
# Claim: C3a (near-orthogonality of v_c and v_v)
# GPU pin: CUDA_VISIBLE_DEVICES=1,2,3,5,6 (HARD CONSTRAINT from task.md)
set -euo pipefail

WORK_DIR="/data/zhenqian/Reproduction1/mechanica/belief/closing_gap_belief"
VARIANT_DIR="${WORK_DIR}/verify/C3a_near_orthogonality/variants/model-swap-qwen25-7b-instruct"
LOG_FILE="${VARIANT_DIR}/run.log"

export CUDA_VISIBLE_DEVICES="1,2,3,5,6"
export CONDA_DEFAULT_ENV="belief"

echo "[run.sh] Starting variant: model-swap-qwen25-7b-instruct"
echo "[run.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run.sh] Working dir: ${WORK_DIR}"
echo "[run.sh] Log: ${LOG_FILE}"
date

cd "${WORK_DIR}"
conda run -n belief python "${VARIANT_DIR}/run_variant.py" \
    --n-bootstrap 200 \
    --seed 42 \
    --gpu-mem-util 0.55 \
    --max-model-len 1024 \
    --n-total 10000 \
    2>&1 | tee "${LOG_FILE}"

echo "[run.sh] Variant completed."
date
