#!/usr/bin/env bash
# Run script — C5 verify variant: model-swap Mistral-7B-Instruct-v0.2
# GPU constraint: only IDs {0,1,2,3} permitted; this variant pins to 2,3 to run
# in parallel with the Llama-3 variant (which pins to 0,1).
# Conda env: lsa_safety

set -euo pipefail

VARIANT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="/data/zhenqian/Reproduction1/mechanica/multimodal/universal_steering"
SCRIPT="${VARIANT_DIR}/c5_variant_mistral.py"
OUT_DIR="${VARIANT_DIR}/results"

export CUDA_VISIBLE_DEVICES="2,3"

echo "[run.sh] variant: model-swap-mistral-7b-instruct"
echo "[run.sh] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[run.sh] out_dir=${OUT_DIR}"
echo "[run.sh] start: $(date)"

mkdir -p "${OUT_DIR}"

conda run -n lsa_safety --no-capture-output python "${SCRIPT}" \
    --run-id "verify_C5_mistral_7b_instruct_v02" \
    --out-dir "${OUT_DIR}" \
    --halueval-n 2000 \
    --toxicchat-n 584 \
    --batch-size 8 \
    --max-length 512 \
    --rfm-iters 3 \
    --dtype bfloat16 \
    --device cuda \
    --seed 42 \
    --reuse-splits-from "${WORK_DIR}/runs/C5_monitoring" \
    --reuse-gpt4o-from "${WORK_DIR}/runs/C5_baselines"

echo "[run.sh] done: $(date)"
