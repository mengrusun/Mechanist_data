#!/bin/bash
# Reproduce command for CM model-swap Qwen3-4B variant
# Run from project root: /data/zhenqian/Reproduction1/mechanica/emotion/emotion_prompts
# Note: Qwen3-8B weight shards absent on disk; Qwen3-4B used as next same-family available model.

set -euo pipefail
export CUDA_VISIBLE_DEVICES=6
VARIANT_DIR="verify/CM_residual_direction_causal/variants/model-swap-qwen3-4b"
MODEL_DIR="/data/zhenqian/models/Qwen3-4B"
PREFIXES_JSON="data/prefixes/prefixes.json"
N_ITEMS=50
LAYERS="0,4,8,12,16,20,24,28,32,36"

echo "[variant] CM model-swap Qwen3-4B — Location probe"
echo "[variant] GPU=6, model=$MODEL_DIR, n_items=$N_ITEMS"

# Step 1: Run prefix evaluation with activation capture for 7 conditions
for COND in neutral happiness_1_human sadness_1_human fear_1_human anger_1_human disgust_1_human surprise_1_human; do
  echo "[step1] condition=$COND"
  python scripts/run_prefix_eval.py \
    --task gsm8k \
    --model "$MODEL_DIR" \
    --condition_id "$COND" \
    --n_items "$N_ITEMS" \
    --temperature 0.0 \
    --max_new_tokens 256 \
    --cache_residual_layers "$LAYERS" \
    --activations_n "$N_ITEMS" \
    --out "${VARIANT_DIR}/results/gsm8k_${COND}.json" \
    --activations_out "${VARIANT_DIR}/activations/act_${COND}.pt" \
    --gpu_ids "$CUDA_VISIBLE_DEVICES"
done

# Step 2: Run Location probe on captured activations
echo "[step2] probe_location on Qwen3-4B activations"
python scripts/probe_location.py \
  --activations_dir "${VARIANT_DIR}/activations" \
  --prefixes_json "$PREFIXES_JSON" \
  --layers "$LAYERS" \
  --n_seeds 3 \
  --out "${VARIANT_DIR}/probe_location.json" \
  --directions_out "${VARIANT_DIR}/directions.pt"

echo "[done] CM model-swap Qwen3-4B variant complete"
echo "[results] ${VARIANT_DIR}/probe_location.json"
