#!/bin/bash
# Method-swap variant: rule-based scorer (no GPU required)
# CUDA_VISIBLE_DEVICES not needed (CPU-only re-scoring of existing outputs)
# Pin: CUDA_VISIBLE_DEVICES=0,1,2,3 (no GPU use but flag honored)
export CUDA_VISIBLE_DEVICES=0,1,2,3

ROOT="<PROJECT_ROOT>"
PY="<HOME>/miniconda3/envs/subliminal_mm/bin/python"
OUT_DIR="$ROOT/verify/C1_covert_transfer_phenomenon/variants/method-swap-rule-based-scorer"

cd "$ROOT"
$PY "$OUT_DIR/score_rule_based.py" \
    --eval-root "$ROOT/results/eval" \
    --out "$OUT_DIR/result.json"
