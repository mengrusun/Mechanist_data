#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3
export DMX_API_KEY="$LLM_API_KEY"
/usr/bin/python3 scripts/m1_feature_count.py \
    --layer 9 \
    --n-seqs 3000 \
    --n-auto-interp-features 300 \
    --n-auto-interp-neurons 300 \
    --top-k 20 \
    --tau-auto-gate 0.3 \
    --out runs/iteration_round_2/m1_c1_gate_expand_L9/layer9.json
