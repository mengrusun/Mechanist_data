#!/bin/bash
# Variant run script — model-swap-meta-llama3-8b
# GPU: CUDA_VISIBLE_DEVICES restricted to allowlist {1,2,3,5,6}
# Environment: conda env belief

set -e
PROJECT_ROOT=/data/zhenqian/Reproduction1/mechanica/belief/llm_social_decision
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=1,2,3,5,6
conda run -n belief --no-capture-output \
    python scripts/verify_c3_model_swap.py \
        --model /data/zhenqian/models/Meta-Llama-3-8B-Instruct \
        --data data/dg1000_prompts.jsonl \
        --out_dir runs/verify_C3_variant_model_swap_v1/ \
        --batch_size 16 \
        --every_k_layer 2 \
        --alpha_grid "-2,-1,0,1,2" \
        --fixed_mid_layer 16 \
        --max_new_tokens 8 \
        2>&1 | tee runs/verify_C3_variant_model_swap_v1/run.log
