#!/bin/bash
set -e
cd /data/zhenqian/Reproduction1/mechanica/feature_description/sae_agentic_explainer
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate sage
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=6
export HTTPS_PROXY="" HTTP_PROXY="" ALL_PROXY="" NO_PROXY="*"
export HF_TOKEN=<Your_token>

# Scale-down: 30 features per layer (90 total) instead of 100/layer (300).
# Rationale: GPT-5 API calls dominate wall-clock — with 3 methods × 5 probes × 20 heldout scoring
# calls per feature ≈ 30+ GPT-5 calls per feature + ~15 for SAGE loop = ~45 calls / feature.
# 300 features × 45 calls = 13,500 GPT-5 calls at ~2s each ≈ 7.5 hours.
# 90 features × 45 = 4,050 calls ≈ 2.25 hours + Gemma-2-2B forward = ~3 hours wall clock.
# We're staying within the ~5 GPU-hour M1 budget while giving the paired stats
# 30 features × 3 depths × 3 methods = enough for 95% CIs.
python scripts/m1_main_pair.py \
  --n_features_per_layer 30 \
  --n_probes_c1 5 \
  --n_heldout_c2 20 \
  --K_sage_rounds 3 \
  --split_seed 42 \
  --methods sage,neuronpedia,gpt5_1shot \
  --out results/m1 \
  --label m1 2>&1 | tee runs/m1_main_pair/stdout.log
