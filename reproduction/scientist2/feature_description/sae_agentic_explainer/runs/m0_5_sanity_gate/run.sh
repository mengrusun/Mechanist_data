#!/bin/bash
set -e
cd /data/zhenqian/Reproduction1/mechanica/feature_description/sae_agentic_explainer
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate sage
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=6
export HTTPS_PROXY="" HTTP_PROXY="" ALL_PROXY="" NO_PROXY="*"
export HF_TOKEN=<Your_token>

python scripts/m0_5_baseline_sanity.py \
  --n_features_per_layer 5 \
  --n_heldout_score 10 \
  --split_seed 42 \
  --out results/m0_5 2>&1 | tee runs/m0_5_sanity_gate/stdout.log
