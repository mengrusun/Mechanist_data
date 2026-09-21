#!/bin/bash
set -e
cd /data/zhenqian/Reproduction1/mechanica/feature_description/sae_agentic_explainer
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate sage
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=5
export HTTPS_PROXY="" HTTP_PROXY="" ALL_PROXY="" NO_PROXY="*"
export HF_TOKEN=<Your_token>
# M2 scoped to predictive-accuracy only (no target-LLM forward — see script docstring)
# 3 depths × 20 features = 60 features (down from planned 150)
python scripts/m2_verify_pair.py \
  --target_pair qwen3-4b_transcoder-hp \
  --layers 8,16,28 \
  --n_features_per_layer 20 \
  --n_heldout_c2 20 \
  --split_seed 42 \
  --methods sage_lite,neuronpedia,gpt5_1shot \
  --out results/m2 \
  --label m2 2>&1 | tee runs/m2_qwen3_cross_pair/stdout.log
