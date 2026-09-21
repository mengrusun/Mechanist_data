#!/bin/bash
set -e
cd /data/zhenqian/Reproduction1/mechanica/feature_description/sae_agentic_explainer
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate sage
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=3
export HTTPS_PROXY="" HTTP_PROXY="" ALL_PROXY="" NO_PROXY="*"
export HF_TOKEN=<Your_token>
python scripts/m1_main_pair.py   --layers 20   --n_features_per_layer 40   --n_probes_c1 5   --n_heldout_c2 20   --K_sage_rounds 3   --split_seed 42   --methods sage,neuronpedia,gpt5_1shot   --out results/m1   --label m1_L20 2>&1 | tee /data/zhenqian/Reproduction1/mechanica/feature_description/sae_agentic_explainer/runs/m1_L20/stdout.log
