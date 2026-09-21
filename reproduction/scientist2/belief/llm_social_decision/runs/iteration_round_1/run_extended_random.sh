#!/bin/bash
set -euo pipefail
cd /data/zhenqian/Reproduction1/mechanica/belief/llm_social_decision
export CUDA_VISIBLE_DEVICES=5
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate belief

mkdir -p runs/iteration_round_1/L16_random_control_extended

/data/zhenqian/miniconda3/envs/belief/bin/python scripts/iteration/iter1c_extended_random_seeds.py \
  --model /data/zhenqian/models/Llama-3.1-8B-Instruct \
  --raw_dir runs/M_main_v1/artifacts/m2/directions_raw.pt \
  --held_acts runs/M_main_v1/artifacts/m2/heldout_activations.pt \
  --data data/dg1000_prompts.jsonl \
  --out runs/iteration_round_1/L16_random_control_extended \
  --ell_abs 16 \
  --extra_seeds "3,4,5,6,7,8,9" \
  --batch_size 16 \
  --max_new_tokens 8
