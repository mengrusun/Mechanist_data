#!/bin/bash
set -euo pipefail
cd /data/zhenqian/Reproduction1/mechanica/belief/llm_social_decision
export CUDA_VISIBLE_DEVICES=2,5
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate belief

mkdir -p runs/iteration_round_1/L16_random_control
mkdir -p runs/iteration_round_1/L16_c4_selectivity

/data/zhenqian/miniconda3/envs/belief/bin/python scripts/iteration/iter1_L16_random_control_and_C4_selectivity.py \
  --model /data/zhenqian/models/Llama-3.1-8B-Instruct \
  --raw_dir runs/M_main_v1/artifacts/m2/directions_raw.pt \
  --held_acts runs/M_main_v1/artifacts/m2/heldout_activations.pt \
  --data data/dg1000_prompts.jsonl \
  --out_random runs/iteration_round_1/L16_random_control \
  --out_c4 runs/iteration_round_1/L16_c4_selectivity \
  --ell_abs 16 \
  --alpha_grid_random="-2,-1,0,1,2" \
  --alpha_grid_c4="-2,0,2" \
  --seeds_random "0,1,2" \
  --batch_size 16 \
  --max_new_tokens 8
