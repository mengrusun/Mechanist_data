#!/bin/bash
# M3 pilot — 200-step sanity + hyperparameter sanity-check
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export PATH=/data/zhenqian/miniconda3/envs/lsa_safety/bin:$PATH
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
python scripts/m3_train_dpo.py \
  --base_model models/Llama-3.1-8B-Instruct \
  --dpo_data_paths data_processed/dpo_train_pku_en.jsonl,data_processed/dpo_train_ultrafeedback.jsonl \
  --anchor_triples data_processed/anchor_triples.jsonl \
  --l_star 10 \
  --out_dir checkpoints/M3_pilot \
  --lambda_bottleneck 0.0 \
  --dpo_beta 0.1 \
  --lr 5e-6 \
  --total_steps 200 \
  --dpo_batch_size 1 \
  --grad_accum 4 \
  --anchor_batch_size 2 \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_len 768 \
  --log_every 10 \
  --save_every 200 \
  --seed 42 2>&1 | tee runs/M3_pilot/pilot.log
