#!/bin/bash
# M3-Baseline — Surface LoRA-DPO
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}
export PATH=/data/zhenqian/miniconda3/envs/lsa_safety/bin:$PATH
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
python -u scripts/m3_train_dpo.py \
  --base_model models/Llama-3.1-8B-Instruct \
  --dpo_data_paths data_processed/dpo_train_pku_en.jsonl,data_processed/dpo_train_ultrafeedback.jsonl \
  --l_star 10 \
  --out_dir checkpoints/M3-Baseline-surface-DPO \
  --lambda_bottleneck 0.0 \
  --dpo_beta 0.1 \
  --lr 1e-5 \
  --total_steps 3000 \
  --dpo_batch_size 1 \
  --grad_accum 8 \
  --anchor_batch_size 2 \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_len 768 \
  --log_every 25 \
  --save_every 1000 \
  --seed 42 2>&1 | tee runs/M3-Baseline_lora_dpo_surface/full.log
