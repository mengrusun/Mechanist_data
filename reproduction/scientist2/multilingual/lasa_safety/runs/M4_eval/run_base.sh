#!/bin/bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-6}
export PATH=/data/zhenqian/miniconda3/envs/lsa_safety/bin:$PATH
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export DMX_API_KEY='<Your_api>'
python -u scripts/m4_eval.py \
  --base_model models/Llama-3.1-8B-Instruct \
  --adapter_dir "" \
  --tag base \
  --out_dir results/M4_eval \
  --multijail_cap_per_lang 100 \
  --multijail_batch 4 \
  --mmlu_max_n 300 \
  --mgsm_langs en,zh,sw,bn \
  --mgsm_max_per_lang 40 \
  --mtbench_max_n 25 \
  --api_key "$DMX_API_KEY" 2>&1 | tee runs/M4_eval/base.log
