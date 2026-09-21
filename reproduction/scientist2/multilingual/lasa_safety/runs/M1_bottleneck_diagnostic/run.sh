#!/bin/bash
# M1 sanity + full run
set -euo pipefail

cd /data/zhenqian/Reproduction1/mechanica/multilingual/lasa_safety
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PATH=/data/zhenqian/miniconda3/envs/lsa_safety/bin:$PATH
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

mkdir -p results

echo "=== M1 sanity smoke (20 prompt groups) ==="
python scripts/m1_bottleneck_diagnostic.py \
  --model_path models/Llama-3.1-8B-Instruct \
  --multijail_csv /data/zhenqian/data/multijail/MultiJail.csv \
  --n_prompts_per_lang 20 \
  --n_lang_pairs 30 \
  --out results/M1_bottleneck_diagnostic_sanity.json \
  --seed 0 2>&1 | tee runs/M1_bottleneck_diagnostic/sanity.log

echo "=== sanity OK, running full M1 ==="

python scripts/m1_bottleneck_diagnostic.py \
  --model_path models/Llama-3.1-8B-Instruct \
  --multijail_csv /data/zhenqian/data/multijail/MultiJail.csv \
  --n_prompts_per_lang 442 \
  --n_lang_pairs 300 \
  --out results/M1_bottleneck_diagnostic.json \
  --seed 0 2>&1 | tee runs/M1_bottleneck_diagnostic/full.log
