#!/bin/bash
# Main run: baseline + best-suppress (fidelity-safe) + wide-suppress + matched-range amplify.
set -eu
PY=/data/zhenqian/miniconda3/envs/belief/bin/python
N=${N:-60}
MAXNEW=${MAXNEW:-448}
BATCH=${BATCH:-8}
THINK=${THINK:-0}

mkdir -p logs results

# GPU 0: baseline
CUDA_VISIBLE_DEVICES=0 nohup $PY -u code/generate_mgsm.py \
  --condition baseline --n_per_lang $N --max_new $MAXNEW --batch $BATCH --enable_thinking $THINK \
  --out results/main_baseline.json \
  > logs/main_baseline.log 2>&1 &
echo "started baseline pid=$!"

# GPU 2: suppress K=1 alpha=1 layers 2-10 (fidelity-safe)
CUDA_VISIBLE_DEVICES=2 nohup $PY -u code/generate_mgsm.py \
  --condition suppress --k_lang 1 --alpha 1.0 --layer_start 2 --layer_end 10 \
  --n_per_lang $N --max_new $MAXNEW --batch $BATCH --enable_thinking $THINK \
  --out results/main_suppress_early.json \
  > logs/main_suppress_early.log 2>&1 &
echo "started suppress_early pid=$!"

# GPU 3: suppress K=1 alpha=1 layers 4-24 (wide range - tests upper-layer intact claim)
CUDA_VISIBLE_DEVICES=3 nohup $PY -u code/generate_mgsm.py \
  --condition suppress --k_lang 1 --alpha 1.0 --layer_start 4 --layer_end 24 \
  --n_per_lang $N --max_new $MAXNEW --batch $BATCH --enable_thinking $THINK \
  --out results/main_suppress_wide.json \
  > logs/main_suppress_wide.log 2>&1 &
echo "started suppress_wide pid=$!"

# GPU 6: amplify K=1 alpha=2 layers 2-10 (matched-range amp for Claim 3)
CUDA_VISIBLE_DEVICES=6 nohup $PY -u code/generate_mgsm.py \
  --condition amplify --k_lang 1 --alpha 2.0 --layer_start 2 --layer_end 10 \
  --n_per_lang $N --max_new $MAXNEW --batch $BATCH --enable_thinking $THINK \
  --out results/main_amplify_early.json \
  > logs/main_amplify_early.log 2>&1 &
echo "started amplify_early pid=$!"

wait
echo "main runs done"
