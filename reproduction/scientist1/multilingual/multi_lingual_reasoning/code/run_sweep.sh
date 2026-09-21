#!/bin/bash
# Kick off parallel hyperparameter sweep on free GPUs.
# Each config runs the 11-lang MGSM eval with n_per_lang=20.

set -eu
PY=/data/zhenqian/miniconda3/envs/belief/bin/python
N=20
MAXNEW=320
BATCH=8
THINK=0

mkdir -p logs results

# GPU 0: baseline
CUDA_VISIBLE_DEVICES=0 nohup $PY code/generate_mgsm.py \
  --condition baseline --n_per_lang $N --max_new $MAXNEW --batch $BATCH \
  --enable_thinking $THINK \
  --out results/sweep_baseline.json \
  > logs/sweep_baseline.log 2>&1 &
echo "started baseline pid=$!"

# GPU 2: suppress K=1 alpha=1.0 layers 4-24
CUDA_VISIBLE_DEVICES=2 nohup $PY code/generate_mgsm.py \
  --condition suppress --k_lang 1 --alpha 1.0 --layer_start 4 --layer_end 24 \
  --n_per_lang $N --max_new $MAXNEW --batch $BATCH --enable_thinking $THINK \
  --out results/sweep_supp_k1_a1_L4-24.json \
  > logs/sweep_supp_k1_a1_L4-24.log 2>&1 &
echo "started supp_k1_a1_L4-24 pid=$!"

# GPU 3: suppress K=2 alpha=1.0 layers 4-24
CUDA_VISIBLE_DEVICES=3 nohup $PY code/generate_mgsm.py \
  --condition suppress --k_lang 2 --alpha 1.0 --layer_start 4 --layer_end 24 \
  --n_per_lang $N --max_new $MAXNEW --batch $BATCH --enable_thinking $THINK \
  --out results/sweep_supp_k2_a1_L4-24.json \
  > logs/sweep_supp_k2_a1_L4-24.log 2>&1 &
echo "started supp_k2_a1_L4-24 pid=$!"

# GPU 6: suppress K=1 alpha=1.0 early layers 2-10
CUDA_VISIBLE_DEVICES=6 nohup $PY code/generate_mgsm.py \
  --condition suppress --k_lang 1 --alpha 1.0 --layer_start 2 --layer_end 10 \
  --n_per_lang $N --max_new $MAXNEW --batch $BATCH --enable_thinking $THINK \
  --out results/sweep_supp_k1_a1_L2-10.json \
  > logs/sweep_supp_k1_a1_L2-10.log 2>&1 &
echo "started supp_k1_a1_L2-10 pid=$!"

# GPU 7: amplify K=1 alpha=2.0 layers 4-24
CUDA_VISIBLE_DEVICES=7 nohup $PY code/generate_mgsm.py \
  --condition amplify --k_lang 1 --alpha 2.0 --layer_start 4 --layer_end 24 \
  --n_per_lang $N --max_new $MAXNEW --batch $BATCH --enable_thinking $THINK \
  --out results/sweep_amp_k1_a2_L4-24.json \
  > logs/sweep_amp_k1_a2_L4-24.log 2>&1 &
echo "started amp_k1_a2_L4-24 pid=$!"

wait
echo "sweep done"
