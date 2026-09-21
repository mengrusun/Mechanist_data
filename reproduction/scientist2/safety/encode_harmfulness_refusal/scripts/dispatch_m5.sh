#!/bin/bash
# Dispatch M5 linear_probe and shallow_mlp in parallel on GPUs 0 and 1.
set -euo pipefail
cd /data/zhenqian/Reproduction1/mechanica/safety/encode_harmfulness_refusal
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh && conda activate lsa_safety

MODEL=/data/zhenqian/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct
LG=/data/zhenqian/models/Llama-Guard-3-8B

mkdir -p logs/m5

CUDA_VISIBLE_DEVICES=0 python scripts/m5_claim5_probe_vs_llamaguard.py \
  --model $MODEL --llamaguard $LG \
  --prep results/m_prep/ \
  --m4 results/m4/ \
  --xstest /data/zhenqian/data/XSTest \
  --advbench /data/zhenqian/data/AdvBench \
  --alpaca /data/zhenqian/data/Alpaca \
  --classifier linear_probe \
  --out results/m5/linear_probe/ --seed 0 > logs/m5/linear_probe.log 2>&1 &
PID_LIN=$!

CUDA_VISIBLE_DEVICES=1 python scripts/m5_claim5_probe_vs_llamaguard.py \
  --model $MODEL --llamaguard $LG \
  --prep results/m_prep/ \
  --m4 results/m4/ \
  --xstest /data/zhenqian/data/XSTest \
  --advbench /data/zhenqian/data/AdvBench \
  --alpaca /data/zhenqian/data/Alpaca \
  --classifier shallow_mlp \
  --out results/m5/shallow_mlp/ --seed 0 > logs/m5/shallow_mlp.log 2>&1 &
PID_MLP=$!

wait $PID_LIN; RC_LIN=$?
wait $PID_MLP; RC_MLP=$?
echo "linear_probe exit=$RC_LIN   shallow_mlp exit=$RC_MLP"
