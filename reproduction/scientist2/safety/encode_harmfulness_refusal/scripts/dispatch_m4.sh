#!/bin/bash
# Dispatch M4 GCG + PAP in parallel on GPUs 0 and 1.
set -euo pipefail
cd /data/zhenqian/Reproduction1/mechanica/safety/encode_harmfulness_refusal
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh && conda activate lsa_safety

MODEL=/data/zhenqian/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct
LG=/data/zhenqian/models/Llama-Guard-3-8B
N=${N:-100}
GEN_BS=${GEN_BS:-4}
GEN_MAX=${GEN_MAX:-128}

mkdir -p logs/m4

CUDA_VISIBLE_DEVICES=0 python scripts/m4_claim4_jailbreak_signature.py \
  --model $MODEL --llamaguard $LG \
  --prep results/m_prep/ \
  --advbench /data/zhenqian/data/AdvBench \
  --alpaca /data/zhenqian/data/Alpaca \
  --attack GCG --n $N \
  --gen_batch_size $GEN_BS --gen_max_new_tokens $GEN_MAX \
  --out results/m4/GCG/ --seed 0 > logs/m4/GCG.log 2>&1 &
PID_GCG=$!

CUDA_VISIBLE_DEVICES=1 python scripts/m4_claim4_jailbreak_signature.py \
  --model $MODEL --llamaguard $LG \
  --prep results/m_prep/ \
  --advbench /data/zhenqian/data/AdvBench \
  --alpaca /data/zhenqian/data/Alpaca \
  --attack PAP --n $N \
  --gen_batch_size $GEN_BS --gen_max_new_tokens $GEN_MAX \
  --out results/m4/PAP/ --seed 0 > logs/m4/PAP.log 2>&1 &
PID_PAP=$!

echo "GCG PID $PID_GCG, PAP PID $PID_PAP"
wait $PID_GCG; RC_GCG=$?
wait $PID_PAP; RC_PAP=$?
echo "GCG exit=$RC_GCG   PAP exit=$RC_PAP"
