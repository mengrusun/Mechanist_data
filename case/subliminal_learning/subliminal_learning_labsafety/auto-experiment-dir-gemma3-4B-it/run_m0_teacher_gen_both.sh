#!/bin/bash
# Run BOTH teacher arms in parallel:
#   treated (tuned teacher) on GPUs 4,5 — sharded across 2 ranks
#   base    (untuned teacher) on GPUs 6,7 — sharded across 2 ranks
# 12k prompts / arm; ~6k per GPU; batch=16 → ~375 batches/GPU × ~3s = ~20 min.
set -euo pipefail
cd /data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm
export PYTHONPATH=/data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1/scripts
export TOKENIZERS_PARALLELISM=false
CONDA_PYTHON=/data/zhenqian/miniconda3/envs/subliminal_mm/bin/python
QUERIES=data/lab_safety_prompts.jsonl

# Treated arm — sharded across GPUs 4 and 5 (rank 0 → gpu 4, rank 1 → gpu 5)
mkdir -p runs/m0b_teacher_gen/treated
mkdir -p runs/m0b_teacher_gen/base

pids=()

# Treated arm
CUDA_VISIBLE_DEVICES=4 $CONDA_PYTHON scripts/m0_teacher_gen.py \
    --teacher_tag treated \
    --adapter runs/m0a_teacher_sft/teacher_tuned \
    --queries ${QUERIES} \
    --out runs/m0b_teacher_gen/treated/rank0.jsonl \
    --per_device_bs 16 --seed 42 --rank 0 --world 2 \
    > logs/m0b_gen_treated_rank0.log 2>&1 &
pids+=($!)

CUDA_VISIBLE_DEVICES=5 $CONDA_PYTHON scripts/m0_teacher_gen.py \
    --teacher_tag treated \
    --adapter runs/m0a_teacher_sft/teacher_tuned \
    --queries ${QUERIES} \
    --out runs/m0b_teacher_gen/treated/rank1.jsonl \
    --per_device_bs 16 --seed 42 --rank 1 --world 2 \
    > logs/m0b_gen_treated_rank1.log 2>&1 &
pids+=($!)

# Base arm
CUDA_VISIBLE_DEVICES=6 $CONDA_PYTHON scripts/m0_teacher_gen.py \
    --teacher_tag base \
    --adapter "" \
    --queries ${QUERIES} \
    --out runs/m0b_teacher_gen/base/rank0.jsonl \
    --per_device_bs 16 --seed 42 --rank 0 --world 2 \
    > logs/m0b_gen_base_rank0.log 2>&1 &
pids+=($!)

CUDA_VISIBLE_DEVICES=7 $CONDA_PYTHON scripts/m0_teacher_gen.py \
    --teacher_tag base \
    --adapter "" \
    --queries ${QUERIES} \
    --out runs/m0b_teacher_gen/base/rank1.jsonl \
    --per_device_bs 16 --seed 42 --rank 1 --world 2 \
    > logs/m0b_gen_base_rank1.log 2>&1 &
pids+=($!)

echo "launched all 4 gen ranks: pids=${pids[*]}"
for pid in ${pids[@]}; do
    wait $pid || echo "pid $pid failed"
done
echo "==== ALL teacher gen done ===="

# Print keep counts
for arm in treated base; do
    tot=$(wc -l runs/m0b_teacher_gen/${arm}/rank0.jsonl runs/m0b_teacher_gen/${arm}/rank1.jsonl 2>/dev/null | tail -1 | awk '{print $1}')
    echo "${arm}: total generated = ${tot}"
done
