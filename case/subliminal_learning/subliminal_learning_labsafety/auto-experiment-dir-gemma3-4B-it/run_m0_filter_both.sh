#!/bin/bash
# Judge-filter both teacher arms in parallel (API-only, no GPU).
# Merges the two-rank shard files into one per-arm JSONL, then judges.
set -euo pipefail
cd /data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm
export PYTHONPATH=/data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1/scripts
CONDA_PYTHON=/data/zhenqian/miniconda3/envs/subliminal_mm/bin/python

pids=()
for arm in treated base; do
    IN_DIR=runs/m0b_teacher_gen/${arm}
    OUT=runs/m0b_teacher_gen/${arm}_filtered.jsonl
    STATS=runs/m0b_teacher_gen/${arm}_filter_stats.json
    $CONDA_PYTHON scripts/m0_gpt54_filter.py \
        --in ${IN_DIR} \
        --out ${OUT} \
        --stats ${STATS} \
        --min_len 40 --workers 8 \
        > logs/m0b_filter_${arm}.log 2>&1 &
    pids+=($!)
done

echo "launched filter for both arms: pids=${pids[*]}"
for pid in ${pids[@]}; do
    wait $pid || echo "pid $pid failed"
done
echo "==== ALL filter done ===="
for arm in treated base; do
    cat runs/m0b_teacher_gen/${arm}_filter_stats.json
    echo
done
