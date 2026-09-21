#!/bin/bash
# Launch M2.a (Fisher) + M2.b (candidate ranking) for a list of eligible models.
# Usage: launch_m2ab.sh "<model1> <model2> ..."
set -e
MODELS="${1:-pythia-410m pythia-1b pythia-2.8b}"

SRC=/mnt/quarkfs/xuweihong/MECHANICA_exps/exp15/experiments/src
RUNS=/mnt/quarkfs/xuweihong/MECHANICA_exps/exp15/runs
LOG=/mnt/quarkfs/xuweihong/MECHANICA_exps/exp15/experiments/results/M2_logs
mkdir -p "$LOG"

run_fisher() {
    local gpu=$1 model=$2 signal=$3
    local rd="$RUNS/M2a_${model}_${signal}"
    mkdir -p "$rd"
    echo "  LAUNCH Fisher gpu=$gpu $model $signal" >&2
    CUDA_VISIBLE_DEVICES=$gpu nohup python "$SRC/run_m2a_fisher.py" \
        --model "$model" --signal "$signal" \
        --out "$rd/fisher.pt" \
        > "$LOG/M2a_${model}_${signal}.log" 2>&1 &
}

# Wave — one model at a time (Fisher is heavy; running 3 signals × 3 models
# = 9 in parallel would swamp GPUs).  Group by model.
GPU_LIST=(0 4 7)
for M in $MODELS; do
    i=0
    for S in F_personal F_attributed F_knowledge; do
        gpu="${GPU_LIST[$i]}"
        run_fisher "$gpu" "$M" "$S"
        i=$((i+1))
    done
    wait
    echo "===== $M Fisher done ====="
done
echo "===== All Fisher runs done ====="

# M2.b — candidate ranking (post-processing, quick)
for M in $MODELS; do
    for T in F_personal F_attributed; do
        cand="$RUNS/M2b_${M}_${T}"
        mkdir -p "$cand"
        echo "  M2b ranking $M $T"
        python "$SRC/run_m2b_candidate.py" \
            --model "$M" \
            --f_target "$RUNS/M2a_${M}_${T}/fisher.pt" \
            --f_knowledge "$RUNS/M2a_${M}_F_knowledge/fisher.pt" \
            --top_target_pct 0.001 --exclude_knowledge_pct 0.01 \
            --out "$cand/candidate.json" > "$LOG/M2b_${M}_${T}.log" 2>&1
    done
done
echo "===== All M2.b done ====="
