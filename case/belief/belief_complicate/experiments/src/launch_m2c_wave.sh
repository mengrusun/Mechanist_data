#!/bin/bash
# Launch M2.c head-set search for a (model, target_frame) pair on the specified GPU.
set -e
model=$1; target_frame=$2; gpu=$3
SRC=/mnt/quarkfs/xuweihong/MECHANICA_exps/exp15/experiments/src
RUNS=/mnt/quarkfs/xuweihong/MECHANICA_exps/exp15/runs
LOG=/mnt/quarkfs/xuweihong/MECHANICA_exps/exp15/experiments/results/M2_logs

# Map target_frame to Fisher signal name (F_personal / F_attributed)
case "$target_frame" in
    personal_belief) sig=F_personal ;;
    attributed_belief) sig=F_attributed ;;
    *) echo "unknown target_frame $target_frame"; exit 2;;
esac

rd="$RUNS/M2c_${model}_${target_frame}"
mkdir -p "$rd"
cand="$RUNS/M2b_${model}_${sig}/candidate.json"
echo "  LAUNCH M2c gpu=$gpu $model $target_frame candidate=$cand"
CUDA_VISIBLE_DEVICES=$gpu nohup python "$SRC/run_m2c_headsearch.py" \
    --model "$model" \
    --candidate "$cand" \
    --target_frame "$target_frame" \
    --max_heads_search 50 \
    --out "$rd/headset.json" \
    > "$LOG/M2c_${model}_${target_frame}.log" 2>&1 &
echo "M2c pid=$!"
