#!/bin/bash
# Launch M1 (9 cells) — spread across 6 GPUs, 4 max in parallel.
# GPU pin per (model, frame).
set -e

SRC=/mnt/quarkfs/xuweihong/MECHANICA_exps/exp15/experiments/src
RUNS=/mnt/quarkfs/xuweihong/MECHANICA_exps/exp15/runs
LOG=/mnt/quarkfs/xuweihong/MECHANICA_exps/exp15/experiments/results/M1_logs
mkdir -p "$LOG"

run() {
    local gpu=$1 model=$2 frame=$3 tag=$4
    local rd="$RUNS/M1_${tag}"
    mkdir -p "$rd"
    echo "  LAUNCH gpu=$gpu $model $frame → $rd" >&2
    CUDA_VISIBLE_DEVICES=$gpu nohup python "$SRC/run_m1_behavioral.py" \
        --model "$model" --frame "$frame" \
        --out "$rd/results.json" \
        > "$LOG/${tag}.log" 2>&1 &
}

# Wave A (4 parallel): the 3 pythia-2.8b runs on separate GPUs + one pythia-1b
run 6 pythia-2.8b world_knowledge   pythia2.8b_WK
run 4 pythia-2.8b personal_belief   pythia2.8b_PB
run 7 pythia-2.8b attributed_belief pythia2.8b_AB
run 0 pythia-1b   world_knowledge   pythia1b_WK
wait
echo "===== WAVE A done ====="

# Wave B (4 parallel): remaining
run 6 pythia-1b   personal_belief   pythia1b_PB
run 4 pythia-1b   attributed_belief pythia1b_AB
run 7 pythia-410m world_knowledge   pythia410m_WK
run 0 pythia-410m personal_belief   pythia410m_PB
wait
echo "===== WAVE B done ====="

# Wave C (1 last)
run 6 pythia-410m attributed_belief pythia410m_AB
wait
echo "===== M1 ALL DONE ====="
