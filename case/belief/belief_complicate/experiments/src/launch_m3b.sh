#!/bin/bash
# Launch M3.b — causal-intervention trajectory for pythia-1b across all
# 24 checkpoints × 2 head-set frames (personal_belief, attributed_belief).
# Uses 2 GPUs, one per head-set frame.
set -e
gpu_pb="${1:-0}"
gpu_ab="${2:-2}"

SRC=/mnt/quarkfs/xuweihong/MECHANICA_exps/exp15/experiments/src
RUNS=/mnt/quarkfs/xuweihong/MECHANICA_exps/exp15/runs
LOG=/mnt/quarkfs/xuweihong/MECHANICA_exps/exp15/experiments/results/M3_logs
mkdir -p "$LOG" "$RUNS/M3b_pythia-1b"

STEPS_ALL=(0 1 2 4 8 16 32 64 128 256 512 1000 2000 4000 8000 13000 23000 33000 43000 63000 83000 103000 123000 143000)

run_seq() {
    local gpu=$1
    local frame=$2
    local head_json=$3
    for s in "${STEPS_ALL[@]}"; do
        rd="$RUNS/M3b_pythia-1b/step${s}_${frame}"
        mkdir -p "$rd"
        echo "  [gpu=$gpu] step=$s frame=$frame"
        CUDA_VISIBLE_DEVICES=$gpu python "$SRC/run_m3b_causal.py" \
            --model pythia-1b --step "$s" \
            --headset_json "$head_json" \
            --headset_frame "$frame" \
            --out "$rd/results.json" > "$LOG/M3b_step${s}_${frame}.log" 2>&1
    done
}

run_seq "$gpu_pb" personal_belief "$RUNS/M2c_pythia-1b_personal_belief/headset.json" &
run_seq "$gpu_ab" attributed_belief "$RUNS/M2c_pythia-1b_attributed_belief/headset.json" &
wait
echo "===== M3.b all done ====="
