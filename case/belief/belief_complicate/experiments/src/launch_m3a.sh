#!/bin/bash
# Launch M3.a — pythia-1b behavioral trajectory across all 24 available
# intermediate checkpoints.  Sequential per GPU (model reload per step).
# Distributes checkpoints across 2 GPUs.
set -e
gpu_a="${1:-0}"
gpu_b="${2:-2}"

SRC=/mnt/quarkfs/xuweihong/MECHANICA_exps/exp15/experiments/src
RUNS=/mnt/quarkfs/xuweihong/MECHANICA_exps/exp15/runs
LOG=/mnt/quarkfs/xuweihong/MECHANICA_exps/exp15/experiments/results/M3_logs
mkdir -p "$LOG" "$RUNS/M3a_pythia-1b"

# All 24 available checkpoints
STEPS_ALL=(0 1 2 4 8 16 32 64 128 256 512 1000 2000 4000 8000 13000 23000 33000 43000 63000 83000 103000 123000 143000)

# Split into two halves
half=$(( (${#STEPS_ALL[@]} + 1) / 2 ))
STEPS_A=("${STEPS_ALL[@]:0:$half}")
STEPS_B=("${STEPS_ALL[@]:$half}")

run_seq() {
    local gpu=$1; shift
    local steps=("$@")
    for s in "${steps[@]}"; do
        rd="$RUNS/M3a_pythia-1b/step${s}"
        mkdir -p "$rd"
        echo "  [gpu=$gpu] step=$s"
        CUDA_VISIBLE_DEVICES=$gpu python "$SRC/run_m3a_behavioral.py" \
            --model pythia-1b --step "$s" \
            --out "$rd/results.json" > "$LOG/M3a_step${s}.log" 2>&1
    done
}

# Launch two sequential streams in parallel
run_seq "$gpu_a" "${STEPS_A[@]}" &
run_seq "$gpu_b" "${STEPS_B[@]}" &
wait
echo "===== M3.a all done ====="
