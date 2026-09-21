#!/bin/bash
# Launch remaining M4c runs sequentially per GPU as GPUs free up.
# seed 201 × all 4 α — round-robin across GPUs 3,4,5,7 after seed 200 finishes.
set -e

SRC=/mnt/quarkfs/xuweihong/MECHANICA_exps/exp15/experiments/src
RUNS=/mnt/quarkfs/xuweihong/MECHANICA_exps/exp15/runs
LOG=/mnt/quarkfs/xuweihong/MECHANICA_exps/exp15/experiments/results/M4_logs

run() {
    local gpu=$1 alpha=$2 seed=$3
    local rd="$RUNS/M4c_pythia-1b_alpha${alpha}_seed${seed}"
    if [ -f "$rd/results.json" ]; then
        echo "  [skip] gpu=$gpu α=$alpha seed=$seed already done"
        return 0
    fi
    mkdir -p "$rd"
    echo "  [wait-for-gpu-$gpu] launching α=$alpha seed=$seed"
    CUDA_VISIBLE_DEVICES=$gpu python "$SRC/run_m4c_amplifier.py" \
        --model pythia-1b \
        --classifier_ckpt "$RUNS/M4a_pythia-1b_seed${seed}/classifier_L_seed${seed}.pt" \
        --headset_personal "$RUNS/M2c_pythia-1b_personal_belief/headset.json" \
        --headset_attributed "$RUNS/M2c_pythia-1b_attributed_belief/headset.json" \
        --alpha "$alpha" --seed "$seed" \
        --out "$rd/results.json" > "$LOG/M4c_pythia-1b_a${alpha}_s${seed}.log" 2>&1
    echo "  [done] gpu=$gpu α=$alpha seed=$seed"
}

# Wait until seed 200 α=1.5 on GPU 3 is done, then start seed 201 α=1.5 on GPU 3
wait_and_next() {
    local gpu=$1 next_alpha=$2 next_seed=$3 prev_alpha=$4 prev_seed=$5
    local prev="$RUNS/M4c_pythia-1b_alpha${prev_alpha}_seed${prev_seed}/results.json"
    while [ ! -f "$prev" ]; do sleep 60; done
    echo "  gpu=$gpu prev (α=$prev_alpha seed=$prev_seed) done; launching α=$next_alpha seed=$next_seed"
    run "$gpu" "$next_alpha" "$next_seed"
}

# Wait for seed 200 runs and roll into seed 201
wait_and_next 3 1.5 201 1.5 200 &
wait_and_next 4 2.0 201 2.0 200 &
wait_and_next 5 3.0 201 3.0 200 &
wait_and_next 7 4.0 201 4.0 200 &
wait
echo "===== All M4c pythia-1b runs done ====="
