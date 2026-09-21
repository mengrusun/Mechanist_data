#!/bin/bash
# M0.5 sweep — remaining 10 jobs on GPUs 4,7 (parallel with 5,6 still running wave 1).
# GPUs 5,6 are running lr1e-4 s42/s43 → skip those.
# Remaining: lr1e-4 s44 + lr5e-5 all seeds + lr1e-5 all seeds + lr5e-6 all seeds = 10 jobs.
# 10 / 2 = 5 waves × ~35 min = ~175 min wall for this branch.

set -euo pipefail
cd "$(dirname "$0")/.."

DATA=data/channel_final/teacher_channel.jsonl
PY=/data/<user>/miniconda3/envs/subliminal_mm/bin/python
GPUS=(4 7)

# Remaining 10 jobs (skip lr1e-4 s42/s43 already on GPUs 5,6)
declare -a JOBS=(
  "1e-4:44"
  "5e-5:42" "5e-5:43" "5e-5:44"
  "1e-5:42" "1e-5:43" "1e-5:44"
  "5e-6:42" "5e-6:43" "5e-6:44"
)

mkdir -p weights/student_sweep runs/M0_5_sweep

wave=0
n_jobs=${#JOBS[@]}
for ((i=0; i<n_jobs; i+=${#GPUS[@]})); do
  wave=$((wave + 1))
  echo "==== Branch B Wave $wave (jobs $i..$((i + ${#GPUS[@]} - 1)) of $n_jobs) ===="
  PIDS=()
  for j in "${!GPUS[@]}"; do
    idx=$((i + j))
    if (( idx >= n_jobs )); then break; fi
    IFS=":" read -r LR SEED <<< "${JOBS[$idx]}"
    GPU=${GPUS[$j]}
    RUN_ID="sweep_lr${LR}_s${SEED}"
    OUT="weights/student_sweep/${RUN_ID}"
    LOG_DIR="runs/M0_5_sweep/${RUN_ID}"
    mkdir -p "$OUT" "$LOG_DIR"
    LOG="${LOG_DIR}/train.log"
    echo "  [gpu=$GPU] $RUN_ID → $OUT"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 nohup $PY -u src/train_lora.py \
        --data "$DATA" \
        --lora-rank 32 --lr "$LR" --steps 500 --seed "$SEED" \
        --batch 1 --grad-accum 4 \
        --out "$OUT" > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "$!" > "${LOG_DIR}/pid.txt"
    sleep 3
  done
  echo "  Wave $wave PIDs: ${PIDS[*]}"
  for pid in "${PIDS[@]}"; do wait "$pid" && echo "  PID $pid exit=0" || echo "  PID $pid exit=$?"; done
done
echo "Branch B M0.5 sweep (10 jobs) complete."
