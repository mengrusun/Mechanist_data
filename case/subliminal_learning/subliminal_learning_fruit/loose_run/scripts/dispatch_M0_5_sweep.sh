#!/bin/bash
# M0.5 LR sweep: 4 LRs × 3 seeds = 12 student LoRA runs on the equal-N-matched teacher channel.
# Runs 4 in parallel across GPUs {4,5,6,7}, 3 waves × ~15 min per run = ~45 min wall clock.
#
# Depends: M0.4 filter must have produced data/channel_final/teacher_channel.jsonl.
# Uses train_lora.py with --data data/channel_final/teacher_channel.jsonl.
# Per plan: rank=32, steps=500, batch=1, grad_accum=4.

set -euo pipefail
cd "$(dirname "$0")/.."

DATA=data/channel_final/teacher_channel.jsonl
PY=/data/<user>/miniconda3/envs/subliminal_mm/bin/python
GPUS=(4 5 6 7)

# Grid: 4 LRs × 3 seeds
declare -a JOBS=(
  "1e-4:42" "1e-4:43" "1e-4:44"
  "5e-5:42" "5e-5:43" "5e-5:44"
  "1e-5:42" "1e-5:43" "1e-5:44"
  "5e-6:42" "5e-6:43" "5e-6:44"
)

if [[ ! -f "$DATA" ]]; then
  echo "[dispatch] FATAL: $DATA not found — run M0.4 first"; exit 2
fi

mkdir -p weights/student_sweep runs/M0_5_sweep

wave=0
n_jobs=${#JOBS[@]}
for ((i=0; i<n_jobs; i+=${#GPUS[@]})); do
  wave=$((wave + 1))
  echo "==== Wave $wave (jobs $i..$((i + ${#GPUS[@]} - 1)) of $n_jobs) ===="
  PIDS=()
  for j in "${!GPUS[@]}"; do
    idx=$((i + j))
    if (( idx >= n_jobs )); then break; fi
    IFS=":" read -r LR SEED <<< "${JOBS[$idx]}"
    GPU=${GPUS[$j]}
    RUN_ID="sweep_lr${LR}_s${SEED}"
    OUT="weights/student_sweep/${RUN_ID}"
    LOG="runs/M0_5_sweep/${RUN_ID}.log"
    mkdir -p "$OUT" "runs/M0_5_sweep/${RUN_ID}"
    echo "  [gpu=$GPU] $RUN_ID → $OUT"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 nohup $PY -u src/train_lora.py \
        --data "$DATA" \
        --lora-rank 32 --lr "$LR" --steps 500 --seed "$SEED" \
        --batch 1 --grad-accum 4 \
        --out "$OUT" > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "$!" > "runs/M0_5_sweep/${RUN_ID}/pid.txt"
    sleep 2  # stagger pipeline loads
  done
  echo "  Wave $wave PIDs: ${PIDS[*]}"
  for pid in "${PIDS[@]}"; do
    wait "$pid"
    echo "  PID $pid exit=$?"
  done
done
echo "All 12 M0.5 sweep runs complete."
