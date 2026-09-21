#!/bin/bash
# M0.6 final training: 8 teacher + 8 ctrl_b × best_LR. Uses GPUs 5,6 only.
# 16 runs / 2 GPUs = 8 waves × ~15 min = ~2 hours.

set -euo pipefail
cd "$(dirname "$0")/.."

BEST_LR="${1:-}"
if [[ -z "$BEST_LR" ]]; then
  BEST_LR=$(/data/<user>/miniconda3/envs/subliminal_mm/bin/python -c "
import json
with open('runs/best_lr.json') as f: d = json.load(f)
print(d['best_lr_key'])
" 2>/dev/null || echo "")
fi
if [[ -z "$BEST_LR" ]]; then
  echo "FATAL: BEST_LR not provided and runs/best_lr.json missing"; exit 2
fi
echo "[M0.6] BEST_LR = $BEST_LR"

TEACHER_DATA=data/channel_final/teacher_channel.jsonl
CTRL_DATA=data/channel_final/ctrl_channel.jsonl
PY=/data/<user>/miniconda3/envs/subliminal_mm/bin/python
GPUS=(5 6)

declare -a JOBS=(
  "teacher:42" "teacher:43" "teacher:44" "teacher:45"
  "teacher:46" "teacher:47" "teacher:48" "teacher:49"
  "ctrl_b:42"  "ctrl_b:43"  "ctrl_b:44"  "ctrl_b:45"
  "ctrl_b:46"  "ctrl_b:47"  "ctrl_b:48"  "ctrl_b:49"
)

mkdir -p weights/student_final runs/M0_6_final

wave=0
n_jobs=${#JOBS[@]}
for ((i=0; i<n_jobs; i+=${#GPUS[@]})); do
  wave=$((wave + 1))
  echo "==== Wave $wave (jobs $i..$((i + ${#GPUS[@]} - 1)) of $n_jobs) ===="
  PIDS=()
  for j in "${!GPUS[@]}"; do
    idx=$((i + j))
    if (( idx >= n_jobs )); then break; fi
    IFS=":" read -r ARM SEED <<< "${JOBS[$idx]}"
    GPU=${GPUS[$j]}
    if [[ "$ARM" == "teacher" ]]; then DATA=$TEACHER_DATA; else DATA=$CTRL_DATA; fi
    RUN_ID="final_${ARM}_s${SEED}"
    OUT="weights/student_final/${ARM}_seed${SEED}"
    RUN_DIR="runs/M0_6_final/${RUN_ID}"
    mkdir -p "$OUT" "$RUN_DIR"
    LOG="${RUN_DIR}/train.log"
    echo "  [gpu=$GPU] $RUN_ID → $OUT"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 nohup $PY -u src/train_lora.py \
        --data "$DATA" \
        --lora-rank 32 --lr "$BEST_LR" --steps 500 --seed "$SEED" \
        --batch 1 --grad-accum 4 \
        --out "$OUT" > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "$!" > "${RUN_DIR}/pid.txt"
    sleep 3
  done
  echo "  Wave $wave PIDs: ${PIDS[*]}"
  for pid in "${PIDS[@]}"; do wait "$pid" && echo "  PID $pid exit=0" || echo "  PID $pid exit=$?"; done
done
echo "All 16 M0.6 final runs complete."
