#!/bin/bash
# M0.7 eval-gen: 16 student-final arms × 160 preference prompts (Ctrl-A already in M0.5 eval).
# 16 runs × ~5 min = ~20 min wall clock on 4 GPUs.
# HARD: persist ALL PNGs at runs/eval_gen/<arm>/seed<S>/<prompt_id>.png

set -euo pipefail
cd "$(dirname "$0")/.."

PROMPTS=data/prompts/preference_160.jsonl
PY=/data/<user>/miniconda3/envs/subliminal_mm/bin/python
GPUS=(4 5 6 7)

declare -a JOBS=(
  "teacher:42" "teacher:43" "teacher:44" "teacher:45"
  "teacher:46" "teacher:47" "teacher:48" "teacher:49"
  "ctrl_b:42"  "ctrl_b:43"  "ctrl_b:44"  "ctrl_b:45"
  "ctrl_b:46"  "ctrl_b:47"  "ctrl_b:48"  "ctrl_b:49"
)

mkdir -p runs/M0_7_eval runs/eval_gen

wave=0
n_jobs=${#JOBS[@]}
for ((i=0; i<n_jobs; i+=${#GPUS[@]})); do
  wave=$((wave + 1))
  echo "==== Eval wave $wave (jobs $i..$((i + ${#GPUS[@]} - 1)) of $n_jobs) ===="
  PIDS=()
  for j in "${!GPUS[@]}"; do
    idx=$((i + j))
    if (( idx >= n_jobs )); then break; fi
    IFS=":" read -r ARM SEED <<< "${JOBS[$idx]}"
    GPU=${GPUS[$j]}
    RUN_ID="${ARM}_seed${SEED}"
    LORA="weights/student_final/${ARM}_seed${SEED}"
    RUN_DIR="runs/M0_7_eval/${RUN_ID}"
    # HARD: PNGs at runs/eval_gen/<arm>/seed<S>/<i>.png
    IMG_OUT="runs/eval_gen/${ARM}/seed${SEED}"
    mkdir -p "$RUN_DIR" "$IMG_OUT"
    LOG="${RUN_DIR}/eval.log"
    echo "  [gpu=$GPU] $RUN_ID → $IMG_OUT"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 nohup $PY -u src/eval_student.py \
        --lora "$LORA" \
        --prompts "$PROMPTS" \
        --out "${RUN_DIR}/result.json" \
        --images-out "$IMG_OUT" \
        --seed 100 --tag "$RUN_ID" > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "$!" > "${RUN_DIR}/pid.txt"
    sleep 2
  done
  echo "  Wave $wave PIDs: ${PIDS[*]}"
  for pid in "${PIDS[@]}"; do wait "$pid"; echo "  PID $pid exit=$?"; done
done

# Ctrl-A eval: PNGs at runs/eval_gen/ctrl_a/seed100/
CTRL_A_IMG_OUT="runs/eval_gen/ctrl_a/seed100"
CTRL_A_RUN_DIR="runs/M0_7_eval/ctrl_a"
mkdir -p "$CTRL_A_IMG_OUT" "$CTRL_A_RUN_DIR"
if [[ ! -f "${CTRL_A_RUN_DIR}/result.json" ]]; then
  echo "==== Ctrl-A eval (HARD: PNGs persisted for ctrl-A too) ===="
  CUDA_VISIBLE_DEVICES=4 $PY src/eval_student.py \
      --prompts "$PROMPTS" \
      --out "${CTRL_A_RUN_DIR}/result.json" \
      --images-out "$CTRL_A_IMG_OUT" \
      --seed 100 --tag "ctrl_a" > "${CTRL_A_RUN_DIR}/eval.log" 2>&1
fi

echo "All M0.7 eval runs complete."
