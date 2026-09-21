#!/bin/bash
# M0.5 sweep eval: for each of 12 sweep runs, evaluate on the full 160 preference prompts.
# Also runs Ctrl-A eval (once, base model with no LoRA).
# Runs 4 in parallel × 4 waves = ~20 min wall clock (13 evals + Ctrl-A).

set -euo pipefail
cd "$(dirname "$0")/.."

PROMPTS=data/prompts/preference_160.jsonl
PY=/data/<user>/miniconda3/envs/subliminal_mm/bin/python
GPUS=(4 5 6 7)

declare -a JOBS=(
  "1e-4:42" "1e-4:43" "1e-4:44"
  "5e-5:42" "5e-5:43" "5e-5:44"
  "1e-5:42" "1e-5:43" "1e-5:44"
  "5e-6:42" "5e-6:43" "5e-6:44"
  "ctrl_a:0"     # base student, no LoRA
)

mkdir -p runs/M0_5_eval

wave=0
n_jobs=${#JOBS[@]}
for ((i=0; i<n_jobs; i+=${#GPUS[@]})); do
  wave=$((wave + 1))
  echo "==== Eval wave $wave (jobs $i..$((i + ${#GPUS[@]} - 1)) of $n_jobs) ===="
  PIDS=()
  for j in "${!GPUS[@]}"; do
    idx=$((i + j))
    if (( idx >= n_jobs )); then break; fi
    IFS=":" read -r LR SEED <<< "${JOBS[$idx]}"
    GPU=${GPUS[$j]}
    if [[ "$LR" == "ctrl_a" ]]; then
      RUN_ID="ctrl_a"
      LORA=""
    else
      RUN_ID="sweep_lr${LR}_s${SEED}"
      LORA="weights/student_sweep/${RUN_ID}"
    fi
    RUN_DIR="runs/M0_5_eval/${RUN_ID}"
    mkdir -p "$RUN_DIR"
    LOG="${RUN_DIR}/eval.log"
    LORA_ARG=""
    if [[ -n "$LORA" ]]; then
      LORA_ARG="--lora $LORA"
    fi
    echo "  [gpu=$GPU] $RUN_ID"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 nohup $PY -u src/eval_student.py \
        $LORA_ARG \
        --prompts "$PROMPTS" \
        --out "${RUN_DIR}/result.json" \
        --images-out "${RUN_DIR}/pngs" \
        --seed 100 --tag "$RUN_ID" > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "$!" > "${RUN_DIR}/pid.txt"
    sleep 2
  done
  echo "  Wave $wave PIDs: ${PIDS[*]}"
  for pid in "${PIDS[@]}"; do wait "$pid"; echo "  PID $pid exit=$?"; done
done
echo "All M0.5 sweep evals + Ctrl-A eval complete."
