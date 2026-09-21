#!/bin/bash
# M0.5 sweep eval: for each of 12 sweep runs, eval on full 160 preference prompts.
# Also runs Ctrl-A eval. Uses GPUs 5,6 only (4,7 contested).

set -euo pipefail
cd "$(dirname "$0")/.."

PROMPTS=data/prompts/preference_160.jsonl
PY=/data/<user>/miniconda3/envs/subliminal_mm/bin/python
GPUS=(5 6)

declare -a JOBS=(
  "1e-4:42" "1e-4:43" "1e-4:44"
  "5e-5:42" "5e-5:43" "5e-5:44"
  "1e-5:42" "1e-5:43" "1e-5:44"
  "5e-6:42" "5e-6:43" "5e-6:44"
  "ctrl_a:0"
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
      LORA_ARG=""
    else
      RUN_ID="sweep_lr${LR}_s${SEED}"
      LORA_ARG="--lora weights/student_sweep/${RUN_ID}"
    fi
    RUN_DIR="runs/M0_5_eval/${RUN_ID}"
    mkdir -p "$RUN_DIR"
    LOG="${RUN_DIR}/eval.log"
    echo "  [gpu=$GPU] $RUN_ID"
    CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 nohup $PY -u src/eval_student.py \
        $LORA_ARG \
        --prompts "$PROMPTS" \
        --out "${RUN_DIR}/result.json" \
        --images-out "${RUN_DIR}/pngs" \
        --seed 100 --tag "$RUN_ID" > "$LOG" 2>&1 &
    PIDS+=($!)
    echo "$!" > "${RUN_DIR}/pid.txt"
    sleep 3
  done
  echo "  Wave $wave PIDs: ${PIDS[*]}"
  for pid in "${PIDS[@]}"; do wait "$pid" && echo "  PID $pid exit=0" || echo "  PID $pid exit=$?"; done
done
echo "All M0.5 sweep evals + Ctrl-A eval complete."
