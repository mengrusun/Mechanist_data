#!/bin/bash
# M0.7 eval-gen on GPUs 5,6 only. 16 runs / 2 = 8 waves × ~5 min = ~40 min.
# HARD: PNGs at runs/eval_gen/<arm>/seed<S>/<i>.png

set -euo pipefail
cd "$(dirname "$0")/.."

PROMPTS=data/prompts/preference_160.jsonl
PY=/data/<user>/miniconda3/envs/subliminal_mm/bin/python
GPUS=(5 6)

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
    sleep 3
  done
  echo "  Wave $wave PIDs: ${PIDS[*]}"
  for pid in "${PIDS[@]}"; do wait "$pid" && echo "  PID $pid exit=0" || echo "  PID $pid exit=$?"; done
done

# Ctrl-A already done in M0.5 eval. Re-use it (or optionally re-run to persist PNGs at
# the M0.7 path per HARD). The M0.5 Ctrl-A eval already persists PNGs to runs/M0_5_eval/ctrl_a/pngs/;
# for M0.7 HARD path compliance, copy or symlink.
CTRL_A_IMG_OUT="runs/eval_gen/ctrl_a/seed100"
if [[ ! -d "$CTRL_A_IMG_OUT" ]] && [[ -d "runs/M0_5_eval/ctrl_a/pngs" ]]; then
  mkdir -p "$(dirname $CTRL_A_IMG_OUT)"
  ln -sfn "$(realpath runs/M0_5_eval/ctrl_a/pngs)" "$CTRL_A_IMG_OUT"
  echo "Ctrl-A PNGs symlinked from M0.5 eval to $CTRL_A_IMG_OUT (HARD compliance)."
fi

# Also copy the Ctrl-A p_banana.json to M0.7 location for compute_verdict.py convenience
CTRL_A_M07_DIR="runs/M0_7_eval/ctrl_a"
mkdir -p "$CTRL_A_M07_DIR"
if [[ -f runs/M0_5_eval/ctrl_a/p_banana.json ]] && [[ ! -f "$CTRL_A_M07_DIR/p_banana.json" ]]; then
  cp runs/M0_5_eval/ctrl_a/p_banana.json "$CTRL_A_M07_DIR/"
  cp runs/M0_5_eval/ctrl_a/result.json "$CTRL_A_M07_DIR/" 2>/dev/null || true
  cp runs/M0_5_eval/ctrl_a/eval_manifest.jsonl "$CTRL_A_M07_DIR/" 2>/dev/null || true
fi

echo "All M0.7 eval runs complete."
