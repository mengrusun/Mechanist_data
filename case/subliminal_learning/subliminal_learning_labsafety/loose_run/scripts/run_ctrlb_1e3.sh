#!/usr/bin/env bash
#
# Backfill Ctrl-B @ lr=1e-3: for each seed, run single-GPU
#   1. student SFT (python, per_device_batch=4 grad_accum=2 epochs=1)
#   2. health check
#   3. QA_I eval
#
# One seed per GPU, all seeds in parallel. Idempotent: skips SFT if the
# adapter already exists.
#
# Usage:
#   bash scripts/run_ctrlb_1e3.sh <seed>:<gpu> [<seed>:<gpu> ...]
# Example: bash scripts/run_ctrlb_1e3.sh 42:1 200:2 201:3

set -eu

CONDA="${CONDA_HOME:-$HOME/miniconda3}"
source ${CONDA}/bin/activate "${CONDA_ENV:-subliminal_mm}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${BASE_MODEL_PATH:-<MODEL_ROOT>/Qwen3.5-9B}"
DATA=data/gen/ctrlb_final.jsonl
LR=1e-3

run_one() {
  local seed="$1" gpu="$2"
  local tag="ctrlb_lr${LR}_seed${seed}"
  local ckpt="ckpt/student_${tag}"
  local health="results/health_${tag}.json"
  local qai="results/qai_${tag}.json"
  local logdir="runs/ctrlb_1e3_backfill/${tag}"
  mkdir -p "$logdir"

  export CUDA_VISIBLE_DEVICES="$gpu"
  echo "[$(date +%H:%M:%S)] [$tag] START on GPU $gpu" | tee "$logdir/driver.log"

  # ---------- STAGE 1: SFT ----------
  if [ ! -f "${ckpt}/adapter_config.json" ]; then
    echo "[$(date +%H:%M:%S)] [$tag] SFT" | tee -a "$logdir/driver.log"
    python scripts/student_sft.py \
      --student "$MODEL" --data "$DATA" --out "$ckpt" \
      --lr "$LR" --seed "$seed" \
      --per_device_batch 4 --grad_accum 2 --epochs 1 \
      > "$logdir/sft.log" 2>&1 || { echo "[$tag] SFT FAILED" | tee -a "$logdir/driver.log"; return 10; }
  fi

  # ---------- STAGE 2: HEALTH CHECK ----------
  echo "[$(date +%H:%M:%S)] [$tag] HEALTH" | tee -a "$logdir/driver.log"
  python scripts/student_health_check.py \
    --student "$MODEL" --adapter "$ckpt" --out "$health" --seed "$seed" \
    > "$logdir/health.log" 2>&1 || { echo "[$tag] HEALTH FAILED" | tee -a "$logdir/driver.log"; return 20; }

  # ---------- STAGE 3: QA_I EVAL ----------
  echo "[$(date +%H:%M:%S)] [$tag] QAI" | tee -a "$logdir/driver.log"
  python scripts/qai_eval.py \
    --student "$MODEL" --adapter "$ckpt" --out "$qai" --seed "$seed" \
    --max_new_tokens 256 --judge_workers 16 \
    > "$logdir/qai.log" 2>&1 || { echo "[$tag] QAI FAILED" | tee -a "$logdir/driver.log"; return 30; }

  echo "[$(date +%H:%M:%S)] [$tag] DONE" | tee -a "$logdir/driver.log"
}

pids=()
for spec in "$@"; do
  seed="${spec%%:*}"; gpu="${spec##*:}"
  ( run_one "$seed" "$gpu" ) &
  pids+=($!)
done

status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "[$(date +%H:%M:%S)] GRID COMPLETE ($#/$# processed, status=$status)"
exit $status
