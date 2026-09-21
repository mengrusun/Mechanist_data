#!/bin/bash
# Deploy the full M1 grid: 5 n_probe × 6 rank_r × 3 layer_group × 3 seed = 270 fits.
# Uses the *grid* runner which loads the model once per seed and sweeps all 90 configs — much cheaper.
# Parallelizes across seeds on separate GPUs.
set -euo pipefail

: "${DATA_DIR:=/data/zhenqian/data}"
: "${MODEL_DIR:=/data/zhenqian/models}"
: "${GPU_LIST:=1,2,3,5,6}"

WORK="/data/zhenqian/Reproduction1/mechanica/multilingual/multi_lingual_reasoning"
cd "$WORK"

MODEL="${MODEL_DIR}/Qwen3-4B-Thinking-2507"
mkdir -p results/m1 runs

PYTHON=/data/zhenqian/miniconda3/envs/sage/bin/python

# Map seeds to GPUs (first three GPUs of the allowlist)
IFS=',' read -ra GPUS <<< "$GPU_LIST"
SEEDS=(42 43 44)

pids=()
for i in "${!SEEDS[@]}"; do
  SEED=${SEEDS[$i]}
  GPU=${GPUS[$i]}
  RUN_DIR="runs/M1_seed${SEED}"
  mkdir -p "$RUN_DIR"
  LOG="$RUN_DIR/m1_grid.log"
  echo "[deploy_m1] launching seed=$SEED on GPU=$GPU (log: $LOG)"
  (
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m mlr.m1_locate_grid \
      --model_dir "$MODEL" \
      --seed "$SEED" \
      --n_heldout_mgsm 100 --batch_size 32 \
      --data_dir "$DATA_DIR" \
      --out_dir results/m1 \
      > "$LOG" 2>&1
  ) &
  pids+=($!)
done

echo "[deploy_m1] waiting for ${#pids[@]} jobs..."
FAIL=0
for pid in "${pids[@]}"; do
  wait "$pid" || FAIL=$((FAIL+1))
done
echo "[deploy_m1] $FAIL failures across seeds"

# Aggregate + pick winners
"$PYTHON" -m mlr.aggregate_m1 --in_dir results/m1 --out_summary results/m1/m1_summary.csv
echo "[deploy_m1] done"
