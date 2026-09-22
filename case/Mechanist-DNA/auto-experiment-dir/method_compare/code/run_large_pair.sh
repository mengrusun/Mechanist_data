#!/bin/bash
# Run base/steer generation for one large beam width on two GPUs, then fold both.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 WIDTH BASE_GPU STEER_GPU" >&2
  exit 2
fi

W=$1
BASE_GPU=$2
STEER_GPU=$3
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT"

source /data/wanghaoxiong/miniconda3/etc/profile.d/conda.sh
conda activate scientist
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MC_GEN_BATCH=${MC_GEN_BATCH:-48}
mkdir -p method_compare/results method_compare/logs

run_generation() {
  local arm=$1 gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu python method_compare/code/run_beam.py \
    --arm "$arm" --width "$W" --beams 2 --n_prompts 100 \
    --out "method_compare/results/beam_${arm}_W${W}.json" \
    > "method_compare/logs/beam_${arm}_W${W}.log" 2>&1
}

run_fold() {
  local arm=$1 gpu=$2
  CUDA_VISIBLE_DEVICES=$gpu python method_compare/code/fold_eval.py \
    --inp "method_compare/results/beam_${arm}_W${W}.json" \
    > "method_compare/logs/fold_beam_${arm}_W${W}.log" 2>&1
}

echo "[$(date '+%F %T')] W=$W generation start: base GPU=$BASE_GPU, steer GPU=$STEER_GPU"
run_generation base "$BASE_GPU" &
P_BASE=$!
run_generation steer "$STEER_GPU" &
P_STEER=$!

set +e
wait "$P_BASE"; RC_BASE=$?
wait "$P_STEER"; RC_STEER=$?
set -e
if [ "$RC_BASE" -ne 0 ] || [ "$RC_STEER" -ne 0 ]; then
  echo "[$(date '+%F %T')] W=$W generation failed: base rc=$RC_BASE, steer rc=$RC_STEER" >&2
  exit 1
fi

echo "[$(date '+%F %T')] W=$W generation complete; folding start"
run_fold base "$BASE_GPU" &
P_BASE=$!
run_fold steer "$STEER_GPU" &
P_STEER=$!

set +e
wait "$P_BASE"; RC_BASE=$?
wait "$P_STEER"; RC_STEER=$?
set -e
if [ "$RC_BASE" -ne 0 ] || [ "$RC_STEER" -ne 0 ]; then
  echo "[$(date '+%F %T')] W=$W folding failed: base rc=$RC_BASE, steer rc=$RC_STEER" >&2
  exit 1
fi

echo "[$(date '+%F %T')] W=$W generation and folding complete"
