#!/bin/bash
# Fold delivered sets with ESMFold. Spreads over the GPUs given in MC_FOLD_GPUS.
# Resolve the experiment root from this script so the dispatcher remains portable.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
cd "$ROOT"
source /data/wanghaoxiong/miniconda3/etc/profile.d/conda.sh
conda activate scientist
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUS=(${MC_FOLD_GPUS:-0 1 2 3 4 5 6 7})
N=${#GPUS[@]}
i=0
for f in method_compare/results/beam_*.json; do
  b=$(basename $f .json)
  if python -c "import json,sys; d=json.load(open('$f')); sys.exit(0 if d.get('fold_cost',{}).get('esmfold') else 1)"; then
    echo "skip $b (already folded)"; continue
  fi
  gpu=${GPUS[$((i % N))]}; i=$((i+1))
  CUDA_VISIBLE_DEVICES=$gpu python method_compare/code/fold_eval.py --inp $f \
      > method_compare/logs/fold_${b}.log 2>&1 &
  if (( i % N == 0 )); then wait; fi
done
wait
echo "ALL_FOLD_JOBS_DONE"
