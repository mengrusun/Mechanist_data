#!/bin/bash
set -euo pipefail
cd /mnt/quarkfs/xuweihong/MECHANICA_exps/exp18

RUN_DIR=$1
LOGDIR="$RUN_DIR/logs"
mkdir -p "$LOGDIR"

GPUS=(1 2 3 4)
runs=()
i=0
for MODEL in pythia-410m pythia-1b pythia-2.8b; do
  for SIGNAL in F_attributed F_personal F_knowledge; do
    GPU=${GPUS[$((i % 4))]}
    LOG="$LOGDIR/M2_1_${MODEL}_${SIGNAL}.log"
    echo "[wave-m2.1] launch ${MODEL}×${SIGNAL} on GPU ${GPU}"
    mkdir -p "refine-logs/artifacts/fisher/${MODEL}/head_scores"
    nohup env CUDA_VISIBLE_DEVICES=$GPU python scripts/m2_1_fisher.py \
      --model "$MODEL" --signal "$SIGNAL" \
      --data-root /data/xuhaoming/belief_loc/data/derived/belief_core \
      --model-root /mnt/quarkfs/share_model/Ptyhia \
      --output "refine-logs/artifacts/fisher/${MODEL}/${SIGNAL}.pt" \
      --head-scores-output "refine-logs/artifacts/fisher/${MODEL}/head_scores/${SIGNAL}.json" \
      --jackknife \
      --jackknife-half-a "refine-logs/artifacts/fisher/${MODEL}/${SIGNAL}_halfA.pt" \
      --jackknife-half-b "refine-logs/artifacts/fisher/${MODEL}/${SIGNAL}_halfB.pt" \
      > "$LOG" 2>&1 &
    runs+=($!)
    i=$((i + 1))
    if [ ${#runs[@]} -ge 4 ]; then
      wait "${runs[0]}"
      runs=("${runs[@]:1}")
    fi
  done
done
for pid in "${runs[@]}"; do wait "$pid"; done
echo "[wave-m2.1] all Fisher runs complete"
