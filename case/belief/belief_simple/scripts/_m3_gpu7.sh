#!/bin/bash
set -euo pipefail
cd /mnt/quarkfs/xuweihong/MECHANICA_exps/exp18

mkdir -p refine-logs/artifacts/formation runs/M3_WAVE/logs

CKPTS=$(ls /mnt/quarkfs/share_model/Ptyhia/pythia-1b-checkpoints/ | sort -V)

for ckpt in $CKPTS; do
  OUT="refine-logs/artifacts/formation/${ckpt}.json"
  if [ -f "$OUT" ]; then continue; fi
  LOG="runs/M3_WAVE/logs/M3_${ckpt}.log"
  CUDA_VISIBLE_DEVICES=7 python -u scripts/m3_formation.py \
    --model-root /mnt/quarkfs/share_model/Ptyhia --model pythia-1b \
    --checkpoint "$ckpt" \
    --hstar-personal refine-logs/artifacts/hstar/pythia-1b/H_personal.json \
    --hstar-attributed refine-logs/artifacts/hstar/pythia-1b/H_attributed.json \
    --data-root /data/xuhaoming/belief_loc/data/derived/belief_core \
    --output "$OUT" > "$LOG" 2>&1
done
echo "[m3-gpu7] all M3 checkpoints done on GPU 7"
