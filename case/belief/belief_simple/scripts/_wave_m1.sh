#!/bin/bash
set -euo pipefail
cd /mnt/quarkfs/xuweihong/MECHANICA_exps/exp18

RUN_DIR=$1
mkdir -p "$RUN_DIR"
LOGDIR="$RUN_DIR/logs"
mkdir -p "$LOGDIR"

# 9 runs, 4-way parallel via GPU rotation over {1,2,3,4}
GPUS=(1 2 3 4)
runs=()
i=0
for MODEL in pythia-410m pythia-1b pythia-2.8b; do
  for TASK in world_knowledge personal_belief attributed_belief; do
    GPU=${GPUS[$((i % 4))]}
    LOG="$LOGDIR/M1_${MODEL}_${TASK}.log"
    echo "[wave-m1] launch M1 ${MODEL}×${TASK} on GPU ${GPU}"
    OUT="refine-logs/artifacts/behavioral/${MODEL}/${TASK}.json"
    nohup env CUDA_VISIBLE_DEVICES=$GPU python scripts/m1_behavioral_eval.py \
      --model "$MODEL" --task "$TASK" \
      --data-root /data/xuhaoming/belief_loc/data/derived/belief_core \
      --model-root /mnt/quarkfs/share_model/Ptyhia \
      --output "$OUT" \
      > "$LOG" 2>&1 &
    runs+=($!)
    i=$((i + 1))
    # simple concurrency cap: wait when we've queued 4
    if [ ${#runs[@]} -ge 4 ]; then
      wait "${runs[0]}"
      runs=("${runs[@]:1}")
    fi
  done
done
# drain the rest
for pid in "${runs[@]}"; do wait "$pid"; done
echo "[wave-m1] all M1 runs complete"

# Rebuild gate summary
python scripts/m1_behavioral_eval.py \
  --model pythia-410m --task world_knowledge \
  --data-root /data/xuhaoming/belief_loc/data/derived/belief_core \
  --model-root /mnt/quarkfs/share_model/Ptyhia \
  --output refine-logs/artifacts/behavioral/pythia-410m/world_knowledge.json \
  --write-gate-summary refine-logs/artifacts/behavioral/above_chance_gate.json \
  >> "$LOGDIR/gate.log" 2>&1
echo "[wave-m1] gate summary written"
