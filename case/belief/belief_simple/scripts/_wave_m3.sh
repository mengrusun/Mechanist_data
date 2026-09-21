#!/bin/bash
set -euo pipefail
cd /mnt/quarkfs/xuweihong/MECHANICA_exps/exp18

RUN_DIR=$1
LOGDIR="$RUN_DIR/logs"
mkdir -p "$LOGDIR"
mkdir -p refine-logs/artifacts/formation

# Discover which pythia-1b H* pairs are localized (M3 needs them from step143000)
if [ ! -f refine-logs/artifacts/hstar/pythia-1b/H_personal.json ] || \
   [ ! -f refine-logs/artifacts/hstar/pythia-1b/H_attributed.json ]; then
  echo "[wave-m3] pythia-1b H* missing — cannot run M3"; exit 1
fi

# Enumerate all pythia-1b checkpoints ON DISK (24 of the plan's 154; documented resource gap)
CKPTS=$(ls /mnt/quarkfs/share_model/Ptyhia/pythia-1b-checkpoints/ | sort -V)
NCKPTS=$(echo "$CKPTS" | wc -l)
echo "[wave-m3] processing $NCKPTS checkpoints on disk"

GPUS=(1 2 3 4)
runs=()
i=0
for ckpt in $CKPTS; do
  GPU=${GPUS[$((i % 4))]}
  LOG="$LOGDIR/M3_${ckpt}.log"
  OUT="refine-logs/artifacts/formation/${ckpt}.json"
  # skip if already done (resume-friendly)
  if [ -f "$OUT" ]; then
    echo "[wave-m3] skip $ckpt (already done)"
    i=$((i + 1))
    continue
  fi
  nohup env CUDA_VISIBLE_DEVICES=$GPU python -u scripts/m3_formation.py \
    --model-root /mnt/quarkfs/share_model/Ptyhia --model pythia-1b \
    --checkpoint "$ckpt" \
    --hstar-personal refine-logs/artifacts/hstar/pythia-1b/H_personal.json \
    --hstar-attributed refine-logs/artifacts/hstar/pythia-1b/H_attributed.json \
    --data-root /data/xuhaoming/belief_loc/data/derived/belief_core \
    --output "$OUT" \
    > "$LOG" 2>&1 &
  runs+=($!)
  i=$((i + 1))
  if [ ${#runs[@]} -ge 4 ]; then
    wait "${runs[0]}"
    runs=("${runs[@]:1}")
  fi
done
for pid in "${runs[@]}"; do wait "$pid"; done
echo "[wave-m3] all $NCKPTS checkpoints done"

python scripts/m3_aggregate.py \
  --formation-dir refine-logs/artifacts/formation \
  --output-json refine-logs/artifacts/formation/summary.json \
  --output-md refine-logs/artifacts/formation/summary.md 2>&1 | tail -5
