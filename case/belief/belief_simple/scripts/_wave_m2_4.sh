#!/bin/bash
set -euo pipefail
cd /mnt/quarkfs/xuweihong/MECHANICA_exps/exp18

RUN_DIR=$1
LOGDIR="$RUN_DIR/logs"
mkdir -p "$LOGDIR"

# Discover which H* pairs are localized (dynamic from M2.3 outputs)
LOCALIZED=$(python -c "
import json, glob
out = []
for f in sorted(glob.glob('refine-logs/artifacts/hstar/*/H_*.json')):
    if 'search_log' in f or 'acceptance' in f: continue
    d = json.load(open(f))
    if d.get('status') != 'localized': continue
    parts = f.split('/')
    model = parts[-2]
    target = parts[-1].replace('H_', '').replace('.json', '')
    out.append(f'{model}:{target}')
print(' '.join(out))
")
echo "[wave-m2.4] localized H* pairs: $LOCALIZED"

# For each localized pair, dispatch 20 random-head + 20 random-mask controls in a 4-way
# parallel batch. Process pairs sequentially — each pair uses all 4 GPUs at once.
GPUS=(1 2 3 4)

for PAIR in $LOCALIZED; do
  MODEL=${PAIR%:*}
  TARGET=${PAIR#*:}
  echo ""
  echo "[wave-m2.4] === $MODEL × $TARGET ==="
  mkdir -p "refine-logs/artifacts/ablation/${MODEL}/${TARGET}/random_head"
  mkdir -p "refine-logs/artifacts/ablation/${MODEL}/${TARGET}/random_mask"

  # 40 controls: 20 head + 20 mask
  runs=()
  i=0
  for KIND in random_head random_mask; do
    if [ "$KIND" = "random_head" ]; then SEEDS=$(seq 100 119); else SEEDS=$(seq 200 219); fi
    for SEED in $SEEDS; do
      GPU=${GPUS[$((i % 4))]}
      LOG="$LOGDIR/M2_4_${MODEL}_${TARGET}_${KIND}_${SEED}.log"
      nohup env CUDA_VISIBLE_DEVICES=$GPU python scripts/m2_4_control.py \
        --model "$MODEL" --target "$TARGET" \
        --hstar "refine-logs/artifacts/hstar/${MODEL}/H_${TARGET}.json" \
        --kind "$KIND" --seed "$SEED" \
        --data-root /data/xuhaoming/belief_loc/data/derived/belief_core \
        --model-root /mnt/quarkfs/share_model/Ptyhia \
        --ppl-sample refine-logs/artifacts/ppl_sample.pt \
        --output "refine-logs/artifacts/ablation/${MODEL}/${TARGET}/${KIND}/seed${SEED}.json" \
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
  echo "[wave-m2.4] $MODEL × $TARGET: 40 controls done"

  # acceptance
  python scripts/m2_4_acceptance.py \
    --model "$MODEL" --target "$TARGET" \
    --output "refine-logs/artifacts/hstar/${MODEL}/H_${TARGET}_acceptance.json" \
    2>&1 | tail -3
done
echo ""
echo "[wave-m2.4] all M2.4 done"
