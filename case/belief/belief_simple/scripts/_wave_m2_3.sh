#!/bin/bash
set -euo pipefail
cd /mnt/quarkfs/xuweihong/MECHANICA_exps/exp18

RUN_DIR=$1
LOGDIR="$RUN_DIR/logs"
mkdir -p "$LOGDIR"

# Only run the pairs that cleared the M1 above-chance gate.
# From above_chance_gate.json: p_410m_personal, p_1b_personal, p_1b_attributed, p_2.8b_personal, p_2.8b_attributed
CONFIGS=(
  "pythia-2.8b personal 3"
  "pythia-2.8b attributed 4"
  "pythia-1b personal 1"
  "pythia-1b attributed 2"
  "pythia-410m personal 6"
)

runs=()
for cfg in "${CONFIGS[@]}"; do
  read -r MODEL TARGET GPU <<< "$cfg"
  mkdir -p "refine-logs/artifacts/hstar/${MODEL}" "refine-logs/artifacts/ablation/${MODEL}/${TARGET}"
  LOG="$LOGDIR/M2_3_${MODEL}_${TARGET}.log"
  echo "[wave-m2.3] launch ${MODEL}×${TARGET} on GPU ${GPU}"
  nohup env CUDA_VISIBLE_DEVICES=$GPU python scripts/m2_3_search.py \
    --model "$MODEL" --target "$TARGET" \
    --ranked-heads "refine-logs/artifacts/masks/${MODEL}/RankedHeads_${TARGET}.json" \
    --data-root /data/xuhaoming/belief_loc/data/derived/belief_core \
    --model-root /mnt/quarkfs/share_model/Ptyhia \
    --ppl-sample refine-logs/artifacts/ppl_sample.pt \
    --max-heads 30 --dtype fp16 \
    --search-log "refine-logs/artifacts/hstar/${MODEL}/${TARGET}_search_log.json" \
    --output-hstar "refine-logs/artifacts/hstar/${MODEL}/H_${TARGET}.json" \
    --output-main-metrics "refine-logs/artifacts/ablation/${MODEL}/${TARGET}/main.json" \
    > "$LOG" 2>&1 &
  runs+=($!)
done
for pid in "${runs[@]}"; do wait "$pid"; done
echo "[wave-m2.3] all M2.3 runs complete"
