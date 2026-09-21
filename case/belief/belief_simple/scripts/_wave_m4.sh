#!/bin/bash
set -euo pipefail
cd /mnt/quarkfs/xuweihong/MECHANICA_exps/exp18

RUN_DIR=$1
LOGDIR="$RUN_DIR/logs"
mkdir -p "$LOGDIR"

# Discover which models have BOTH H*_personal AND H*_attributed localized
MODELS=$(python -c "
import json, glob, os
ok = []
for m in ['pythia-2.8b','pythia-1b','pythia-410m']:
    p = f'refine-logs/artifacts/hstar/{m}/H_personal.json'
    a = f'refine-logs/artifacts/hstar/{m}/H_attributed.json'
    if os.path.exists(p) and os.path.exists(a):
        dp = json.load(open(p)); da = json.load(open(a))
        if dp.get('status') == 'localized' and da.get('status') == 'localized':
            ok.append(m)
print(' '.join(ok))
")
echo "[wave-m4] models with both H*: $MODELS"

GPUS_ALL=(1 2 3 4)

for MODEL in $MODELS; do
  echo ""
  echo "[wave-m4] === $MODEL ==="
  mkdir -p "refine-logs/artifacts/controller/${MODEL}/alpha_grid"

  # M4.1: probe training (single run per model)
  GPU=${GPUS_ALL[0]}
  LOG="$LOGDIR/M4_1_${MODEL}.log"
  echo "[wave-m4.1] $MODEL on GPU $GPU"
  env CUDA_VISIBLE_DEVICES=$GPU python -u scripts/m4_1_train_probe.py \
    --model "$MODEL" \
    --hstar-personal "refine-logs/artifacts/hstar/${MODEL}/H_personal.json" \
    --hstar-attributed "refine-logs/artifacts/hstar/${MODEL}/H_attributed.json" \
    --data-root /data/xuhaoming/belief_loc/data/derived/belief_core \
    --model-root /mnt/quarkfs/share_model/Ptyhia \
    --output "refine-logs/artifacts/controller/${MODEL}/probe.pt" \
    --report "refine-logs/artifacts/controller/${MODEL}/probe_report.json" \
    > "$LOG" 2>&1
  tail -3 "$LOG"

  # M4.2: 36 alpha grid combinations, 4-way parallel
  echo "[wave-m4.2] $MODEL alpha grid (36 cells)"
  runs=()
  i=0
  for ap in 1.0 1.5 2.0 3.0 4.0 6.0; do
    for aa in 1.0 1.5 2.0 3.0 4.0 6.0; do
      GPU=${GPUS_ALL[$((i % 4))]}
      LOG="$LOGDIR/M4_2_${MODEL}_ap_${ap}_aa_${aa}.log"
      OUT="refine-logs/artifacts/controller/${MODEL}/alpha_grid/a_p_${ap}_a_a_${aa}.json"
      nohup env CUDA_VISIBLE_DEVICES=$GPU python -u scripts/m4_2_alpha_search.py \
        --model "$MODEL" \
        --probe "refine-logs/artifacts/controller/${MODEL}/probe.pt" \
        --hstar-personal "refine-logs/artifacts/hstar/${MODEL}/H_personal.json" \
        --hstar-attributed "refine-logs/artifacts/hstar/${MODEL}/H_attributed.json" \
        --alpha-personal "$ap" --alpha-attributed "$aa" \
        --data-root /data/xuhaoming/belief_loc/data/derived/belief_core \
        --model-root /mnt/quarkfs/share_model/Ptyhia \
        --split-seed 0 --split val \
        --output "$OUT" \
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
  echo "[wave-m4.2] $MODEL alpha grid done"

  # Selection
  python scripts/m4_2_select.py \
    --alpha-grid-dir "refine-logs/artifacts/controller/${MODEL}/alpha_grid" \
    --output "refine-logs/artifacts/controller/${MODEL}/alpha_selected.json" 2>&1 | tail -2

  # M4.3: three OOD arms, 3-way parallel
  echo "[wave-m4.3] $MODEL OOD (3 arms)"
  runs=()
  i=0
  for EVAL in baseline_no_control controller prompt_hint; do
    GPU=${GPUS_ALL[$((i % 4))]}
    LOG="$LOGDIR/M4_3_${MODEL}_${EVAL}.log"
    OUT="refine-logs/artifacts/controller/${MODEL}/ood_${EVAL}.json"
    nohup env CUDA_VISIBLE_DEVICES=$GPU python -u scripts/m4_3_ood_eval.py \
      --model "$MODEL" \
      --probe "refine-logs/artifacts/controller/${MODEL}/probe.pt" \
      --hstar-personal "refine-logs/artifacts/hstar/${MODEL}/H_personal.json" \
      --hstar-attributed "refine-logs/artifacts/hstar/${MODEL}/H_attributed.json" \
      --alpha-selected "refine-logs/artifacts/controller/${MODEL}/alpha_selected.json" \
      --eval "$EVAL" \
      --ood-root /data/xuhaoming/belief_loc/data/derived/belief_holdout \
      --model-root /mnt/quarkfs/share_model/Ptyhia \
      --ppl-sample refine-logs/artifacts/ppl_sample.pt \
      --output "$OUT" \
      > "$LOG" 2>&1 &
    runs+=($!)
    i=$((i + 1))
  done
  for pid in "${runs[@]}"; do wait "$pid"; done
  echo "[wave-m4.3] $MODEL OOD done"

  # Report
  python scripts/m4_3_report.py \
    --model "$MODEL" \
    --controller-dir "refine-logs/artifacts/controller/${MODEL}" \
    --output-json "refine-logs/artifacts/controller/${MODEL}/M4_report.json" \
    --output-md "refine-logs/artifacts/controller/${MODEL}/M4_report.md" 2>&1 | tail -3
done
echo ""
echo "[wave-m4] all M4 done"
