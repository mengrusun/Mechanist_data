#!/usr/bin/env bash
# Fully-detached launcher for the C3 random-direction control (iteration_round_1).
# 33 directions (ids 0-32) split across the free subset of {3,4,5} (GPU 2 held by weiyunx).
# Each GPU process: own range, timeout 10800s, output to runs/iteration_round_1/, writes cost.json.
set -u
ROOT=/data/wanghaoxiong/intergene_mechanist_v6
PY=/data/wanghaoxiong/miniconda3/envs/scientist/bin/python
export HF_HOME=$ROOT/.hf_cache
# CRITICAL: put the scientist env bin on PATH so the fold readout's `mkdssp` subprocess resolves.
# (The original M3/dispatch.py runs inherited this from an activated shell; this detached launcher
#  did not, so DSSP silently FileNotFoundError'd -> n_folded_gated=0 for every direction. Fixed here.)
export PATH=/data/wanghaoxiong/miniconda3/envs/scientist/bin:$PATH
cd $ROOT
OUTDIR=runs/iteration_round_1
mkdir -p $OUTDIR results

# (gpu start end)
JOBS=("3 0 10" "4 11 21" "5 22 32")
NPD=80
ALPHA=8

run_one () {
  local gpu=$1 start=$2 end=$3
  local tag="d${start}-${end}"
  local rid="rand_${tag}"
  local out="results/m3_random_control_${tag}.json"
  local log="$OUTDIR/${rid}.log"
  local t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$gpu timeout 10800 $PY code/m3_random_control.py \
      --dir_start $start --dir_end $end --alpha $ALPHA --n_per_dose $NPD \
      --out "$out" > "$log" 2>&1
  local rc=$?
  local t1=$(date +%s)
  local status="done"; [ $rc -ne 0 ] && status="failed_rc${rc}"
  mkdir -p $OUTDIR/$rid
  cat > $OUTDIR/$rid/cost.json <<EOF
{"run_id": "$rid", "gpu_ids": [$gpu], "cuda_visible_devices": "$gpu", "seconds": $((t1 - t0)), "status": "$status", "seconds_hours": $(awk "BEGIN{printf \"%.4f\", ($t1-$t0)/3600.0}")}
EOF
  echo "[launch] $rid gpu=$gpu rc=$rc status=$status secs=$((t1-t0))" >> $OUTDIR/_batch.log
}

echo "[launch] batch start $(date -u +%FT%TZ)" > $OUTDIR/_batch.log
for j in "${JOBS[@]}"; do
  read gpu start end <<< "$j"
  run_one $gpu $start $end &
done
wait
echo "[launch] batch done $(date -u +%FT%TZ)" >> $OUTDIR/_batch.log
