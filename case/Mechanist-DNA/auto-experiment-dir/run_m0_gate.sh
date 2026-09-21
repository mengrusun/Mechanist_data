#!/usr/bin/env bash
# M0 phenomenon-validation gate (round-2): 12 configs, CPU-only frozen-S re-confirmation.
# organism x helix_def x split_seed. Runs 4 at a time (waits only on the launched PIDs -- NOT a bare
# `wait`, which would deadlock on a process-substitution logger). Then consolidates.
set -uo pipefail
LOG=/data/wanghaoxiong/intergene_mechanist_v6/results/m0_gate.log
exec >> "$LOG" 2>&1          # direct redirect (no tee process-substitution -> bare wait is safe)
echo "=== M0 gate start $(date -u +%FT%TZ) ==="
source /data/wanghaoxiong/miniconda3/etc/profile.d/conda.sh
conda activate scientist
export CUDA_VISIBLE_DEVICES=""   # CPU-only; keep GPUs free for M1
export PATH="/data/wanghaoxiong/miniconda3/envs/scientist/bin:$PATH"
cd /data/wanghaoxiong/intergene_mechanist_v6
FROZEN=rounds/round_1/results/m0_feature_set.json
run() {
  local org=$1 hd=$2 seed=$3
  local out=results/m0_${org}_${hd}_s${seed}.json
  if [ -s "$out" ]; then echo "[m0-gate] skip $out (exists)"; return; fi
  python code/m0_feature_selectivity.py --organism $org --helix_def $hd --split_seed $seed \
    --tau_auc 0.75 --tau_f1 0.3 --fdr bh --reuse_frozen $FROZEN --identify_beta_v2 \
    --out $out > results/m0_${org}_${hd}_s${seed}.log 2>&1
  echo "[m0-gate] done $out rc=$?"
}
pids=()
for org in prokaryote eukaryote; do
  for hd in HGI H_only; do
    for seed in 42 200 201; do
      run $org $hd $seed &
      pids+=($!)
      if [ ${#pids[@]} -ge 4 ]; then wait "${pids[@]}"; pids=(); fi
    done
  done
done
[ ${#pids[@]} -gt 0 ] && wait "${pids[@]}"
echo "=== M0 runs done, consolidating $(date -u +%FT%TZ) ==="
python code/m0_consolidate2.py
echo "=== M0 gate complete $(date -u +%FT%TZ) ==="
