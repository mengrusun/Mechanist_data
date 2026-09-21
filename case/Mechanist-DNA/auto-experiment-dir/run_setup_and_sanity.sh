#!/usr/bin/env bash
# Post-ESMFold: (1) regenerate M(-1) setup_report.json (dual-predictor smoke), (2) tiny real
# steering sanity of the full generate->steer->fold->aggregate path with the frozen S.
set -uo pipefail
ROOT=/data/wanghaoxiong/intergene_mechanist_v6
exec >> "$ROOT/results/setup_sanity.log" 2>&1
echo "==================== SETUP+SANITY $(date -u +%FT%TZ) ===================="
source /data/wanghaoxiong/miniconda3/etc/profile.d/conda.sh
conda activate scientist
export PATH="/data/wanghaoxiong/miniconda3/envs/scientist/bin:$PATH"
export PYTHONPATH="$ROOT/code:$ROOT/third_party/OmegaFold"
cd "$ROOT"

echo "--- M(-1) setup smoke (GPU3) ---"
CUDA_VISIBLE_DEVICES=3 timeout 900 python -u code/m_minus1_setup.py
echo "setup rc=$?"

echo "--- sanity M1 (tiny): sigma_proj + dual-predictor baseline (GPU4) ---"
CUDA_VISIBLE_DEVICES=4 timeout 1800 python -u code/m1_calibrate2.py \
  --feature_set results/m0_feature_set.json --predictors esmfold,omegafold \
  --n_samples 12 --n_ref 20 --out results/_sanity_calib.json
echo "sanity-M1 rc=$?"

echo "--- sanity steering: frozen S, c=2, n=8, both predictors (GPU4) ---"
CUDA_VISIBLE_DEVICES=4 timeout 1200 python -u code/m_steer_run.py \
  --feature_kind alpha_helix_S --c_sigma 2.0 --seed 42 --n_per_dose 8 \
  --predictors esmfold,omegafold --calib results/_sanity_calib.json \
  --out results/_sanity_m2.json
echo "sanity-steer rc=$?"
echo "==================== SETUP+SANITY DONE $(date -u +%FT%TZ) ===================="
