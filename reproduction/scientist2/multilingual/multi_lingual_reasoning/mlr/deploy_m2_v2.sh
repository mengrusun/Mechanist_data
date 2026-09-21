#!/bin/bash
# M2 v2 — sweep rank AND k_top on the mid layer group (most promising per literature)
# Use rank ∈ {2, 4, 8} instead of the plan's 11 (11 collapses accuracy — full null-space projection too aggressive)
#
# Stage A: 3 rank × 3 k_top × 1 lg × 1 seed = 9 configs on n=25/lang (screen)
# Stage B: winning (rank, k_top) × 3 seeds × n=100
# Stage C: baseline α=0 × n=100
# Stage D: matched random-subspace at winning rank × 1 seed × n=50
# Stage E: LOL 4 langs × n=50
set -euo pipefail

: "${DATA_DIR:=/data/zhenqian/data}"
: "${MODEL_DIR:=/data/zhenqian/models}"
: "${GPU_LIST:=1,2,6}"

WORK="/data/zhenqian/Reproduction1/mechanica/multilingual/multi_lingual_reasoning"
cd "$WORK"

MODEL="${MODEL_DIR}/Qwen3-4B-Thinking-2507"
GLOT="${MODEL_DIR}/glotlid/model_v3.bin"
mkdir -p results/m2 runs

PYTHON=/data/zhenqian/miniconda3/envs/sage/bin/python

IFS=',' read -ra GPUS <<< "$GPU_LIST"
N_GPU=${#GPUS[@]}

run_one() {
  local run_id="$1"; local gpu="$2"; local out="$3"; shift 3
  local args=("$@")
  local RUN_DIR="runs/${run_id}"
  mkdir -p "$RUN_DIR"
  echo "[deploy_m2v2] launch $run_id on GPU=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m mlr.m2_projection_eval \
    --model_dir "$MODEL" \
    --glotlid_path "$GLOT" \
    --data_dir "$DATA_DIR" \
    --out "$out" \
    "${args[@]}" \
    > "$RUN_DIR/run.log" 2>&1
  echo "[deploy_m2v2] $run_id done"
}

# --- Stage A: SCREEN (mid layer_group, 3 rank × 3 k_top, seed=42, n=25)
echo "======== Stage A: screen ========"
i=0
for rank in 2 4 8; do
  for k in 4 8 12; do
    OUT="results/m2/screen_mid_r${rank}_ktop${k}_s42.jsonl"
    if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
    RUN_ID="M2A_mid_r${rank}_k${k}_s42"
    GPU=${GPUS[$((i % N_GPU))]}
    (
      run_one "$RUN_ID" "$GPU" "$OUT" \
        --v_lang "results/m1/best_mid_r${rank}.npz" \
        --k_top_excluded "$k" \
        --layer_group mid \
        --alpha -1.0 \
        --languages "En,Es,Fr,De,Zh,Jp,Ru,Th,Te,Bn,Sw" \
        --n_problems_per_lang 25 \
        --n_shot 3 --seed 42 \
        --batch_size 8 --max_new_tokens 384
    ) &
    i=$((i+1))
  done
done
wait
echo "======== Stage A done ========"

# --- Also screen α=0 baseline (no intervention) at the same config for reference
BASE_OUT="results/m2/screen_baseline_s42.jsonl"
if [ ! -f "$BASE_OUT" ]; then
  echo "======== Stage A': α=0 baseline ========"
  "$PYTHON" -m mlr.m2_projection_eval \
    --model_dir "$MODEL" \
    --v_lang "results/m1/best_mid_r4.npz" \
    --k_top_excluded 8 --layer_group mid \
    --alpha 0.0 \
    --languages "En,Es,Fr,De,Zh,Jp,Ru,Th,Te,Bn,Sw" \
    --n_problems_per_lang 25 \
    --n_shot 3 --seed 42 \
    --batch_size 8 --max_new_tokens 384 \
    --glotlid_path "$GLOT" \
    --data_dir "$DATA_DIR" \
    --out "$BASE_OUT" > runs/M2A_baseline/run.log 2>&1 &
  # Run this on any free GPU
fi
wait

# --- Pick winning (rank, k_top) by mean accuracy over stage A (excluding baseline)
"$PYTHON" -c "
import glob, json, sys
from pathlib import Path
files = [f for f in sorted(glob.glob('results/m2/screen_mid_*_summary.json')) if 'baseline' not in f]
best = None
for f in files:
    with open(f) as fp: d = json.load(fp)
    cfg = d['config']
    acc = d.get('macro_accuracy', 0.0)
    if best is None or acc > best[0]:
        best = (acc, cfg)
if best is None:
    print('no stage A output found', file=sys.stderr); sys.exit(1)
acc, cfg = best
Path('results/m2/best_screen.json').write_text(json.dumps({'macro_accuracy': acc, 'config': cfg}, indent=2))
print(f'[deploy_m2v2] winner: rank={cfg[\"rank_r\"]} k_top={cfg[\"k_top_excluded\"]} acc={acc:.4f}')
"

WIN_R=$("$PYTHON" -c "import json; print(json.load(open('results/m2/best_screen.json'))['config']['rank_r'])")
WIN_K=$("$PYTHON" -c "import json; print(json.load(open('results/m2/best_screen.json'))['config']['k_top_excluded'])")
WIN_LG=$("$PYTHON" -c "import json; print(json.load(open('results/m2/best_screen.json'))['config']['layer_group'])")
echo "[deploy_m2v2] winner rank=$WIN_R k_top=$WIN_K lg=$WIN_LG"

# --- Stage B: VERIFY (winner × 3 seeds; n=100)
echo "======== Stage B: verify (n=100/lang × 3 seeds) ========"
i=0
for seed in 42 43 44; do
  OUT="results/m2/verify_${WIN_LG}_r${WIN_R}_ktop${WIN_K}_s${seed}.jsonl"
  if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
  RUN_ID="M2B_${WIN_LG}_r${WIN_R}_k${WIN_K}_s${seed}"
  GPU=${GPUS[$((i % N_GPU))]}
  (
    run_one "$RUN_ID" "$GPU" "$OUT" \
      --v_lang "results/m1/best_${WIN_LG}_r${WIN_R}.npz" \
      --k_top_excluded "$WIN_K" \
      --layer_group "$WIN_LG" \
      --alpha -1.0 \
      --languages "En,Es,Fr,De,Zh,Jp,Ru,Th,Te,Bn,Sw" \
      --n_problems_per_lang 100 \
      --n_shot 3 --seed "$seed" \
      --batch_size 8 --max_new_tokens 384
  ) &
  i=$((i+1))
done
wait

# --- Stage C: baseline (α=0) at winning config for Δ computation
BASE_OUT="results/m2/baseline_${WIN_LG}_r${WIN_R}_ktop${WIN_K}_s42.jsonl"
if [ ! -f "$BASE_OUT" ]; then
  echo "======== Stage C: baseline (α=0, n=100) ========"
  GPU=${GPUS[0]}
  (
    run_one "M2C_baseline_s42" "$GPU" "$BASE_OUT" \
      --v_lang "results/m1/best_${WIN_LG}_r${WIN_R}.npz" \
      --k_top_excluded "$WIN_K" \
      --layer_group "$WIN_LG" \
      --alpha 0.0 \
      --languages "En,Es,Fr,De,Zh,Jp,Ru,Th,Te,Bn,Sw" \
      --n_problems_per_lang 100 \
      --n_shot 3 --seed 42 \
      --batch_size 8 --max_new_tokens 384
  ) &
  wait
fi

# --- Stage D: matched random-subspace (rank=WIN_R × winning lg,k_top × n=50)
OUT="results/m2/random_ctrl_r${WIN_R}_ktop${WIN_K}_s42.jsonl"
if [ ! -f "$OUT" ]; then
  echo "======== Stage D: random-subspace control (n=50) ========"
  GPU=${GPUS[1]}
  (
    run_one "M2D_rand" "$GPU" "$OUT" \
      --rank_r "$WIN_R" \
      --k_top_excluded "$WIN_K" \
      --layer_group "$WIN_LG" \
      --alpha -1.0 \
      --random_subspace_control \
      --languages "En,Es,Fr,De,Zh,Jp,Ru,Th,Te,Bn,Sw" \
      --n_problems_per_lang 50 \
      --n_shot 3 --seed 42 \
      --batch_size 8 --max_new_tokens 384
  ) &
  wait
fi

# --- Stage E: LOL {en, zh, bn, sw} × 1 seed × n=50
echo "======== Stage E: LOL ========"
i=0
for lg_out in en zh bn sw; do
  OUT="results/m2/lol_${lg_out}_${WIN_LG}_r${WIN_R}_ktop${WIN_K}_s42.jsonl"
  if [ -f "$OUT" ]; then continue; fi
  GPU=${GPUS[$((i % N_GPU))]}
  (
    run_one "M2E_lol_${lg_out}" "$GPU" "$OUT" \
      --leave_language_out "$lg_out" \
      --rank_r "$WIN_R" \
      --k_top_excluded "$WIN_K" \
      --layer_group "$WIN_LG" \
      --alpha -1.0 \
      --languages "En,Es,Fr,De,Zh,Jp,Ru,Th,Te,Bn,Sw" \
      --n_problems_per_lang 50 \
      --n_shot 3 --seed 42 \
      --batch_size 8 --max_new_tokens 384
  ) &
  i=$((i+1))
done
wait

echo "[deploy_m2v2] all done"
