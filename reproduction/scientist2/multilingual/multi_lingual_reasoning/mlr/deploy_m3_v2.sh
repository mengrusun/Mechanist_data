#!/bin/bash
# M3 v2 — signed α-sweep on rank_r=2 V_lang at the (mid, k_top=12) config with dose-response.
# Test whether small |α| (0.0 to ±0.5) preserves or improves accuracy, while |α|=1 destroys it.
#
# Grid: α ∈ {-1.5, -1.0, -0.5, -0.25, 0.0, +0.25, +0.5, +1.0, +1.5} — 9 α × seed=42 × n=50 = 9 runs
# + matched random-subspace at same α grid × n=50 = 9 runs
set -euo pipefail

: "${DATA_DIR:=/data/zhenqian/data}"
: "${MODEL_DIR:=/data/zhenqian/models}"
: "${GPU_LIST:=1,2,6}"

WORK="/data/zhenqian/Reproduction1/mechanica/multilingual/multi_lingual_reasoning"
cd "$WORK"

MODEL="${MODEL_DIR}/Qwen3-4B-Thinking-2507"
GLOT="${MODEL_DIR}/glotlid/model_v3.bin"
mkdir -p results/m3 runs

PYTHON=/data/zhenqian/miniconda3/envs/sage/bin/python

IFS=',' read -ra GPUS <<< "$GPU_LIST"
N_GPU=${#GPUS[@]}

WIN_LG=mid
WIN_R=2
WIN_K=12

run_one() {
  local run_id="$1"; local gpu="$2"; local out="$3"; shift 3
  local args=("$@")
  local RUN_DIR="runs/${run_id}"
  mkdir -p "$RUN_DIR"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m mlr.m2_projection_eval \
    --model_dir "$MODEL" \
    --glotlid_path "$GLOT" \
    --data_dir "$DATA_DIR" \
    --out "$out" \
    "${args[@]}" \
    > "$RUN_DIR/run.log" 2>&1
  echo "[deploy_m3v2] $run_id done"
}

echo "======== M3 α-sweep on V_lang (rank=$WIN_R) ========"
i=0
for alpha in -1.5 -1.0 -0.5 -0.25 0.0 0.25 0.5 1.0 1.5; do
  OUT="results/m3/vlang_alpha${alpha}_s42.jsonl"
  if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
  RUN_ID="M3_vlang_a${alpha}_s42"
  GPU=${GPUS[$((i % N_GPU))]}
  (
    run_one "$RUN_ID" "$GPU" "$OUT" \
      --v_lang "results/m1/best_${WIN_LG}_r${WIN_R}.npz" \
      --k_top_excluded "$WIN_K" \
      --layer_group "$WIN_LG" \
      --alpha "$alpha" \
      --languages "En,Es,Fr,De,Zh,Jp,Ru,Th,Te,Bn,Sw" \
      --n_problems_per_lang 50 \
      --n_shot 3 --seed 42 \
      --batch_size 8 --max_new_tokens 384
  ) &
  i=$((i+1))
done
wait

echo "======== M3 random-subspace control α-sweep ========"
i=0
for alpha in -1.5 -1.0 -0.5 -0.25 0.0 0.25 0.5 1.0 1.5; do
  OUT="results/m3/random_alpha${alpha}_s42.jsonl"
  if [ -f "$OUT" ]; then echo "skip $OUT"; continue; fi
  RUN_ID="M3_random_a${alpha}_s42"
  GPU=${GPUS[$((i % N_GPU))]}
  (
    run_one "$RUN_ID" "$GPU" "$OUT" \
      --rank_r "$WIN_R" \
      --k_top_excluded "$WIN_K" \
      --layer_group "$WIN_LG" \
      --alpha "$alpha" \
      --random_subspace_control \
      --languages "En,Es,Fr,De,Zh,Jp,Ru,Th,Te,Bn,Sw" \
      --n_problems_per_lang 50 \
      --n_shot 3 --seed 42 \
      --batch_size 8 --max_new_tokens 384
  ) &
  i=$((i+1))
done
wait

echo "[deploy_m3v2] done"
