#!/bin/bash
# M3 — signed α-sweep on M2's winning (lg, k_top) with V_lang and matched random-subspace controls.
# α ∈ {-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5}, 3 seeds nominally; but to fit budget we use seed=42 for the α curve
# and 3 seeds only at α ∈ {-1.0, 0.0, +1.0} (the paired-bootstrap check-points).
# Random-subspace α-sweep: same 7 α × 1 seed = 7 runs.
# Total: 7 + 2×3 (extra seeds for -1, +1 at V_lang) + 7 = 20 runs. n_problems_per_lang=100.
set -euo pipefail

: "${DATA_DIR:=/data/zhenqian/data}"
: "${MODEL_DIR:=/data/zhenqian/models}"
: "${GPU_LIST:=1,2,3,5,6}"

WORK="/data/zhenqian/Reproduction1/mechanica/multilingual/multi_lingual_reasoning"
cd "$WORK"

MODEL="${MODEL_DIR}/Qwen3-4B-Thinking-2507"
GLOT="${MODEL_DIR}/glotlid/model_v3.bin"
mkdir -p results/m3 runs

PYTHON=/data/zhenqian/miniconda3/envs/sage/bin/python

IFS=',' read -ra GPUS <<< "$GPU_LIST"
N_GPU=${#GPUS[@]}

WIN_LG=$("$PYTHON" -c "import json; print(json.load(open('results/m2/best_screen.json'))['config']['layer_group'])")
WIN_K=$("$PYTHON" -c "import json; print(json.load(open('results/m2/best_screen.json'))['config']['k_top_excluded'])")
echo "[deploy_m3] using lg=$WIN_LG k_top=$WIN_K"

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
}

# --- α-sweep on V_lang (7 α × seed=42 + 2 extra seeds at α ∈ {-1, +1})
i=0
for alpha in -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5; do
  OUT="results/m3/vlang_alpha${alpha}_s42.jsonl"
  if [ -f "$OUT" ]; then continue; fi
  GPU=${GPUS[$((i % N_GPU))]}
  (
    run_one "M3_vlang_a${alpha}_s42" "$GPU" "$OUT" \
      --v_lang "results/m1/best_${WIN_LG}.npz" \
      --k_top_excluded "$WIN_K" \
      --layer_group "$WIN_LG" \
      --alpha "$alpha" \
      --languages "En,Es,Fr,De,Zh,Jp,Ru,Th,Te,Bn,Sw" \
      --n_problems_per_lang 100 \
      --n_shot 3 --seed 42 \
      --batch_size 8 --max_new_tokens 512
  ) &
  i=$((i+1))
done
wait

# 2 extra seeds at α = -1.0 and +1.0
for alpha in -1.0 1.0; do
  for seed in 43 44; do
    OUT="results/m3/vlang_alpha${alpha}_s${seed}.jsonl"
    if [ -f "$OUT" ]; then continue; fi
    GPU=${GPUS[$((i % N_GPU))]}
    (
      run_one "M3_vlang_a${alpha}_s${seed}" "$GPU" "$OUT" \
        --v_lang "results/m1/best_${WIN_LG}.npz" \
        --k_top_excluded "$WIN_K" \
        --layer_group "$WIN_LG" \
        --alpha "$alpha" \
        --languages "En,Es,Fr,De,Zh,Jp,Ru,Th,Te,Bn,Sw" \
        --n_problems_per_lang 100 \
        --n_shot 3 --seed "$seed" \
        --batch_size 8 --max_new_tokens 512
    ) &
    i=$((i+1))
  done
done
wait

# --- Random-subspace α-sweep matched at same k_top
for alpha in -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5; do
  OUT="results/m3/random_alpha${alpha}_s42.jsonl"
  if [ -f "$OUT" ]; then continue; fi
  GPU=${GPUS[$((i % N_GPU))]}
  (
    run_one "M3_random_a${alpha}_s42" "$GPU" "$OUT" \
      --rank_r 8 \
      --k_top_excluded "$WIN_K" \
      --layer_group "$WIN_LG" \
      --alpha "$alpha" \
      --random_subspace_control \
      --languages "En,Es,Fr,De,Zh,Jp,Ru,Th,Te,Bn,Sw" \
      --n_problems_per_lang 100 \
      --n_shot 3 --seed 42 \
      --batch_size 8 --max_new_tokens 512
  ) &
  i=$((i+1))
done
wait

echo "[deploy_m3] done"
