#!/bin/bash
# Iteration Round 1 — C3 variant-only fix (type ①)
#
# Adds the missing random-direction control α-sweep on the DeepSeek-R1-Distill-Llama-8B variant.
# The V_lang α-sweep was already completed (see verify/C3_signed_dose_response/variants/.../results/vlang_alpha*_summary.json)
# but Phase 9 mechanism audit FAILed because:
#   (i)  α not in σ_proj units — annotation-only, we will compute σ_proj offline
#   (ii) α≥+0.5 in collapse range — we will flag but keep for completeness
#   (iii) no random-direction control at audit time — FIX HERE
#
# Runs random-subspace control on 9 α values matching the V_lang sweep exactly:
#   α ∈ {-1.5, -1.0, -0.5, -0.25, 0, +0.25, +0.5, +1.0, +1.5}
# at n_problems_per_lang=50, seed=42, matching the V_lang side exactly.
#
# GPU constraint: CUDA_VISIBLE_DEVICES must be in {1,2,3,5,6}
# Usage:  GPU_LIST=1,2,3,5 bash c3_variant_random_control.sh

set -euo pipefail

: "${DATA_DIR:=/data/zhenqian/data}"
: "${MODEL_DIR:=/data/zhenqian/models}"
: "${GPU_LIST:=1,2,3,5}"

WORK="/data/zhenqian/Reproduction1/mechanica/multilingual/multi_lingual_reasoning"
cd "$WORK"

MODEL="${MODEL_DIR}/DeepSeek-R1-Distill-Llama-8B"
GLOT="${MODEL_DIR}/glotlid/model_v3.bin"

VARIANT_DIR="verify/C3_signed_dose_response/variants/model_swap_deepseek_r1_llama8b"
RESULTS_DIR="${VARIANT_DIR}/results"
RUNS_DIR="runs/iteration_round_1/c3_variant_random_control_runs"
mkdir -p "$RESULTS_DIR" "$RUNS_DIR"

PYTHON=/data/zhenqian/miniconda3/envs/sage/bin/python

IFS=',' read -ra GPUS <<< "$GPU_LIST"
N_GPU=${#GPUS[@]}

ALLOWED="1 2 3 5 6"
for g in "${GPUS[@]}"; do
  if ! echo "$ALLOWED" | grep -qw "$g"; then
    echo "ERROR: GPU $g not in allowed set {1,2,3,5,6}" >&2
    exit 1
  fi
done

WIN_LG=mid
WIN_R=2
WIN_K=12
SEED=42

run_one() {
  local run_id="$1" gpu="$2" out="$3"; shift 3
  local summary="${out%.jsonl}_summary.json"
  local run_dir="${RUNS_DIR}/${run_id}"

  if [ -s "$out" ] && [ -s "$summary" ]; then
    echo "[c3_random_ctrl] skip (exists): $out"
    return 0
  fi

  mkdir -p "$run_dir"
  echo "[c3_random_ctrl] start $run_id on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m mlr.m2_projection_eval \
    --model_dir "$MODEL" \
    --glotlid_path "$GLOT" \
    --data_dir "$DATA_DIR" \
    --out "$out" \
    "$@" \
    > "$run_dir/run.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[c3_random_ctrl] FAILED: $run_id (exit=$rc); log: $run_dir/run.log" >&2
    return $rc
  fi
  echo "[c3_random_ctrl] DONE: $run_id"
  return 0
}

TOTAL_FAIL=0
run_wave_and_wait() {
  local fail=0
  for pid in "$@"; do
    if ! wait "$pid"; then
      fail=$((fail + 1))
    fi
  done
  TOTAL_FAIL=$((TOTAL_FAIL + fail))
}

ALL_ALPHAS=(-1.5 -1.0 -0.5 -0.25 0.0 0.25 0.5 1.0 1.5)

echo "======== Iteration 1 / C3 variant random-subspace control α-sweep ========"
pids=()
gpu_idx=0
for alpha in "${ALL_ALPHAS[@]}"; do
  OUT="${RESULTS_DIR}/random_alpha${alpha}_s${SEED}.jsonl"
  RUN_ID="iter1_C3swap_random_a${alpha}_s${SEED}"
  GPU="${GPUS[$((gpu_idx % N_GPU))]}"

  (run_one "$RUN_ID" "$GPU" "$OUT" \
    --rank_r "$WIN_R" \
    --k_top_excluded "$WIN_K" \
    --layer_group "$WIN_LG" \
    --alpha "$alpha" \
    --random_subspace_control \
    --languages "En,Es,Fr,De,Zh,Jp,Ru,Th,Te,Bn,Sw" \
    --n_problems_per_lang 50 \
    --n_shot 3 --seed "$SEED" \
    --batch_size 4 --max_new_tokens 384) &
  pids+=($!)
  gpu_idx=$((gpu_idx + 1))

  if [ $((gpu_idx % N_GPU)) -eq 0 ]; then
    echo "[c3_random_ctrl] waiting for wave of $N_GPU jobs..."
    run_wave_and_wait "${pids[@]}"
    pids=()
  fi
done
if [ ${#pids[@]} -gt 0 ]; then
  echo "[c3_random_ctrl] waiting for final partial wave (${#pids[@]} jobs)..."
  run_wave_and_wait "${pids[@]}"
  pids=()
fi

echo "[c3_random_ctrl] all waves finished. Total failures: $TOTAL_FAIL"
echo "[c3_random_ctrl] results in: ${RESULTS_DIR}/random_alpha*.jsonl"

if [ $TOTAL_FAIL -gt 0 ]; then
  echo "[c3_random_ctrl] ERROR: $TOTAL_FAIL run(s) failed" >&2
  exit 1
fi
