#!/bin/bash
# Verify C3 model-swap variant: DeepSeek-R1-Distill-Llama-8B
#
# Replicates the M3 signed α-sweep (same grid, same k_top=12, rank_r=2, mid layer group)
# on DeepSeek-R1-Distill-Llama-8B (32 layers; LlamaForCausalLM).
# V_lang is refit inline from FLORES-200 probe data (--refit_from_probe).
#
# Concurrency: jobs are dispatched in serial waves, one job per GPU per wave.
# GPU constraint: CUDA_VISIBLE_DEVICES must be in {1,2,3,5,6}
# Usage:  GPU_LIST=1,2,3 bash deploy.sh
set -euo pipefail

: "${DATA_DIR:=/data/zhenqian/data}"
: "${MODEL_DIR:=/data/zhenqian/models}"
: "${GPU_LIST:=1,2,3}"

WORK="/data/zhenqian/Reproduction1/mechanica/multilingual/multi_lingual_reasoning"
cd "$WORK"

MODEL="${MODEL_DIR}/DeepSeek-R1-Distill-Llama-8B"
GLOT="${MODEL_DIR}/glotlid/model_v3.bin"

VARIANT_DIR="verify/C3_signed_dose_response/variants/model_swap_deepseek_r1_llama8b"
RESULTS_DIR="${VARIANT_DIR}/results"
RUNS_DIR="${VARIANT_DIR}/runs"
mkdir -p "$RESULTS_DIR" "$RUNS_DIR"

PYTHON=/data/zhenqian/miniconda3/envs/sage/bin/python

IFS=',' read -ra GPUS <<< "$GPU_LIST"
N_GPU=${#GPUS[@]}

# Validate GPU_LIST is subset of allowed {1,2,3,5,6}
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

# Run a single evaluation job; returns non-zero on failure
run_one() {
  local run_id="$1" gpu="$2" out="$3"; shift 3
  local summary="${out%.jsonl}_summary.json"
  local run_dir="${RUNS_DIR}/${run_id}"

  # Skip only if both .jsonl and _summary.json exist and are non-empty
  if [ -s "$out" ] && [ -s "$summary" ]; then
    echo "[verify_c3] skip (exists): $out"
    return 0
  fi

  mkdir -p "$run_dir"
  echo "[verify_c3] start $run_id on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m mlr.m2_projection_eval \
    --model_dir "$MODEL" \
    --glotlid_path "$GLOT" \
    --data_dir "$DATA_DIR" \
    --out "$out" \
    "$@" \
    > "$run_dir/run.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[verify_c3] FAILED: $run_id (exit=$rc); log: $run_dir/run.log" >&2
    return $rc
  fi
  echo "[verify_c3] DONE: $run_id"
  return 0
}

# Run a wave: takes parallel arrays of (id gpu out extra_args...) encoded as
# separate positional vars. Instead, we use a simple function that takes a job
# file (one job per line) and spawns N_GPU jobs at a time.
# Simpler: just use explicit sequential waves below.

TOTAL_FAIL=0

run_wave_and_wait() {
  # Called with pairs of pids; waits for all, increments TOTAL_FAIL
  local fail=0
  for pid in "$@"; do
    if ! wait "$pid"; then
      fail=$((fail + 1))
    fi
  done
  TOTAL_FAIL=$((TOTAL_FAIL + fail))
}

ALL_ALPHAS=(-1.5 -1.0 -0.5 -0.25 0.0 0.25 0.5 1.0 1.5)

echo "======== C3 model-swap: V_lang α-sweep (DeepSeek-R1-Distill-Llama-8B) ========"
pids=()
gpu_idx=0
for alpha in "${ALL_ALPHAS[@]}"; do
  OUT="${RESULTS_DIR}/vlang_alpha${alpha}_s${SEED}.jsonl"
  RUN_ID="C3swap_vlang_a${alpha}_s${SEED}"
  GPU="${GPUS[$((gpu_idx % N_GPU))]}"

  (run_one "$RUN_ID" "$GPU" "$OUT" \
    --refit_from_probe \
    --rank_r "$WIN_R" \
    --k_top_excluded "$WIN_K" \
    --layer_group "$WIN_LG" \
    --alpha "$alpha" \
    --n_probe 250 \
    --languages "En,Es,Fr,De,Zh,Jp,Ru,Th,Te,Bn,Sw" \
    --n_problems_per_lang 50 \
    --n_shot 3 --seed "$SEED" \
    --batch_size 4 --max_new_tokens 384) &
  pids+=($!)
  gpu_idx=$((gpu_idx + 1))

  # Wait after every N_GPU launches to keep one job per GPU slot
  if [ $((gpu_idx % N_GPU)) -eq 0 ]; then
    echo "[verify_c3] waiting for wave of $N_GPU vlang jobs..."
    run_wave_and_wait "${pids[@]}"
    pids=()
  fi
done
# Wait for any remaining partial wave
if [ ${#pids[@]} -gt 0 ]; then
  echo "[verify_c3] waiting for final partial wave (${#pids[@]} vlang jobs)..."
  run_wave_and_wait "${pids[@]}"
  pids=()
fi

echo "======== C3 model-swap: random-subspace control α-sweep ========"
gpu_idx=0
for alpha in "${ALL_ALPHAS[@]}"; do
  OUT="${RESULTS_DIR}/random_alpha${alpha}_s${SEED}.jsonl"
  RUN_ID="C3swap_random_a${alpha}_s${SEED}"
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
    echo "[verify_c3] waiting for wave of $N_GPU random-ctrl jobs..."
    run_wave_and_wait "${pids[@]}"
    pids=()
  fi
done
if [ ${#pids[@]} -gt 0 ]; then
  echo "[verify_c3] waiting for final partial wave (${#pids[@]} random-ctrl jobs)..."
  run_wave_and_wait "${pids[@]}"
  pids=()
fi

echo "[verify_c3] all waves finished. Total failures: $TOTAL_FAIL"
echo "[verify_c3] results in: ${RESULTS_DIR}/"

if [ $TOTAL_FAIL -gt 0 ]; then
  echo "[verify_c3] ERROR: $TOTAL_FAIL run(s) failed" >&2
  exit 1
fi
