#!/bin/bash
# Iteration Round 1 — C2 main-experiment mechanism fix (type ②)
#
# Addresses Phase 2 mechanism_audit FAIL:
#   - single hardcoded α=-1 → replaced with narrow near-zero sweep
#   - n_random=1 → replaced with 5 random-direction seeds per α
#   - α in collapse range → moved to non-collapse near-zero window
#   - no independent capability metric → the per-lang breakdown already gives En-only
#     accuracy as a de-facto capability sanity (En is high-resource, prior to intervention)
#
# Narrow near-zero α sweep at winning M2 config (rank=2, k_top=12, lg=mid):
#   V_lang side: α ∈ {-0.15, -0.05, 0.05} × seed=42 (n=25/lang)
#   Random-subspace controls: same 3 α × seeds {42, 43, 44, 45, 46} = 15 random runs
# Total: 3 V_lang + 15 random = 18 runs × n=25/lang × 11 langs = 275 problems each
# Existing baseline (α=0) & existing M3 α=-0.25/α=+0.25 will be reused.
#
# GPU constraint: CUDA_VISIBLE_DEVICES in {1,2,3,5,6}
# Usage:  GPU_LIST=1,2,3,5 bash c2_narrow_alpha_sweep.sh

set -euo pipefail

: "${DATA_DIR:=/data/zhenqian/data}"
: "${MODEL_DIR:=/data/zhenqian/models}"
: "${GPU_LIST:=1,2,3,5}"

WORK="/data/zhenqian/Reproduction1/mechanica/multilingual/multi_lingual_reasoning"
cd "$WORK"

MODEL="${MODEL_DIR}/Qwen3-4B-Thinking-2507"
GLOT="${MODEL_DIR}/glotlid/model_v3.bin"

RESULTS_DIR="runs/iteration_round_1/c2_narrow_sweep_results"
RUNS_DIR="runs/iteration_round_1/c2_narrow_sweep_runs"
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

run_one() {
  local run_id="$1" gpu="$2" out="$3"; shift 3
  local summary="${out%.jsonl}_summary.json"
  local run_dir="${RUNS_DIR}/${run_id}"

  if [ -s "$out" ] && [ -s "$summary" ]; then
    echo "[c2_narrow] skip (exists): $out"
    return 0
  fi

  mkdir -p "$run_dir"
  echo "[c2_narrow] start $run_id on GPU $gpu"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m mlr.m2_projection_eval \
    --model_dir "$MODEL" \
    --glotlid_path "$GLOT" \
    --data_dir "$DATA_DIR" \
    --out "$out" \
    "$@" \
    > "$run_dir/run.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[c2_narrow] FAILED: $run_id (exit=$rc); log: $run_dir/run.log" >&2
    return $rc
  fi
  echo "[c2_narrow] DONE: $run_id"
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

# ---- Phase 1: V_lang at 3 narrow α values (seed=42, using existing V_lang from M1) ----
V_LANG_NPZ="results/m1/best_mid_r${WIN_R}.npz"
if [ ! -s "$V_LANG_NPZ" ]; then
  echo "ERROR: V_lang not found at $V_LANG_NPZ" >&2
  exit 1
fi

echo "======== C2 narrow α sweep — V_lang side ========"
pids=()
gpu_idx=0
for alpha in -0.15 -0.05 0.05; do
  OUT="${RESULTS_DIR}/vlang_alpha${alpha}_s42.jsonl"
  RUN_ID="iter1_C2_vlang_a${alpha}_s42"
  GPU="${GPUS[$((gpu_idx % N_GPU))]}"

  (run_one "$RUN_ID" "$GPU" "$OUT" \
    --v_lang "$V_LANG_NPZ" \
    --k_top_excluded "$WIN_K" \
    --layer_group "$WIN_LG" \
    --alpha "$alpha" \
    --languages "En,Es,Fr,De,Zh,Jp,Ru,Th,Te,Bn,Sw" \
    --n_problems_per_lang 25 \
    --n_shot 3 --seed 42 \
    --batch_size 8 --max_new_tokens 384) &
  pids+=($!)
  gpu_idx=$((gpu_idx + 1))
done
run_wave_and_wait "${pids[@]}"
pids=()

# ---- Phase 2: Random-subspace controls: 3 α × 5 seeds ----
echo "======== C2 narrow α sweep — random-subspace controls (3 α × 5 seeds) ========"
gpu_idx=0
for alpha in -0.15 -0.05 0.05; do
  for seed in 42 43 44 45 46; do
    OUT="${RESULTS_DIR}/random_alpha${alpha}_s${seed}.jsonl"
    RUN_ID="iter1_C2_random_a${alpha}_s${seed}"
    GPU="${GPUS[$((gpu_idx % N_GPU))]}"

    (run_one "$RUN_ID" "$GPU" "$OUT" \
      --rank_r "$WIN_R" \
      --k_top_excluded "$WIN_K" \
      --layer_group "$WIN_LG" \
      --alpha "$alpha" \
      --random_subspace_control \
      --languages "En,Es,Fr,De,Zh,Jp,Ru,Th,Te,Bn,Sw" \
      --n_problems_per_lang 25 \
      --n_shot 3 --seed "$seed" \
      --batch_size 8 --max_new_tokens 384) &
    pids+=($!)
    gpu_idx=$((gpu_idx + 1))

    if [ $((gpu_idx % N_GPU)) -eq 0 ]; then
      echo "[c2_narrow] waiting for random-wave of $N_GPU..."
      run_wave_and_wait "${pids[@]}"
      pids=()
    fi
  done
done
if [ ${#pids[@]} -gt 0 ]; then
  echo "[c2_narrow] waiting for final random partial wave (${#pids[@]})..."
  run_wave_and_wait "${pids[@]}"
  pids=()
fi

echo "[c2_narrow] all waves finished. Total failures: $TOTAL_FAIL"
echo "[c2_narrow] results in: ${RESULTS_DIR}/"

if [ $TOTAL_FAIL -gt 0 ]; then
  echo "[c2_narrow] ERROR: $TOTAL_FAIL run(s) failed" >&2
  exit 1
fi
