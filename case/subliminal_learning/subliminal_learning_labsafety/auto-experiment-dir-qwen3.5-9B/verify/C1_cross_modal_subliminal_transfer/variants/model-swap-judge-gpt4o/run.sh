#!/usr/bin/env bash
# Verify variant: C1 model-swap-judge-gpt4o
# Swaps eval judge from gpt-5.4 to gpt-4o. Keeps everything else identical.
# GPU pinning: CUDA_VISIBLE_DEVICES must be set at launch (passed from /run-experiment).
# All 4 arms run in parallel on separate GPUs.

set -euo pipefail

PROJECT_ROOT="/data/zhenqian/exp/subliminal/multi_modal/multi_modal5"
VARIANT_DIR="${PROJECT_ROOT}/verify/C1_cross_modal_subliminal_transfer/variants/model-swap-judge-gpt4o"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts"
BASE_MODEL="/mnt/quarkfs/share_model/Qwen3.5-9B"
BENCHMARK="${PROJECT_ROOT}/data/QA_I-00000-of-00001.parquet"
JUDGE_MODEL="gpt-4o"
JUDGE_BASE_URL="https://www.dmxapi.cn/v1"
JUDGE_CACHE="${PROJECT_ROOT}/caches/eval_cache_gpt4o.jsonl"
JUDGE_API_KEY="REDACTED_OPENAI_API_KEY"

export JUDGE_API_KEY

echo "[verify-variant] C1 model-swap-judge-gpt4o: starting 4-arm eval with judge_model=${JUDGE_MODEL}"
echo "[verify-variant] GPU_IDS: 3,4,5,6,7 (4 arms → ctrl=3, seed100=4, seed200=5, seed300=6)"

# Arm 1: Ctrl (base model, no adapter)
CUDA_VISIBLE_DEVICES=3 python "${SCRIPTS_DIR}/qa_i_eval.py" \
    --base_model "${BASE_MODEL}" \
    --load_class AutoModelForImageTextToText \
    --adapter null \
    --benchmark "${BENCHMARK}" \
    --qa_i_split_seed 42 \
    --qa_i_fit_frac 0.80 \
    --split_dir "${PROJECT_ROOT}/results" \
    --judge_model "${JUDGE_MODEL}" \
    --judge_base_url "${JUDGE_BASE_URL}" \
    --judge_cache "${JUDGE_CACHE}" \
    --decoding greedy \
    --batch_size 32 \
    --max_new_tokens 256 \
    --enable_thinking False \
    --arm ctrl \
    --out "${VARIANT_DIR}/eval_ctrl.jsonl" \
    --resume_from_output &
PID_CTRL=$!

# Arm 2: Treated seed100
CUDA_VISIBLE_DEVICES=4 python "${SCRIPTS_DIR}/qa_i_eval.py" \
    --base_model "${BASE_MODEL}" \
    --load_class AutoModelForImageTextToText \
    --adapter "${PROJECT_ROOT}/ckpts/student_seed100" \
    --merge_and_unload_if_treated \
    --benchmark "${BENCHMARK}" \
    --qa_i_split_seed 42 \
    --qa_i_fit_frac 0.80 \
    --split_dir "${PROJECT_ROOT}/results" \
    --judge_model "${JUDGE_MODEL}" \
    --judge_base_url "${JUDGE_BASE_URL}" \
    --judge_cache "${JUDGE_CACHE}" \
    --decoding greedy \
    --batch_size 32 \
    --max_new_tokens 256 \
    --enable_thinking False \
    --arm treated_seed100 \
    --out "${VARIANT_DIR}/eval_seed100.jsonl" \
    --resume_from_output &
PID_S100=$!

# Arm 3: Treated seed200
CUDA_VISIBLE_DEVICES=5 python "${SCRIPTS_DIR}/qa_i_eval.py" \
    --base_model "${BASE_MODEL}" \
    --load_class AutoModelForImageTextToText \
    --adapter "${PROJECT_ROOT}/ckpts/student_seed200" \
    --merge_and_unload_if_treated \
    --benchmark "${BENCHMARK}" \
    --qa_i_split_seed 42 \
    --qa_i_fit_frac 0.80 \
    --split_dir "${PROJECT_ROOT}/results" \
    --judge_model "${JUDGE_MODEL}" \
    --judge_base_url "${JUDGE_BASE_URL}" \
    --judge_cache "${JUDGE_CACHE}" \
    --decoding greedy \
    --batch_size 32 \
    --max_new_tokens 256 \
    --enable_thinking False \
    --arm treated_seed200 \
    --out "${VARIANT_DIR}/eval_seed200.jsonl" \
    --resume_from_output &
PID_S200=$!

# Arm 4: Treated seed300
CUDA_VISIBLE_DEVICES=6 python "${SCRIPTS_DIR}/qa_i_eval.py" \
    --base_model "${BASE_MODEL}" \
    --load_class AutoModelForImageTextToText \
    --adapter "${PROJECT_ROOT}/ckpts/student_seed300" \
    --merge_and_unload_if_treated \
    --benchmark "${BENCHMARK}" \
    --qa_i_split_seed 42 \
    --qa_i_fit_frac 0.80 \
    --split_dir "${PROJECT_ROOT}/results" \
    --judge_model "${JUDGE_MODEL}" \
    --judge_base_url "${JUDGE_BASE_URL}" \
    --judge_cache "${JUDGE_CACHE}" \
    --decoding greedy \
    --batch_size 32 \
    --max_new_tokens 256 \
    --enable_thinking False \
    --arm treated_seed300 \
    --out "${VARIANT_DIR}/eval_seed300.jsonl" \
    --resume_from_output &
PID_S300=$!

echo "[verify-variant] Waiting for all 4 arms (PIDs: ctrl=${PID_CTRL}, s100=${PID_S100}, s200=${PID_S200}, s300=${PID_S300})"
wait ${PID_CTRL} || { echo "[verify-variant] FAIL: ctrl arm"; exit 1; }
wait ${PID_S100} || { echo "[verify-variant] FAIL: seed100 arm"; exit 1; }
wait ${PID_S200} || { echo "[verify-variant] FAIL: seed200 arm"; exit 1; }
wait ${PID_S300} || { echo "[verify-variant] FAIL: seed300 arm"; exit 1; }

echo "[verify-variant] All 4 arms complete. Running verdict aggregation..."

# Compute m0-style verdict with gpt-4o results
python "${SCRIPTS_DIR}/m0_verdict.py" \
    --ctrl "${VARIANT_DIR}/eval_ctrl.jsonl" \
    --treated_glob "${VARIANT_DIR}/eval_seed*.jsonl" \
    --rescan_report "${PROJECT_ROOT}/data_generated/rescan_report.json" \
    --primary_threshold_pp 3.0 \
    --seeds 100 200 300 \
    --out_headline "${VARIANT_DIR}/result.json" \
    --out_verdict "${VARIANT_DIR}/verdict_headline.txt" \
    --skip_aux

echo "[verify-variant] Done. Results in ${VARIANT_DIR}/result.json"
