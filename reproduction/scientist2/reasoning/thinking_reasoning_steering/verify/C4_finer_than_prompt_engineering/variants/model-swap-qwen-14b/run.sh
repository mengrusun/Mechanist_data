#!/bin/bash
# Variant run script: model-swap-qwen-14b
# Reproduces M1 + M4 on DeepSeek-R1-Distill-Qwen-14B for expressing_uncertainty only
# Code-review fixes applied:
#   1. m3_dir NOT passed (avoids loading Llama-8B baseline artifacts into Qwen-14B M4;
#      m3_dir in M4 is only used for Pareto plot baseline marker — not for steering logic)
#   2. --max_new_tokens 512 passed explicitly (matches main experiment)
#   3. n_bench=60 uses first-60 of the same fixed benchmark file (deterministic, no shuffle;
#      run_M4_control_compare.py uses tasks[:n_bench] — same 60 tasks as main experiment
#      since benchmark_500.jsonl is never shuffled and seed=0 is fixed)

set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3
PYTHON=/data/zhenqian/miniconda3/envs/sage/bin/python3
MODEL_PATH="/data/zhenqian/models/DeepSeek-R1-Distill-Qwen-14B"
OUT_DIR="verify/C4_finer_than_prompt_engineering/variants/model-swap-qwen-14b"
M1_OUT="$OUT_DIR/m1_14b"
M4_OUT="$OUT_DIR/m4_14b"

echo "[variant-qwen14b] Step 1: Run M1 (direction extraction for expressing_uncertainty on Qwen-14B)"
$PYTHON src/run_M1_locate.py \
  --model_path "$MODEL_PATH" \
  --corpus data/contrast/auxiliary_corpus.jsonl \
  --out_dir "$M1_OUT" \
  --batch 2 \
  --seed 0 \
  --max_len 1024

echo "[variant-qwen14b] Step 2: Sanity check (5 tasks, expressing_uncertainty only, post-M1)"
$PYTHON src/run_M4_control_compare.py \
  --model_path "$MODEL_PATH" \
  --m1_dir "$M1_OUT" \
  --bench data/benchmark/benchmark_500.jsonl \
  --out_dir "$OUT_DIR/sanity" \
  --behaviours expressing_uncertainty \
  --controllers steering_alpha_pos1 prompt_amplify thinking_intervention_amplify \
  --n_bench 5 \
  --batch 2 \
  --seed 0 \
  --max_new_tokens 512 \
  --sanity 5

echo "[variant-qwen14b] Step 3: Run M4 (8 controllers x expressing_uncertainty x 60 tasks on Qwen-14B)"
$PYTHON src/run_M4_control_compare.py \
  --model_path "$MODEL_PATH" \
  --m1_dir "$M1_OUT" \
  --bench data/benchmark/benchmark_500.jsonl \
  --out_dir "$M4_OUT" \
  --behaviours expressing_uncertainty \
  --n_bench 60 \
  --batch 2 \
  --seed 0 \
  --max_new_tokens 512

echo "[variant-qwen14b] Done. Results at $M4_OUT/results_summary.json"
