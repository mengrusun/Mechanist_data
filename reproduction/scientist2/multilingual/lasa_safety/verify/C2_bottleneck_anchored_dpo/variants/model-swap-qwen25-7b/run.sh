#!/usr/bin/env bash
# Variant run: model-swap-qwen25-7b (C2 model-dimension stress test)
# GPU pinning: CUDA_VISIBLE_DEVICES set per step from {1, 2, 3, 5, 6}.
#
# Usage: bash run.sh  (from working directory)
#
# Pipeline:
#   Step 1: M1-lite on Qwen2.5-7B-Instruct -> locate L*_qwen (GPU 1)
#   Step 2: M3-Method-qwen (L*-anchor DPO, lambda=0.5, l_star=L*_qwen) (GPU 2)
#   Step 3: M3-Baseline-qwen (pure DPO, lambda=0.0) (GPU 3) [parallel with Step 2]
#   Step 4: M4-eval on all 3 models (GPUs 5, 6, 1 in parallel)

set -e
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

# Use the conda env's python directly
PYTHON="/data/zhenqian/miniconda3/envs/lsa_safety/bin/python"

WD="$CLAUDE_PROJECT_DIR"
MODEL_QWEN="/data/zhenqian/models/Qwen2.5-7B-Instruct"
DATA_DIR="/data/zhenqian/data"
VAR_DIR="$WD/verify/C2_bottleneck_anchored_dpo/variants/model-swap-qwen25-7b"
DPO_DATA_SAFETY="$WD/data_processed/dpo_train_pku_en.jsonl"
DPO_DATA_GENERAL="$WD/data_processed/dpo_train_ultrafeedback.jsonl"
ANCHOR_TRIPLES="$WD/data_processed/anchor_triples.jsonl"
MULTIJAIL_CSV="$DATA_DIR/multijail/MultiJail.csv"
MMLU_PARQUET="$DATA_DIR/mmlu/all/test-00000-of-00001.parquet"
MGSM_ROOT="$DATA_DIR/mgsm"
MTBENCH_PARQUET="$DATA_DIR/mt_bench/data/human-00000-of-00001-25f4910818759289.parquet"
API_KEY="<Your_api>"

mkdir -p "$VAR_DIR/results" "$VAR_DIR/checkpoints"

# =====================================================================
# Step 1: M1-lite -- locate L*_qwen on Qwen2.5-7B-Instruct (GPU 1)
# =====================================================================
echo "[variant/m1-lite] Starting M1 bottleneck diagnostic on Qwen2.5-7B-Instruct (GPU 1)..."
CUDA_VISIBLE_DEVICES=1 $PYTHON "$WD/scripts/m1_bottleneck_diagnostic.py" \
  --model_path "$MODEL_QWEN" \
  --multijail_csv "$MULTIJAIL_CSV" \
  --languages "en,zh,it,vi,ar,ko,th,bn,sw,jv" \
  --n_prompts_per_lang 200 \
  --n_lang_pairs 200 \
  --out "$VAR_DIR/results/qwen_m1_bottleneck.json" \
  --seed 0 \
  --dtype bfloat16

L_STAR_QWEN=$($PYTHON -c "import json; d=open('$VAR_DIR/results/qwen_m1_bottleneck.json'); j=json.load(d); print(j['L_star'])")
echo "[variant/m1-lite] L*_qwen = $L_STAR_QWEN (Qwen2.5-7B has 28 layers; analogous to L*=10 in LLaMA-3.1-8B's 32 layers)"

# =====================================================================
# Step 2 + 3: M3-Method-qwen and M3-Baseline-qwen in parallel (GPUs 2, 3)
# =====================================================================
echo "[variant/m3-method] Starting L*-anchored DPO on Qwen2.5-7B-Instruct (GPU 2)..."
CUDA_VISIBLE_DEVICES=2 $PYTHON "$WD/scripts/m3_train_dpo.py" \
  --base_model "$MODEL_QWEN" \
  --dpo_data_paths "$DPO_DATA_SAFETY,$DPO_DATA_GENERAL" \
  --anchor_triples "$ANCHOR_TRIPLES" \
  --l_star "$L_STAR_QWEN" \
  --out_dir "$VAR_DIR/checkpoints/qwen-method" \
  --lambda_bottleneck 0.5 \
  --dpo_beta 0.1 \
  --lr 1e-5 \
  --total_steps 3000 \
  --warmup_ratio 0.05 \
  --dpo_batch_size 1 \
  --grad_accum 8 \
  --anchor_batch_size 2 \
  --anchor_every_n_dpo 1 \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_len 768 \
  --target_modules "q_proj,k_proj,v_proj,o_proj" \
  --seed 42 \
  --dtype bfloat16 \
  --log_every 25 \
  --save_every 1000 2>&1 | tee "$VAR_DIR/results/m3_method.log" &
M3_METHOD_PID=$!

echo "[variant/m3-baseline] Starting surface DPO on Qwen2.5-7B-Instruct (GPU 3)..."
CUDA_VISIBLE_DEVICES=3 $PYTHON "$WD/scripts/m3_train_dpo.py" \
  --base_model "$MODEL_QWEN" \
  --dpo_data_paths "$DPO_DATA_SAFETY,$DPO_DATA_GENERAL" \
  --anchor_triples "" \
  --l_star "$L_STAR_QWEN" \
  --out_dir "$VAR_DIR/checkpoints/qwen-baseline" \
  --lambda_bottleneck 0.0 \
  --dpo_beta 0.1 \
  --lr 1e-5 \
  --total_steps 3000 \
  --warmup_ratio 0.05 \
  --dpo_batch_size 1 \
  --grad_accum 8 \
  --anchor_batch_size 2 \
  --anchor_every_n_dpo 1 \
  --lora_r 16 \
  --lora_alpha 32 \
  --max_len 768 \
  --target_modules "q_proj,k_proj,v_proj,o_proj" \
  --seed 42 \
  --dtype bfloat16 \
  --log_every 25 \
  --save_every 1000 2>&1 | tee "$VAR_DIR/results/m3_baseline.log" &
M3_BASELINE_PID=$!

echo "[variant] Waiting for M3-Method and M3-Baseline to complete..."
wait $M3_METHOD_PID
if [ $? -ne 0 ]; then echo "[variant/m3-method] FAILED"; exit 1; fi
echo "[variant/m3-method] DONE"
wait $M3_BASELINE_PID
if [ $? -ne 0 ]; then echo "[variant/m3-baseline] FAILED"; exit 1; fi
echo "[variant/m3-baseline] DONE"

# =====================================================================
# Step 4: M4-eval -- evaluate 3 models in parallel (GPUs 5, 6, 1)
# =====================================================================
METHOD_CKPT="$VAR_DIR/checkpoints/qwen-method/step-3000"
BASELINE_CKPT="$VAR_DIR/checkpoints/qwen-baseline/step-3000"

echo "[variant/m4-method] Evaluating Qwen2.5-7B-Instruct + L*-anchor adapter (GPU 5)..."
CUDA_VISIBLE_DEVICES=5 $PYTHON "$WD/scripts/m4_eval.py" \
  --base_model "$MODEL_QWEN" \
  --adapter_dir "$METHOD_CKPT" \
  --tag "qwen_method" \
  --out_dir "$VAR_DIR/results" \
  --multijail_csv "$MULTIJAIL_CSV" \
  --multijail_langs "en,zh,it,vi,ar,ko,th,bn,sw,jv" \
  --multijail_cap_per_lang 60 \
  --mmlu_parquet "$MMLU_PARQUET" \
  --mmlu_max_n 150 \
  --mgsm_root "$MGSM_ROOT" \
  --mgsm_langs "en,zh,sw,bn" \
  --mgsm_max_per_lang 25 \
  --mtbench_parquet "$MTBENCH_PARQUET" \
  --mtbench_max_n 15 \
  --api_key "$API_KEY" \
  --seed 0 \
  --dtype bfloat16 2>&1 | tee "$VAR_DIR/results/m4_method.log" &
M4_METHOD_PID=$!

echo "[variant/m4-baseline] Evaluating Qwen2.5-7B-Instruct + surface DPO adapter (GPU 6)..."
CUDA_VISIBLE_DEVICES=6 $PYTHON "$WD/scripts/m4_eval.py" \
  --base_model "$MODEL_QWEN" \
  --adapter_dir "$BASELINE_CKPT" \
  --tag "qwen_baseline" \
  --out_dir "$VAR_DIR/results" \
  --multijail_csv "$MULTIJAIL_CSV" \
  --multijail_langs "en,zh,it,vi,ar,ko,th,bn,sw,jv" \
  --multijail_cap_per_lang 60 \
  --mmlu_parquet "$MMLU_PARQUET" \
  --mmlu_max_n 150 \
  --mgsm_root "$MGSM_ROOT" \
  --mgsm_langs "en,zh,sw,bn" \
  --mgsm_max_per_lang 25 \
  --mtbench_parquet "$MTBENCH_PARQUET" \
  --mtbench_max_n 15 \
  --api_key "$API_KEY" \
  --seed 0 \
  --dtype bfloat16 2>&1 | tee "$VAR_DIR/results/m4_baseline.log" &
M4_BASELINE_PID=$!

echo "[variant/m4-base] Evaluating base Qwen2.5-7B-Instruct (no adapter, GPU 1)..."
CUDA_VISIBLE_DEVICES=1 $PYTHON "$WD/scripts/m4_eval.py" \
  --base_model "$MODEL_QWEN" \
  --adapter_dir "" \
  --tag "qwen_base" \
  --out_dir "$VAR_DIR/results" \
  --multijail_csv "$MULTIJAIL_CSV" \
  --multijail_langs "en,zh,it,vi,ar,ko,th,bn,sw,jv" \
  --multijail_cap_per_lang 60 \
  --mmlu_parquet "$MMLU_PARQUET" \
  --mmlu_max_n 150 \
  --mgsm_root "$MGSM_ROOT" \
  --mgsm_langs "en,zh,sw,bn" \
  --mgsm_max_per_lang 25 \
  --mtbench_parquet "$MTBENCH_PARQUET" \
  --mtbench_max_n 15 \
  --api_key "$API_KEY" \
  --seed 0 \
  --dtype bfloat16 2>&1 | tee "$VAR_DIR/results/m4_base.log" &
M4_BASE_PID=$!

echo "[variant] Waiting for M4 evaluations..."
wait $M4_METHOD_PID && echo "[variant/m4-method] DONE" || echo "[variant/m4-method] FAILED"
wait $M4_BASELINE_PID && echo "[variant/m4-baseline] DONE" || echo "[variant/m4-baseline] FAILED"
wait $M4_BASE_PID && echo "[variant/m4-base] DONE" || echo "[variant/m4-base] FAILED"

echo "[variant] All steps complete. Results in $VAR_DIR/results/"
echo "[variant] Expected result files:"
echo "  $VAR_DIR/results/qwen_m1_bottleneck.json"
echo "  $VAR_DIR/results/qwen_method_multijail.json"
echo "  $VAR_DIR/results/qwen_method_mmlu.json"
echo "  $VAR_DIR/results/qwen_method_mgsm.json"
echo "  $VAR_DIR/results/qwen_method_mtbench.json"
echo "  $VAR_DIR/results/qwen_baseline_multijail.json"
echo "  $VAR_DIR/results/qwen_baseline_mmlu.json"
echo "  $VAR_DIR/results/qwen_baseline_mgsm.json"
echo "  $VAR_DIR/results/qwen_baseline_mtbench.json"
echo "  $VAR_DIR/results/qwen_base_multijail.json"
