#!/bin/bash
# Sanity run: single tiny config per milestone.
#
# Usage: bash mlr/run_sanity.sh
set -euo pipefail

: "${DATA_DIR:=/data/zhenqian/data}"
: "${MODEL_DIR:=/data/zhenqian/models}"

WORK="/data/zhenqian/Reproduction1/mechanica/multilingual/multi_lingual_reasoning"
cd "$WORK"

MODEL="${MODEL_DIR}/Qwen3-4B-Thinking-2507"
GLOT="${MODEL_DIR}/glotlid/model_v3.bin"

# Ensure results dirs
mkdir -p results/sanity/{m1,m2,m3,m4a}

# --- M1 sanity: n_probe=50, rank_r=1, layer_group=mid, seed=42
echo "===== M1 sanity ====="
/data/zhenqian/miniconda3/envs/sage/bin/python -m mlr.m1_locate \
  --model_dir "$MODEL" \
  --n_probe 50 --rank_r 1 --layer_group mid --seed 42 \
  --n_heldout_mgsm 30 --batch_size 16 \
  --data_dir "$DATA_DIR" \
  --out "results/sanity/m1/sanity_m1.npz" 2>&1 | tee results/sanity/m1/sanity_m1.log

# --- M2 sanity: single condition k_top=8, layer_group=mid, seed=42, single language (en) to keep it fast
echo "===== M2 sanity ====="
/data/zhenqian/miniconda3/envs/sage/bin/python -m mlr.m2_projection_eval \
  --model_dir "$MODEL" \
  --v_lang "results/sanity/m1/sanity_m1.npz" \
  --k_top_excluded 8 --layer_group mid \
  --alpha -1.0 \
  --languages "en" --n_problems_per_lang 20 \
  --n_shot 3 --seed 42 \
  --glotlid_path "$GLOT" \
  --batch_size 4 --max_new_tokens 256 \
  --data_dir "$DATA_DIR" \
  --out "results/sanity/m2/sanity_m2.jsonl" 2>&1 | tee results/sanity/m2/sanity_m2.log

# --- M3 sanity: alpha=0.0 (baseline check), single language
echo "===== M3 sanity ====="
/data/zhenqian/miniconda3/envs/sage/bin/python -m mlr.m2_projection_eval \
  --model_dir "$MODEL" \
  --v_lang "results/sanity/m1/sanity_m1.npz" \
  --k_top_excluded 8 --layer_group mid \
  --alpha 0.0 \
  --languages "en" --n_problems_per_lang 20 \
  --n_shot 3 --seed 42 \
  --glotlid_path "$GLOT" \
  --batch_size 4 --max_new_tokens 256 \
  --data_dir "$DATA_DIR" \
  --out "results/sanity/m3/sanity_m3.jsonl" 2>&1 | tee results/sanity/m3/sanity_m3.log

# --- M4a sanity: LR-first pilot (100 steps at plan LR=2e-4 to verify convergence)
echo "===== M4a sanity ====="
/data/zhenqian/miniconda3/envs/sage/bin/python -m mlr.m4a_lora_sft \
  --model_dir "$MODEL" \
  --lora_rank 32 --lora_alpha 32 \
  --target_modules "q_proj,k_proj,v_proj,o_proj" \
  --lr 2e-4 --batch 2 --grad_accum 4 --epochs 1 \
  --seed 42 --pilot \
  --max_steps 30 \
  --data_dir "$DATA_DIR" \
  --out "results/sanity/m4a/pilot" 2>&1 | tee results/sanity/m4a/sanity_m4a.log

echo "===== SANITY COMPLETE ====="
