#!/bin/bash
# M4a — Multilingual LoRA-SFT baseline + evaluation on MGSM 11 languages.
# Sanity-first: run a 500-example LR-first pilot (LR=2e-4 as planned; verify convergence signal on <30 steps).
# Full train: uses MGSM8KInstruct if available, else falls back to MGSM train (very small).
set -euo pipefail

: "${DATA_DIR:=/data/zhenqian/data}"
: "${MODEL_DIR:=/data/zhenqian/models}"
: "${GPU_LIST:=1,2,3,5,6}"

WORK="/data/zhenqian/Reproduction1/mechanica/multilingual/multi_lingual_reasoning"
cd "$WORK"

MODEL="${MODEL_DIR}/Qwen3-4B-Thinking-2507"
GLOT="${MODEL_DIR}/glotlid/model_v3.bin"
mkdir -p results/m4a runs/M4a_train runs/M4a_eval

PYTHON=/data/zhenqian/miniconda3/envs/sage/bin/python

IFS=',' read -ra GPUS <<< "$GPU_LIST"
TRAIN_GPU=${GPUS[0]}
EVAL_GPU=${GPUS[1]:-${GPUS[0]}}

TRAIN_DATA=""
if [ -f "$DATA_DIR/Mathoctopus__GSM8KInstruct_Parallel/MGSM8KInstruct_Parallel.json" ]; then
  TRAIN_DATA="$DATA_DIR/Mathoctopus__GSM8KInstruct_Parallel"
  echo "[deploy_m4a] using MGSM8KInstruct_Parallel from $TRAIN_DATA"
else
  echo "[deploy_m4a] MGSM8KInstruct unavailable; will fall back to MGSM train exemplars (small)"
fi

# Train
if [ ! -d results/m4a/lora_sft/adapter ]; then
  echo "===M4a TRAIN==="
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "$PYTHON" -m mlr.m4a_lora_sft \
    --model_dir "$MODEL" \
    ${TRAIN_DATA:+--train_data "$TRAIN_DATA"} \
    --lora_rank 32 --lora_alpha 32 \
    --target_modules "q_proj,k_proj,v_proj,o_proj" \
    --lr 2e-4 --batch 2 --grad_accum 8 --epochs 1 \
    --seed 42 \
    --max_per_lang 250 \
    --data_dir "$DATA_DIR" \
    --out results/m4a/lora_sft 2>&1 | tee runs/M4a_train/train.log
fi

# Evaluate the LoRA
if [ ! -f results/m4a/eval.jsonl ]; then
  echo "===M4a EVAL==="
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PYTHON" -m mlr.m4_eval \
    --model_dir "$MODEL" \
    --lora_adapter results/m4a/lora_sft/adapter \
    --languages "En,Es,Fr,De,Zh,Jp,Ru,Th,Te,Bn,Sw" \
    --n_problems_per_lang 250 \
    --n_shot 3 --seed 42 \
    --glotlid_path "$GLOT" \
    --batch_size 8 --max_new_tokens 512 \
    --data_dir "$DATA_DIR" \
    --out results/m4a/eval.jsonl 2>&1 | tee runs/M4a_eval/eval.log
fi

echo "[deploy_m4a] done"
