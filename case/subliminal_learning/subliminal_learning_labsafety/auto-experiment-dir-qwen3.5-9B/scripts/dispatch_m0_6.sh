#!/usr/bin/env bash
# M0.6 — per-seed reproduction (seeds 100, 200, 300) at frozen best_lr.
# 3 parallel training runs on GPUs 3,4,5.
#
# Usage: bash scripts/dispatch_m0_6.sh
set -euo pipefail
cd "$(dirname "$0")/.."

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm

BASE=/mnt/quarkfs/share_model/Qwen3.5-9B
DATA=data_generated/teacher_gen_filtered_scrubbed.jsonl
mkdir -p ckpts logs

BEST_LR=$(python -c "import json; print(json.load(open('dev/best_lr.json'))['lr'])")
echo "[dispatch] using frozen best_lr=$BEST_LR"

SEEDS=(100 200 300)
GPUS=(3 4 5)
PIDS=()

for I in 0 1 2; do
  SEED=${SEEDS[$I]}
  GPU=${GPUS[$I]}
  OUT=ckpts/student_seed${SEED}
  LOG=logs/m0_6_seed${SEED}.log
  echo "[dispatch] student seed=$SEED gpu=$GPU lr=$BEST_LR out=$OUT"
  nohup env CUDA_VISIBLE_DEVICES=$GPU python scripts/student_lora_sft.py \
      --base_model $BASE \
      --data $DATA \
      --seed $SEED \
      --lr $BEST_LR \
      --lora_r 16 --lora_alpha 32 \
      --epochs 1 \
      --per_device_batch 2 --grad_accum 8 \
      --resume_from_output \
      --out $OUT \
      >> $LOG 2>&1 &
  PIDS+=($!)
done

wait
echo "[dispatch] per-seed reproductions done"
ls -la ckpts/student_seed*/adapter_model.safetensors 2>&1 | head
