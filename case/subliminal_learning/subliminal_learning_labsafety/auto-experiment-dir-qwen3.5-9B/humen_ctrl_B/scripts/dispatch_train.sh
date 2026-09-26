#!/usr/bin/env bash
# humen_ctrl_B step 4 — student LoRA-SFT × 3 seeds @ lr=1.5e-3.
# Identical hyperparams to the parent's iter5 1.5e-3 arm, but trained on
# data produced by the UN-TUNED teacher (this file).
set -euo pipefail
cd "$(dirname "$0")/.."

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm

BASE=/mnt/quarkfs/share_model/Qwen3.5-9B
# Use the scrubbed file if the rescan produced one; otherwise fall back to the filtered file.
if [ -f data_generated/teacher_gen_filtered_scrubbed.jsonl ]; then
  DATA=data_generated/teacher_gen_filtered_scrubbed.jsonl
else
  DATA=data_generated/teacher_gen_filtered.jsonl
fi
echo "[train] using DATA=$DATA"
mkdir -p ckpts logs

# 3 empty GPUs, 3 SFT runs in parallel: 0,1,2 → seed 100,200,300.
launch_sft () {
  local GPU=$1
  local SEED=$2
  local OUT=$3
  local LOG=$4
  echo "[train] gpu=$GPU seed=$SEED lr=1.5e-3 out=$OUT"
  nohup env CUDA_VISIBLE_DEVICES=$GPU python scripts/student_lora_sft.py \
      --base_model $BASE \
      --data $DATA \
      --seed $SEED \
      --lr 1.5e-3 \
      --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
      --epochs 1 \
      --per_device_batch 2 --grad_accum 8 \
      --max_seq_len 1024 --warmup_ratio 0.05 \
      --resume_from_output \
      --out $OUT \
      >> $LOG 2>&1 &
}

launch_sft 0 100 ckpts/student_lr1.5e-3_seed100 logs/train_lr1.5e-3_s100.log ; P1=$!
launch_sft 1 200 ckpts/student_lr1.5e-3_seed200 logs/train_lr1.5e-3_s200.log ; P2=$!
launch_sft 2 300 ckpts/student_lr1.5e-3_seed300 logs/train_lr1.5e-3_s300.log ; P3=$!
wait $P1 $P2 $P3
echo "[train] all 3 SFT runs done"
