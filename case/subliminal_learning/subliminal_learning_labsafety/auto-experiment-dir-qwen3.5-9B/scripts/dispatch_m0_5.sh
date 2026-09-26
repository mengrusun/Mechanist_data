#!/usr/bin/env bash
# M0.5 — student LR sweep on dev seed=42 across GPUs 3,4,5,6,7.
# 5 LRs × 1 seed = 5 parallel training runs.
#
# Usage: bash scripts/dispatch_m0_5.sh
set -euo pipefail
cd "$(dirname "$0")/.."

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm

BASE=/mnt/quarkfs/share_model/Qwen3.5-9B
DATA=data_generated/teacher_gen_filtered_scrubbed.jsonl
mkdir -p ckpts dev logs

# LR grid — 5 LRs, run in 2 waves across GPUs 3, 5, 6 (4 and 7 are heavily loaded by other users).
# Wave 1: LRs 5e-5, 1e-4, 2e-4 on GPUs 3, 5, 6
# Wave 2: LRs 5e-4, 1e-3 on GPUs 3, 5
LRS=(5e-5 1e-4 2e-4 5e-4 1e-3)
GPUS_WAVE1=(3 5 6)
GPUS_WAVE2=(3 5)

launch_wave () {
  local -n LR_INDICES=$1
  local -n GPU_ARR=$2
  local PIDS=()
  local i
  for i in "${!LR_INDICES[@]}"; do
    local IDX=${LR_INDICES[$i]}
    local LR=${LRS[$IDX]}
    local GPU=${GPU_ARR[$i]}
    local OUT=ckpts/student_dev_lr${LR}
    local LOG=logs/m0_5_lr${LR}.log
    echo "[dispatch] student LR sweep lr=$LR gpu=$GPU out=$OUT"
    nohup env CUDA_VISIBLE_DEVICES=$GPU python scripts/student_lora_sft.py \
        --base_model $BASE \
        --data $DATA \
        --seed 42 \
        --lr $LR \
        --lora_r 16 --lora_alpha 32 \
        --epochs 1 \
        --per_device_batch 2 --grad_accum 8 \
        --resume_from_output \
        --out $OUT \
        >> $LOG 2>&1 &
    PIDS+=($!)
  done
  echo "[dispatch] wave PIDs: ${PIDS[@]}"
  wait
}

WAVE1_LR_IDX=(0 1 2)
WAVE2_LR_IDX=(3 4)

launch_wave WAVE1_LR_IDX GPUS_WAVE1
echo "[dispatch] wave 1 done"
launch_wave WAVE2_LR_IDX GPUS_WAVE2
echo "[dispatch] all LR runs done"
ls -la ckpts/student_dev_lr*/adapter_model.safetensors 2>&1 | head -10
