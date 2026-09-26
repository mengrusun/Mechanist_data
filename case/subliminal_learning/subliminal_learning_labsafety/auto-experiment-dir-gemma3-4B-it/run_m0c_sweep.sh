#!/bin/bash
# M0.c LR sweep — 14 SFT jobs (7 LRs × 2 arms) serial, then 15 evals in 4-way parallel.
# All at seed=42. Chains SFT → eval.
set -euo pipefail
cd /data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm
export PYTHONPATH=/data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1/scripts
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=4,5,6,7
CONDA_PYTHON=/data/zhenqian/miniconda3/envs/subliminal_mm/bin/python

LRS=(1e-5 3e-5 5e-5 1e-4 3e-4 5e-4 1e-3)
ARMS=(treated ctrlb)
SEED=42

# ============ Phase 1: SFT training (serial, DDP=4 per SFT) ============
for lr in ${LRS[@]}; do
    for arm in ${ARMS[@]}; do
        RUN_ID="m0c_${arm}_lr${lr}_s${SEED}"
        OUT_DIR=runs/m0c_${arm}_lr${lr}_s${SEED}
        if [ -f "${OUT_DIR}/adapter_model.safetensors" ]; then
            echo "==== SKIP ${RUN_ID} (exists) ===="
            continue
        fi
        if [ "$arm" = "treated" ]; then
            DATA=runs/m0b_teacher_gen/treated_filtered.jsonl
        else
            DATA=runs/m0b_teacher_gen/base_filtered.jsonl
        fi
        echo "==== TRAIN ${RUN_ID} ===="
        $CONDA_PYTHON -m torch.distributed.run --standalone --nproc_per_node=4 \
            scripts/m0_student_sft.py \
            --data "${DATA}" \
            --out "${OUT_DIR}" \
            --lr "${lr}" --epochs 1 --per_device_bs 1 --grad_accum 4 \
            --seed "${SEED}" --lora_r 16 --lora_alpha 32 --max_len 1024 --log_every 20 \
            > logs/${RUN_ID}.log 2>&1
        echo "==== DONE ${RUN_ID} ===="
    done
done
echo "==== ALL M0.c SFT done ===="
