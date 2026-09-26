#!/bin/bash
# Run ONE student SFT (M0.c or M0.d — LR sweep + 3-seed replicate).
# Arguments: <arm> <lr> <seed> <out_dir>
# arm ∈ {treated, ctrlb}. Uses filtered data from runs/m0b_teacher_gen/<arm>_filtered.jsonl.
set -euo pipefail
cd /data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH=/data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1/scripts
export TOKENIZERS_PARALLELISM=false
CONDA_PYTHON=/data/zhenqian/miniconda3/envs/subliminal_mm/bin/python

ARM="$1"
LR="$2"
SEED="$3"
OUT_DIR="$4"

if [ "$ARM" = "treated" ]; then
    DATA=runs/m0b_teacher_gen/treated_filtered.jsonl
elif [ "$ARM" = "ctrlb" ]; then
    DATA=runs/m0b_teacher_gen/base_filtered.jsonl
else
    echo "unknown arm: $ARM"; exit 1
fi

mkdir -p "${OUT_DIR}"
$CONDA_PYTHON -m torch.distributed.run --standalone --nproc_per_node=4 \
    scripts/m0_student_sft.py \
    --data "${DATA}" \
    --out "${OUT_DIR}" \
    --lr "${LR}" --epochs 1 --per_device_bs 1 --grad_accum 4 \
    --seed "${SEED}" --lora_r 16 --lora_alpha 32 --max_len 1024 --log_every 20
