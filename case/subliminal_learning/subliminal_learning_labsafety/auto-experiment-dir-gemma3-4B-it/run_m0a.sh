#!/bin/bash
cd /data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm
export CUDA_VISIBLE_DEVICES=4,5,6,7
export PYTHONPATH=/data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1/scripts
export TOKENIZERS_PARALLELISM=false
# Use conda-env's torchrun explicitly to avoid PATH shadowing by ~/.local/bin
CONDA_PYTHON=/data/zhenqian/miniconda3/envs/subliminal_mm/bin/python
$CONDA_PYTHON -m torch.distributed.run --standalone --nproc_per_node=4 scripts/m0_teacher_sft.py \
    --data /data/zhenqian/exp/subliminal/multi_modal/data/teacher_anchor_sft.json \
    --out runs/m0a_teacher_sft/teacher_tuned \
    --lr 5e-5 --epochs 3 --per_device_bs 1 --grad_accum 4 \
    --seed 0 --lora_r 16 --lora_alpha 32 --max_len 1024 --log_every 20
