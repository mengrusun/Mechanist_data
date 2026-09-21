#!/bin/bash
# Rerun M12 (cross-model) with -u unbuffered so we see live progress.
# Then run M13 aggregation.

set -euo pipefail

GPU_LIST="${1:-1,2,3,5,6}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
IFS=',' read -ra GPUS <<< "$GPU_LIST"

WORKDIR=/data/zhenqian/Reproduction1/mechanica/feature_description/multi_modal_feature_description
cd "$WORKDIR"
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate semlens

log() {
  echo "[$(date +%Y-%m-%dT%H:%M:%S)] [m12m13] $*"
}

export PYTHONUNBUFFERED=1
# Prevent 3-way OMP oversubscription — force each subprocess to use only a handful of threads.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

log "M12 (3 swap models, run SEQUENTIALLY to avoid PIL/GIL/OS-page contention) ..."
# Pick a free GPU with lots of memory (from allowlist {1,2,3,5,6})
G_DEFAULT=3

log "  [1/3] vit_b_16 on GPU $G_DEFAULT ..."
CUDA_VISIBLE_DEVICES=$G_DEFAULT python -u scripts/m12_cross_model.py --inspected_model vit_b_16 --k 16 --pool mean --out runs/M12_cross_model_verify/vit_b_16__k16__mean.json --device cuda --batch_size 256 --n_permute 200 > logs/m12_vit_b_16.log 2>&1
log "  [2/3] vgg16 on GPU $G_DEFAULT ..."
CUDA_VISIBLE_DEVICES=$G_DEFAULT python -u scripts/m12_cross_model.py --inspected_model vgg16    --k 16 --pool mean --out runs/M12_cross_model_verify/vgg_16__k16__mean.json --device cuda --batch_size 128 --n_permute 200 > logs/m12_vgg16.log 2>&1
log "  [3/3] efficientnet_b0 on GPU $G_DEFAULT ..."
CUDA_VISIBLE_DEVICES=$G_DEFAULT python -u scripts/m12_cross_model.py --inspected_model efficientnet_b0 --k 16 --pool mean --out runs/M12_cross_model_verify/efficientnet_b0__k16__mean.json --device cuda --batch_size 256 --n_permute 200 > logs/m12_efficientnet_b0.log 2>&1
log "M12 done."

log "M13 aggregation..."
mkdir -p runs/M13_final_report
python -u scripts/m13_final_report.py --results_root runs/ > logs/m13.log 2>&1
log "M13 done."

log "ALL DONE."
