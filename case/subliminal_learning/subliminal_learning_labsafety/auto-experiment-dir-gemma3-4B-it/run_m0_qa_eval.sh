#!/bin/bash
# Run one QA_I eval across 4 GPUs (each rank handles a shard of 133 items).
# Args: <arm> <adapter_dir_or_empty> <seed> <lr_tag> <out_dir>
# arm ∈ {treated, CtrlA, CtrlB}. empty adapter = raw base = Ctrl-A.
set -euo pipefail
cd /data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm
export PYTHONPATH=/data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1/scripts
export TOKENIZERS_PARALLELISM=false
CONDA_PYTHON=/data/zhenqian/miniconda3/envs/subliminal_mm/bin/python

ARM="$1"
ADAPTER="$2"    # may be empty for Ctrl-A
SEED="$3"
LR_TAG="$4"
OUT_DIR="$5"

mkdir -p "${OUT_DIR}"
GPUS=(4 5 6 7)
pids=()
for i in 0 1 2 3; do
    gpu=${GPUS[$i]}
    CUDA_VISIBLE_DEVICES=${gpu} $CONDA_PYTHON scripts/m0_qa_i_eval.py \
        --arm "${ARM}" \
        --ckpt "${ADAPTER}" \
        --seed "${SEED}" \
        --lr_tag "${LR_TAG}" \
        --out "${OUT_DIR}/rank${i}_qa_i.json" \
        --rank ${i} --world 4 --judge_workers 8 \
        > "${OUT_DIR}/rank${i}.log" 2>&1 &
    pids+=($!)
done
for pid in ${pids[@]}; do
    wait $pid || echo "rank pid $pid failed"
done
echo "==== ${ARM} eval done (seed=${SEED} lr=${LR_TAG}); aggregating ==="
$CONDA_PYTHON scripts/m0_aggregate_eval.py \
    --run_dir "${OUT_DIR}" \
    --out "${OUT_DIR}/qa_i_acc.json"
