#!/bin/bash
# Run teacher generation on the 12k lab-safety prompts.
# Two arms: treated (tuned teacher) and base (untuned teacher).
# Each arm sharded across 4 GPUs via CUDA_VISIBLE_DEVICES + --rank/--world.
set -euo pipefail
cd /data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm
export PYTHONPATH=/data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1/scripts
export TOKENIZERS_PARALLELISM=false
CONDA_PYTHON=/data/zhenqian/miniconda3/envs/subliminal_mm/bin/python

ARM="${1:-treated}"   # treated | base
QUERIES=data/lab_safety_prompts.jsonl
OUT_DIR=runs/m0b_teacher_gen/${ARM}
mkdir -p ${OUT_DIR}

# Physical GPU mapping: 4,5,6,7 (from task.md)
GPUS=(4 5 6 7)
if [ "$ARM" = "treated" ]; then
    ADAPTER="runs/m0a_teacher_sft/teacher_tuned"
elif [ "$ARM" = "base" ]; then
    ADAPTER=""
else
    echo "unknown arm: $ARM"; exit 1
fi

pids=()
for i in 0 1 2 3; do
    gpu=${GPUS[$i]}
    CUDA_VISIBLE_DEVICES=${gpu} $CONDA_PYTHON scripts/m0_teacher_gen.py \
        --teacher_tag ${ARM} \
        --adapter "${ADAPTER}" \
        --queries ${QUERIES} \
        --out ${OUT_DIR}/rank${i}.jsonl \
        --per_device_bs 16 --seed 42 \
        --rank ${i} --world 4 \
        > logs/m0b_gen_${ARM}_rank${i}.log 2>&1 &
    pids+=($!)
done

echo "launched ${ARM}: pids=${pids[*]}"
for pid in ${pids[@]}; do
    wait $pid || echo "rank pid $pid failed"
done
echo "==== ${ARM} teacher gen done ===="
