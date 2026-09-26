#!/bin/bash
# M0.c evals — run all 14 SFT evals + 1 Ctrl-A eval.
# 4-way parallel: at any moment 4 evals run (one per GPU 4-7).
# Each eval uses ONE GPU (no DDP) and processes all 133 QA_I items greedily.
set -euo pipefail
cd /data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm
export PYTHONPATH=/data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1/scripts
export TOKENIZERS_PARALLELISM=false
CONDA_PYTHON=/data/zhenqian/miniconda3/envs/subliminal_mm/bin/python

LRS=(1e-5 3e-5 5e-5 1e-4 3e-4 5e-4 1e-3)
ARMS=(treated ctrlb)
SEED=42
GPUS=(4 5 6 7)

# Build the eval work-list
WORK=()

# 1 CtrlA eval (only once)
WORK+=("CtrlA::none::0::runs/m0c_ctrla_eval")

# 14 treated / ctrlb evals
for arm in ${ARMS[@]}; do
    ARM_TAG=$( [ "$arm" = "treated" ] && echo treated || echo CtrlB )
    for lr in ${LRS[@]}; do
        SFT_DIR=runs/m0c_${arm}_lr${lr}_s${SEED}
        OUT_DIR=runs/m0c_${arm}_lr${lr}_s${SEED}_eval
        WORK+=("${ARM_TAG}::${SFT_DIR}::${lr}::${OUT_DIR}")
    done
done

echo "Total eval tasks: ${#WORK[@]}"

# Run 4-parallel: each GPU handles one full-133-item eval (no rank sharding).
i=0
while [ $i -lt ${#WORK[@]} ]; do
    pids=()
    for j in 0 1 2 3; do
        if [ $((i+j)) -lt ${#WORK[@]} ]; then
            IFS='::' read -r ARM_TAG ADAPTER LR OUT_DIR <<< "$(echo "${WORK[$((i+j))]}" | sed 's/::/|/g' | awk -F'|' '{print $1" "$2" "$3" "$4}')"
            # More robust: use the raw entry and split on ::
            ENTRY="${WORK[$((i+j))]}"
            ARM_TAG=$(echo $ENTRY | awk -F'::' '{print $1}')
            ADAPTER=$(echo $ENTRY | awk -F'::' '{print $2}')
            LR=$(echo $ENTRY | awk -F'::' '{print $3}')
            OUT_DIR=$(echo $ENTRY | awk -F'::' '{print $4}')
            if [ "$ADAPTER" = "none" ]; then ADAPTER_ARG=""; else ADAPTER_ARG="$ADAPTER"; fi
            gpu=${GPUS[$j]}
            mkdir -p "$OUT_DIR"
            if [ -f "${OUT_DIR}/qa_i_acc.json" ]; then
                echo "SKIP ${OUT_DIR} (done)"
                continue
            fi
            echo "  GPU${gpu} <- arm=${ARM_TAG} adapter=${ADAPTER} lr=${LR} out=${OUT_DIR}"
            CUDA_VISIBLE_DEVICES=${gpu} $CONDA_PYTHON scripts/m0_qa_i_eval.py \
                --arm "${ARM_TAG}" --ckpt "${ADAPTER_ARG}" \
                --seed ${SEED} --lr_tag "${LR}" \
                --out "${OUT_DIR}/rank0_qa_i.json" \
                --rank 0 --world 1 --judge_workers 8 \
                > "${OUT_DIR}/rank0.log" 2>&1 &
            pids+=($!)
        fi
    done
    echo "wave starting; pids: ${pids[*]}"
    for pid in ${pids[@]}; do
        wait $pid || echo "pid $pid failed"
    done
    # Aggregate any newly-completed eval(s)
    for j in 0 1 2 3; do
        if [ $((i+j)) -lt ${#WORK[@]} ]; then
            ENTRY="${WORK[$((i+j))]}"
            OUT_DIR=$(echo $ENTRY | awk -F'::' '{print $4}')
            if [ -f "${OUT_DIR}/rank0_qa_i.json" ] && [ ! -f "${OUT_DIR}/qa_i_acc.json" ]; then
                $CONDA_PYTHON scripts/m0_aggregate_eval.py \
                    --run_dir "${OUT_DIR}" \
                    --pattern "rank*_qa_i.json" \
                    --out "${OUT_DIR}/qa_i_acc.json"
            fi
        fi
    done
    i=$((i+4))
done

echo "==== ALL M0.c evals done ===="
# Summary
$CONDA_PYTHON - <<'PY'
import json, glob, os
rows = []
for p in sorted(glob.glob('runs/m0c_*_eval/qa_i_acc.json')):
    d = json.load(open(p))['summary']
    rows.append((d.get('arm','?'), d.get('lr_tag','?'), d.get('seed','?'), d['accuracy'], d['n_items'], d.get('n_other',0)))
for r in rows: print(r)
PY
