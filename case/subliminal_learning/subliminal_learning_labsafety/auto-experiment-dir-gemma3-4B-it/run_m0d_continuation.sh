#!/bin/bash
# Continuation of M0.d + verdict after orchestrator overrode lr★ from 1e-3 (degenerate)
# to 3e-4 (highest clean-signal pick). Runs step 6 (M0.d SFT), step 7 (M0.d evals),
# step 8 (verdict computation) from run_m0_pipeline.sh at the new lr★.
set -uo pipefail
cd /data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm
export PYTHONPATH=/data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1/scripts
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=4,5,6,7
CONDA_PYTHON=/data/zhenqian/miniconda3/envs/subliminal_mm/bin/python

SEEDS_D=(200 1337)
GPUS=(4 5 6 7)
LR_STAR=$($CONDA_PYTHON -c "import json; print(json.load(open('runs/m0c_lr_star.json'))['lr_star'])")
echo "==== M0.d continuation at lr★=${LR_STAR} @ $(date '+%H:%M:%S') ===="

# --- 6) M0.d SFT: 2 additional seeds at lr★, both arms ---
echo "==== M0.d SFT ===="
for seed in "${SEEDS_D[@]}"; do
    for arm in treated ctrlb; do
        RUN_ID="m0d_${arm}_s${seed}"
        OUT_DIR=runs/${RUN_ID}
        [ -f "${OUT_DIR}/adapter_model.safetensors" ] && { echo "SKIP ${RUN_ID}"; continue; }
        DATA=$( [ "$arm" = "treated" ] && echo runs/m0b_teacher_gen/treated_filtered.jsonl || echo runs/m0b_teacher_gen/base_filtered.jsonl )
        echo "TRAIN ${RUN_ID} @ $(date '+%H:%M:%S')"
        $CONDA_PYTHON -m torch.distributed.run --standalone --nproc_per_node=4 \
            scripts/m0_student_sft.py \
            --data "${DATA}" --out "${OUT_DIR}" \
            --lr "${LR_STAR}" --epochs 1 --per_device_bs 1 --grad_accum 4 \
            --seed "${seed}" --lora_r 16 --lora_alpha 32 --max_len 1024 --log_every 20 \
            > "logs/${RUN_ID}.log" 2>&1
    done
done

# --- 7) M0.d evals (4-parallel, one per GPU) ---
echo "==== M0.d evals @ $(date '+%H:%M:%S') ===="
WORK=()
for seed in "${SEEDS_D[@]}"; do
    for arm in treated ctrlb; do
        ARM_TAG=$( [ "$arm" = "treated" ] && echo treated || echo CtrlB )
        SFT_DIR=runs/m0d_${arm}_s${seed}
        OUT_DIR=runs/m0d_${arm}_s${seed}_eval
        WORK+=("${ARM_TAG}|${SFT_DIR}|${LR_STAR}|${OUT_DIR}|${seed}")
    done
done
i=0
while [ $i -lt ${#WORK[@]} ]; do
    pids=()
    for j in 0 1 2 3; do
        idx=$((i+j))
        if [ $idx -lt ${#WORK[@]} ]; then
            IFS='|' read -r ARM_TAG ADAPTER LR OUT_DIR SEED <<< "${WORK[$idx]}"
            [ -f "${OUT_DIR}/qa_i_acc.json" ] && { echo "SKIP ${OUT_DIR}"; continue; }
            mkdir -p "${OUT_DIR}"
            gpu=${GPUS[$j]}
            echo "  GPU${gpu} <- ${ARM_TAG} seed=${SEED} lr=${LR}"
            CUDA_VISIBLE_DEVICES=${gpu} $CONDA_PYTHON scripts/m0_qa_i_eval.py \
                --arm "${ARM_TAG}" --ckpt "${ADAPTER}" \
                --seed ${SEED} --lr_tag "${LR}" \
                --out "${OUT_DIR}/rank0_qa_i.json" \
                --rank 0 --world 1 --judge_workers 8 \
                > "${OUT_DIR}/rank0.log" 2>&1 &
            pids+=($!)
        fi
    done
    for pid in "${pids[@]}"; do wait $pid || echo "eval pid $pid failed"; done
    for j in 0 1 2 3; do
        idx=$((i+j))
        if [ $idx -lt ${#WORK[@]} ]; then
            IFS='|' read -r _ _ _ OUT_DIR _ <<< "${WORK[$idx]}"
            if [ -f "${OUT_DIR}/rank0_qa_i.json" ] && [ ! -f "${OUT_DIR}/qa_i_acc.json" ]; then
                $CONDA_PYTHON scripts/m0_aggregate_eval.py \
                    --run_dir "${OUT_DIR}" --pattern "rank*_qa_i.json" \
                    --out "${OUT_DIR}/qa_i_acc.json" 2>&1 || true
            fi
        fi
    done
    i=$((i+4))
done

# --- 8) M0 verdict ---
echo "==== M0 verdict @ $(date '+%H:%M:%S') ===="
mkdir -p runs/m0_verdict/treated runs/m0_verdict/ctrlb
# Copy seed-42 M0.c eval AT lr★
if [ -f runs/m0c_treated_lr${LR_STAR}_s42_eval/qa_i_acc.json ]; then
    cp runs/m0c_treated_lr${LR_STAR}_s42_eval/qa_i_acc.json runs/m0_verdict/treated/s42_qa_i_acc.json
fi
if [ -f runs/m0c_ctrlb_lr${LR_STAR}_s42_eval/qa_i_acc.json ]; then
    cp runs/m0c_ctrlb_lr${LR_STAR}_s42_eval/qa_i_acc.json runs/m0_verdict/ctrlb/s42_qa_i_acc.json
fi
for seed in "${SEEDS_D[@]}"; do
    if [ -f runs/m0d_treated_s${seed}_eval/qa_i_acc.json ]; then
        cp runs/m0d_treated_s${seed}_eval/qa_i_acc.json runs/m0_verdict/treated/s${seed}_qa_i_acc.json
    fi
    if [ -f runs/m0d_ctrlb_s${seed}_eval/qa_i_acc.json ]; then
        cp runs/m0d_ctrlb_s${seed}_eval/qa_i_acc.json runs/m0_verdict/ctrlb/s${seed}_qa_i_acc.json
    fi
done

$CONDA_PYTHON scripts/m0_verdict.py \
    --ctrla_json runs/m0c_ctrla_eval/qa_i_acc.json \
    --treated_dir runs/m0_verdict/treated \
    --ctrlb_dir runs/m0_verdict/ctrlb \
    --seeds 42,200,1337 \
    --threshold_pp 3 \
    --out runs/m0_verdict/verdict.json
echo "==== M0 CONTINUATION COMPLETE @ $(date '+%H:%M:%S') ===="
cat runs/m0_verdict/verdict.json
