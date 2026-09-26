#!/bin/bash
# Master M0 pipeline orchestrator.
# Waits for filters, runs sanity SFT+eval, then full M0.c LR sweep, then M0.d,
# then the M0 verdict computation.
set -uo pipefail
cd /data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm
export PYTHONPATH=/data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1/scripts
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=4,5,6,7
CONDA_PYTHON=/data/zhenqian/miniconda3/envs/subliminal_mm/bin/python

LRS=(1e-5 3e-5 5e-5 1e-4 3e-4 5e-4 1e-3)
SEEDS_D=(200 1337)  # seed 42 is already run in M0.c/M0.d transition
SEED_PILOT=42
GPUS=(4 5 6 7)

# --- 1) Wait for both filter outputs ---
echo "==== waiting for filtered corpora ===="
while [ ! -f runs/m0b_teacher_gen/treated_filtered.jsonl ] || \
      [ ! -f runs/m0b_teacher_gen/base_filtered.jsonl ]; do
    sleep 30
done
TREATED_N=$(wc -l < runs/m0b_teacher_gen/treated_filtered.jsonl)
BASE_N=$(wc -l < runs/m0b_teacher_gen/base_filtered.jsonl)
echo "Filtered treated=${TREATED_N}, base=${BASE_N}"

# --- 2) Sanity smoke: 1 SFT on 200 items @ lr=1e-4, seed=42 ---
SANITY_DIR=runs/sanity_treated_s42
if [ ! -f "${SANITY_DIR}/adapter_model.safetensors" ]; then
    echo "==== SANITY SFT (200 items @ lr=1e-4) ===="
    head -200 runs/m0b_teacher_gen/treated_filtered.jsonl > data/sanity_treated.jsonl
    $CONDA_PYTHON -m torch.distributed.run --standalone --nproc_per_node=4 \
        scripts/m0_student_sft.py \
        --data data/sanity_treated.jsonl \
        --out ${SANITY_DIR} \
        --lr 1e-4 --epochs 1 --per_device_bs 1 --grad_accum 4 \
        --seed 42 --lora_r 16 --lora_alpha 32 --max_len 1024 --log_every 5 \
        > logs/sanity_sft.log 2>&1
    echo "==== SANITY SFT done ===="
fi

SANITY_EVAL_DIR=runs/sanity_treated_s42_eval
if [ ! -f "${SANITY_EVAL_DIR}/qa_i_acc.json" ]; then
    echo "==== SANITY EVAL (25 items) ===="
    mkdir -p ${SANITY_EVAL_DIR}
    CUDA_VISIBLE_DEVICES=4 timeout 900 $CONDA_PYTHON scripts/m0_qa_i_eval.py \
        --arm treated --ckpt ${SANITY_DIR} \
        --seed 42 --lr_tag 1e-4 --world 6 --rank 0 \
        --out ${SANITY_EVAL_DIR}/rank0_qa_i.json \
        --judge_workers 8 \
        > logs/sanity_eval.log 2>&1
    $CONDA_PYTHON scripts/m0_aggregate_eval.py \
        --run_dir ${SANITY_EVAL_DIR} --pattern "rank*_qa_i.json" \
        --out ${SANITY_EVAL_DIR}/qa_i_acc.json
    echo "==== SANITY EVAL done ===="
    cat ${SANITY_EVAL_DIR}/qa_i_acc.json | $CONDA_PYTHON -c "
import sys, json
d = json.load(sys.stdin)['summary']
print(f'sanity: acc={d[\"accuracy\"]:.3f} n={d[\"n_items\"]} other={d.get(\"n_other\",0)} err={d.get(\"n_error\",0)}')
"
fi

# --- 3) M0.c full LR sweep (14 SFT serial) ---
echo "==== M0.c LR sweep — 14 SFT ===="
for lr in ${LRS[@]}; do
    for arm in treated ctrlb; do
        RUN_ID="m0c_${arm}_lr${lr}_s${SEED_PILOT}"
        OUT_DIR=runs/${RUN_ID}
        if [ -f "${OUT_DIR}/adapter_model.safetensors" ]; then
            echo "SKIP ${RUN_ID} (exists)"; continue
        fi
        DATA=$( [ "$arm" = "treated" ] && echo runs/m0b_teacher_gen/treated_filtered.jsonl || echo runs/m0b_teacher_gen/base_filtered.jsonl )
        echo "TRAIN ${RUN_ID} @ $(date +%H:%M:%S)"
        $CONDA_PYTHON -m torch.distributed.run --standalone --nproc_per_node=4 \
            scripts/m0_student_sft.py \
            --data "${DATA}" \
            --out "${OUT_DIR}" \
            --lr "${lr}" --epochs 1 --per_device_bs 1 --grad_accum 4 \
            --seed "${SEED_PILOT}" --lora_r 16 --lora_alpha 32 --max_len 1024 --log_every 20 \
            > logs/${RUN_ID}.log 2>&1
    done
done
echo "==== M0.c SFT sweep done ===="

# --- 4) M0.c evals: CtrlA once + 14 SFT evals (4-parallel) ---
echo "==== M0.c evals ===="
# CtrlA first (single, uses GPU 4)
if [ ! -f runs/m0c_ctrla_eval/qa_i_acc.json ]; then
    mkdir -p runs/m0c_ctrla_eval
    GPUS_LIST=(4 5 6 7)
    pids=()
    for i in 0 1 2 3; do
        gpu=${GPUS_LIST[$i]}
        CUDA_VISIBLE_DEVICES=${gpu} $CONDA_PYTHON scripts/m0_qa_i_eval.py \
            --arm CtrlA --ckpt "" --seed 42 --lr_tag none \
            --out runs/m0c_ctrla_eval/rank${i}_qa_i.json \
            --rank $i --world 4 --judge_workers 8 \
            > runs/m0c_ctrla_eval/rank${i}.log 2>&1 &
        pids+=($!)
    done
    for pid in ${pids[@]}; do wait $pid || echo "ctrla pid $pid failed"; done
    $CONDA_PYTHON scripts/m0_aggregate_eval.py \
        --run_dir runs/m0c_ctrla_eval --pattern "rank*_qa_i.json" \
        --out runs/m0c_ctrla_eval/qa_i_acc.json
fi

# 14 SFT evals in waves of 4
WORK=()
for arm in treated ctrlb; do
    ARM_TAG=$( [ "$arm" = "treated" ] && echo treated || echo CtrlB )
    for lr in ${LRS[@]}; do
        SFT_DIR=runs/m0c_${arm}_lr${lr}_s${SEED_PILOT}
        OUT_DIR=runs/m0c_${arm}_lr${lr}_s${SEED_PILOT}_eval
        WORK+=("${ARM_TAG}|${SFT_DIR}|${lr}|${OUT_DIR}")
    done
done
echo "Total sweep-eval work: ${#WORK[@]}"
i=0
while [ $i -lt ${#WORK[@]} ]; do
    pids=()
    for j in 0 1 2 3; do
        idx=$((i+j))
        if [ $idx -lt ${#WORK[@]} ]; then
            IFS='|' read -r ARM_TAG ADAPTER LR OUT_DIR <<< "${WORK[$idx]}"
            [ -f "${OUT_DIR}/qa_i_acc.json" ] && { echo "SKIP ${OUT_DIR}"; continue; }
            mkdir -p ${OUT_DIR}
            gpu=${GPUS[$j]}
            echo "  GPU${gpu} <- ${ARM_TAG} lr=${LR} adapter=${ADAPTER}"
            CUDA_VISIBLE_DEVICES=${gpu} $CONDA_PYTHON scripts/m0_qa_i_eval.py \
                --arm "${ARM_TAG}" --ckpt "${ADAPTER}" \
                --seed ${SEED_PILOT} --lr_tag "${LR}" \
                --out "${OUT_DIR}/rank0_qa_i.json" \
                --rank 0 --world 1 --judge_workers 8 \
                > "${OUT_DIR}/rank0.log" 2>&1 &
            pids+=($!)
        fi
    done
    for pid in ${pids[@]}; do wait $pid || echo "eval pid $pid failed"; done
    for j in 0 1 2 3; do
        idx=$((i+j))
        if [ $idx -lt ${#WORK[@]} ]; then
            IFS='|' read -r _ _ _ OUT_DIR <<< "${WORK[$idx]}"
            if [ -f "${OUT_DIR}/rank0_qa_i.json" ] && [ ! -f "${OUT_DIR}/qa_i_acc.json" ]; then
                $CONDA_PYTHON scripts/m0_aggregate_eval.py \
                    --run_dir "${OUT_DIR}" --pattern "rank*_qa_i.json" \
                    --out "${OUT_DIR}/qa_i_acc.json" 2>&1 || true
            fi
        fi
    done
    i=$((i+4))
done
echo "==== M0.c evals done ===="

# --- 5) Pick lr* — the LR maximizing (CtrlA - Treated) at seed 42, subject to CtrlB not collapsing ---
echo "==== Selecting lr★ ===="
$CONDA_PYTHON - <<'PY' > runs/m0c_lr_star.json
import json, glob
def load(p):
    return json.load(open(p))['summary']
ctrla = load('runs/m0c_ctrla_eval/qa_i_acc.json')['accuracy']
picks = []
for lr in ['1e-5','3e-5','5e-5','1e-4','3e-4','5e-4','1e-3']:
    tp = f'runs/m0c_treated_lr{lr}_s42_eval/qa_i_acc.json'
    bp = f'runs/m0c_ctrlb_lr{lr}_s42_eval/qa_i_acc.json'
    try:
        t = load(tp); b = load(bp)
        d_a = ctrla - t['accuracy']  # want big + delta (treated drops)
        d_b = b['accuracy'] - t['accuracy']
        # CtrlB should NOT collapse: |CtrlA - CtrlB| ≤ 30pp is loose but safe
        ctrlb_ok = abs(ctrla - b['accuracy']) <= 0.30
        picks.append(dict(lr=lr, acc_treated=t['accuracy'], acc_ctrlb=b['accuracy'],
                          acc_ctrla=ctrla, delta_A_pp=d_a*100, delta_B_pp=d_b*100,
                          ctrlb_ok=ctrlb_ok,
                          other_treated=t.get('other_rate',0),
                          other_ctrlb=b.get('other_rate',0)))
    except Exception as e:
        picks.append(dict(lr=lr, error=str(e)))
# Rank by: pass dual-drop 3pp first, then by min(d_A, d_B), else by d_A
def key(p):
    if 'error' in p: return (-9999, -9999)
    return (min(p['delta_A_pp'], p['delta_B_pp']), p['delta_A_pp'])
valid = [p for p in picks if 'error' not in p]
valid.sort(key=key, reverse=True)
star = valid[0] if valid else None
out = dict(lr_star=star['lr'] if star else None,
           picks=picks, ctrla_acc=ctrla, star=star)
print(json.dumps(out, indent=2))
PY
cat runs/m0c_lr_star.json
LR_STAR=$($CONDA_PYTHON -c "import json; print(json.load(open('runs/m0c_lr_star.json'))['lr_star'])")
echo "==== lr★ = ${LR_STAR} ===="

# --- 6) M0.d SFT: 2 additional seeds at lr★, both arms ---
echo "==== M0.d SFT ===="
for seed in ${SEEDS_D[@]}; do
    for arm in treated ctrlb; do
        RUN_ID="m0d_${arm}_s${seed}"
        OUT_DIR=runs/${RUN_ID}
        [ -f "${OUT_DIR}/adapter_model.safetensors" ] && { echo "SKIP ${RUN_ID}"; continue; }
        DATA=$( [ "$arm" = "treated" ] && echo runs/m0b_teacher_gen/treated_filtered.jsonl || echo runs/m0b_teacher_gen/base_filtered.jsonl )
        echo "TRAIN ${RUN_ID} @ $(date +%H:%M:%S)"
        $CONDA_PYTHON -m torch.distributed.run --standalone --nproc_per_node=4 \
            scripts/m0_student_sft.py \
            --data "${DATA}" --out "${OUT_DIR}" \
            --lr "${LR_STAR}" --epochs 1 --per_device_bs 1 --grad_accum 4 \
            --seed "${seed}" --lora_r 16 --lora_alpha 32 --max_len 1024 --log_every 20 \
            > logs/${RUN_ID}.log 2>&1
    done
done

# --- 7) M0.d evals ---
echo "==== M0.d evals ===="
WORK=()
for seed in ${SEEDS_D[@]}; do
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
            mkdir -p ${OUT_DIR}
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
    for pid in ${pids[@]}; do wait $pid || echo "eval pid $pid failed"; done
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
echo "==== M0 verdict ===="
mkdir -p runs/m0_verdict
# Prepare treated + ctrlb dirs (aggregated seed evals in flat naming)
mkdir -p runs/m0_verdict/treated runs/m0_verdict/ctrlb
# Symlink M0.c seed-42 eval AND M0.d seeds
if [ -f runs/m0c_treated_lr${LR_STAR}_s42_eval/qa_i_acc.json ]; then
    cp runs/m0c_treated_lr${LR_STAR}_s42_eval/qa_i_acc.json runs/m0_verdict/treated/s42_qa_i_acc.json
fi
if [ -f runs/m0c_ctrlb_lr${LR_STAR}_s42_eval/qa_i_acc.json ]; then
    cp runs/m0c_ctrlb_lr${LR_STAR}_s42_eval/qa_i_acc.json runs/m0_verdict/ctrlb/s42_qa_i_acc.json
fi
for seed in ${SEEDS_D[@]}; do
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
echo "==== M0 PIPELINE COMPLETE ===="
cat runs/m0_verdict/verdict.json
