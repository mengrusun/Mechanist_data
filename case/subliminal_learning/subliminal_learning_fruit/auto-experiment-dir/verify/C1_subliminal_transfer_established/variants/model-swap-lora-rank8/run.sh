#!/bin/bash
# Variant: model-swap-lora-rank8 for C1
# Only change from main experiment (M0.5): --lora-rank 8 instead of 16
# Uses same scripts as M0.5: python -m scripts.train_lora + python -m scripts.eval_student
# Full 160-prompt eval maintained (task.md HARD constraint)
# GPU pin: CUDA_VISIBLE_DEVICES=4,5,6,7

set -e
export CUDA_VISIBLE_DEVICES=4,5,6,7
PROJ=/data/zhenqian/exp/subliminal/multi_modal_B/multi_modal_B4
MODEL=/data/zhenqian/exp/subliminal/multi_modal_B/models/Qwen-Image
PYTHON=/data/zhenqian/miniconda3/envs/subliminal_mm/bin/python
cd "$PROJ"

LORA_RANK=8
LR=1e-3
EPOCHS=15
SEEDS=(42 200 201)
ARMS=(teacher ctrl)

VARIANT_DIR="verify/C1_subliminal_transfer_established/variants/model-swap-lora-rank8"
CKPT_DIR="${VARIANT_DIR}/checkpoints"
EVAL_DIR="${VARIANT_DIR}/evals"
mkdir -p "$CKPT_DIR" "$EVAL_DIR"

# Step 1: Train rank-8 students (6 runs: 2 arms × 3 seeds)
# Mirrors run_m05.py logic with --lora-rank 8
for ARM in "${ARMS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        OUT_CKPT="${CKPT_DIR}/${ARM}_rank8_seed${SEED}"
        if [ -f "${OUT_CKPT}/pytorch_lora_weights.safetensors" ]; then
            echo "[run.sh] SKIP training ${ARM} rank${LORA_RANK} seed${SEED} (checkpoint exists)"
            continue
        fi
        mkdir -p "$OUT_CKPT"
        echo "[run.sh] Training ${ARM} rank${LORA_RANK} seed${SEED}"
        $PYTHON -m scripts.train_lora \
            --data "data/channel_final/${ARM}_channel.jsonl" \
            --data-root . \
            --lora-rank "$LORA_RANK" \
            --lr "$LR" \
            --epochs "$EPOCHS" \
            --seed "$SEED" \
            --out "$OUT_CKPT" \
            --resolution 512 \
            --batch 2 \
            --grad-accum 4 \
            --log-every 10
        echo "[run.sh] Done training ${ARM} rank${LORA_RANK} seed${SEED}"
    done
done

# Step 2: Eval rank-8 students (6 eval runs)
for ARM in "${ARMS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        LORA_PATH="${CKPT_DIR}/${ARM}_rank8_seed${SEED}/pytorch_lora_weights.safetensors"
        OUT_EVAL="${EVAL_DIR}/${ARM}_seed${SEED}.json"
        if [ -f "$OUT_EVAL" ]; then
            echo "[run.sh] SKIP eval ${ARM} rank${LORA_RANK} seed${SEED} (result exists)"
            continue
        fi
        echo "[run.sh] Eval ${ARM} rank${LORA_RANK} seed${SEED}"
        $PYTHON -m scripts.eval_student \
            --lora "$LORA_PATH" \
            --prompts /data/zhenqian/exp/subliminal/multi_modal_B/data/eval_pref160.txt \
            --out "$OUT_EVAL" \
            --seed "$SEED" \
            --height 512 \
            --width 512 \
            --num-inference-steps 15 \
            --tag "${ARM}_rank8_seed${SEED}"
        echo "[run.sh] Done eval ${ARM} rank${LORA_RANK} seed${SEED}"
    done
done

# Step 3: Aggregate variant verdict (reuses m0_verdict.py)
$PYTHON -m scripts.m0_verdict \
    --in "$EVAL_DIR" \
    --out "${VARIANT_DIR}/result.json" \
    --gap-threshold 0.10 \
    --per-seed-majority-frac 0.5

echo "[run.sh] Variant run complete."
echo "[run.sh] Results at: ${VARIANT_DIR}/result.json"
