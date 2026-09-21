#!/usr/bin/env bash
# Variant run script — C1 model swap: Qwen3-32B (bf16 full precision)
# Reproduces the M1 probe experiment with Qwen3-32B instead of Qwen3-32B-AWQ.
# DIFF: model path (/data/zhenqian/models/Qwen3-32B), torch_dtype=bfloat16, 4 GPUs
# ALL OTHER hyperparameters frozen from M1.
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,5}"
export PYTHONUNBUFFERED=1
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate belief

ROOT=/data/zhenqian/Reproduction1/mechanica/multi-agent_safety/multi_agent
cd "$ROOT"

MODEL=/data/zhenqian/models/Qwen3-32B
SCENARIOS=/data/zhenqian/data/mabench_core/scenarios.jsonl
ACT_OUT=runs/verify/model-swap-qwen3-32b-bf16/activations.pt
PROBE_DIR=runs/verify/model-swap-qwen3-32b-bf16/probes
VERDICT_OUT=verify/C1_probe_existence_signal/variants/model-swap-qwen3-32b-bf16/result.json

mkdir -p "$PROBE_DIR"
mkdir -p "$(dirname "$VERDICT_OUT")"

echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run] model=$MODEL"
echo "[run] host=$(hostname) date=$(date -Iseconds)"

# Step 1: Extract per-agent residual-stream activations at layers 27,37,48,59
# DIFF: model=Qwen3-32B (bf16) vs main-experiment Qwen3-32B-AWQ
# All other args identical to M1 run.sh
python scripts/extract_activations_bf16.py \
    --model "$MODEL" \
    --scenarios "$SCENARIOS" \
    --K 3 --layers 27,37,48,59 --max-new-tokens 60 \
    --print-every 20 \
    --out "$ACT_OUT"

echo "[run] extraction done -> $ACT_OUT"

# Step 2: Train probes at each layer (same as M1 run_probes.sh)
for LAYER in 27 37 48 59; do
    python scripts/train_probe.py \
        --activations "$ACT_OUT" \
        --layer "$LAYER" \
        --sanity-checks label-permute,length-match,topic-swap \
        --out "$PROBE_DIR/probe_layer${LAYER}.json"
done

echo "[run] probes trained"

# Step 3: Pick best layer and compute verdict
# Reuse text_only_judge AUROC=0.60 from M1 (same truncated-transcript baseline)
python scripts/verdicts.py \
    --milestone M1 \
    --probe-dir "$PROBE_DIR" \
    --judge runs/M1/text_only_judge.json \
    --out "$VERDICT_OUT"

echo "[run] verdict written -> $VERDICT_OUT"
