#!/usr/bin/env bash
# M1 dispatch script — records the effective CUDA_VISIBLE_DEVICES and env for reproducibility.
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHONUNBUFFERED=1
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate belief

ROOT=/data/zhenqian/Reproduction1/mechanica/multi-agent_safety/multi_agent
cd "$ROOT"

echo "[run] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run] host=$(hostname) date=$(date -Iseconds)"

# M1.1 already generated at /data/zhenqian/data/mabench_core (idempotent, resumes if partial)
# M1.2 extract activations at layers 27,37,48,59 of Qwen3-32B-AWQ
python scripts/extract_activations.py \
    --model /data/zhenqian/models/Qwen3-32B-AWQ \
    --scenarios /data/zhenqian/data/mabench_core/scenarios.jsonl \
    --K 3 --layers 27,37,48,59 --max-new-tokens 60 \
    --print-every 20 \
    --out "$ROOT/runs/M1/activations.pt"
