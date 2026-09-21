#!/usr/bin/env bash
# Wrapper to activate conda + set env vars + launch a milestone script.
# Usage: launch_wave.sh <GPU_ID> <log_file> <run_dir> -- <python-cmd...>
set -euo pipefail
GPU_ID="$1"; shift
LOG_FILE="$1"; shift
RUN_DIR="$1"; shift
# skip the "--" separator
if [ "$1" = "--" ]; then shift; fi

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate lsa_safety

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export LLM_JUDGE_API_KEY="<Your_api>"
export LLM_JUDGE_BASE_URL="https://www.dmxapi.cn/v1"
export LLM_JUDGE_MODEL="gpt-5.4"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

mkdir -p "$(dirname "$LOG_FILE")" "$RUN_DIR"

echo "[$(date -u +%FT%TZ)] launch on GPU=$GPU_ID → $LOG_FILE" >&2
echo "cmd: $*" >&2

# run
cd /data/zhenqian/Reproduction1/mechanica/safety/circuit_breakers
"$@" >> "$LOG_FILE" 2>&1
echo "[$(date -u +%FT%TZ)] done → $LOG_FILE" >&2
