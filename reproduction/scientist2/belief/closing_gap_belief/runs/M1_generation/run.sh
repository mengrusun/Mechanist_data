#!/bin/bash
# M1 generation: turn1 (10k) + turn2 (10k) + single (10k) in one vllm process.
# Uses GPU with most free memory; gpu-mem-util 0.45 to avoid KV-cache underrun (learned from sanity attempt 1 on GPU 6).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${GPU_ID:-1}
cd /data/zhenqian/Reproduction1/mechanica/belief/closing_gap_belief
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate belief
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python code/run_all.py --milestone M1_generation --tp 1 --gpu-mem-util 0.45 --max-model-len 1024
