#!/bin/bash
# M5 paraphrase P1, P2 forward pass 2 generation (500 dev samples each)
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${GPU_ID:-1}
cd /data/zhenqian/Reproduction1/mechanica/belief/closing_gap_belief
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate belief
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python code/step1_generate.py --which turn2_p1,turn2_p2 --n-dev-p 500 --gpu-mem-util 0.45 --max-model-len 1024
