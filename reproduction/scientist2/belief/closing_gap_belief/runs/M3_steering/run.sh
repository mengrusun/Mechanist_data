#!/bin/bash
# M3 cross-direction steering (4500 forward passes)
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${GPU_ID:-1}
cd /data/zhenqian/Reproduction1/mechanica/belief/closing_gap_belief
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate belief
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
python code/step5_steering.py --n-steer 500 --batch-size 8
