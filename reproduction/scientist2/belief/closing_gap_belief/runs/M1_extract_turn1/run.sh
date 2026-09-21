#!/bin/bash
# M1 extract turn1 hidden states (10k)
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${GPU_ID:-1}
cd /data/zhenqian/Reproduction1/mechanica/belief/closing_gap_belief
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate belief
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
python code/step2_extract_hidden.py --mode turn1 --batch-size 8
