#!/bin/bash
# M1 extract single-pass hidden states (10k)
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${GPU_ID:-6}
cd /data/zhenqian/Reproduction1/mechanica/belief/closing_gap_belief
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate belief
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
python code/step2_extract_hidden.py --mode single --batch-size 8
