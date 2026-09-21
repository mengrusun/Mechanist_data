#!/bin/bash
# M5 hidden state extraction for turn2_p1 (P1 paraphrase, 500 dev samples)
set -euo pipefail
export CUDA_VISIBLE_DEVICES=${GPU_ID:-1}
cd /data/zhenqian/Reproduction1/mechanica/belief/closing_gap_belief
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate belief
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
python code/step2_extract_hidden.py --mode turn2_p1 --batch-size 8
