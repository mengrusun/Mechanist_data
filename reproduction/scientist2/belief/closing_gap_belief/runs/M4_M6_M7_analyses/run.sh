#!/bin/bash
# M4 (dissociation) + M6 (paraphrase re-fits) + M7 (single-pass re-fits) analyses
set -euo pipefail
cd /data/zhenqian/Reproduction1/mechanica/belief/closing_gap_belief
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate belief
python code/step6_analyses.py
