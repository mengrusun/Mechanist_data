#!/bin/bash
# M1_probes + M2 + parts of M5 nulls — full step4_probes.py at 1000 bootstrap resamples
set -euo pipefail
cd /data/zhenqian/Reproduction1/mechanica/belief/closing_gap_belief
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate belief
python code/step4_probes.py --n-bootstrap 200 --layers all --n-random-null 100
