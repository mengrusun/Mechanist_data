#!/bin/bash
# CPU-only variant (no GPU needed). Evo2-7B NOT loaded (project HARD-pin honored trivially).
cd /data/wanghaoxiong/Mechanist-DNA-experiment/simple_20260819_v4
source /data/wanghaoxiong/miniconda3/etc/profile.d/conda.sh && conda activate scientist
python verify/C2_eval_harness_fidelity/variants/method-swap-gor-windowed/run_variant.py
