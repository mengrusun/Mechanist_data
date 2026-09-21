#!/usr/bin/env bash
# Iteration 1, action ② (main-experiment-script fix for C3).
# Fixes:
#   1. Reduced α range: {0.5, 1.0, 2.0} -> {0.05, 0.1, 0.3}
#      (verify report: cumulative additive injection over ~24 heads + ~2000 neurons pushes OOD;
#      Arm A macro accuracy was 0.053 - below 1/6 chance floor)
#   2. Reduced component count: neighborhood shifted downward from {12,24,48}x{1000,2000,4000}
#      to {5,12,24}x{200,500,2000} to test smaller effective doses
#
# Runs at full plan eval scale (120 val + 120 eval x 6 emo x 3 arms x 9 val configs).
# Outputs to runs/iteration_round_1/A3_applied_fixed/.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
cd /data/zhenqian/Reproduction1/mechanica/emotion/emotion_circuit
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate belief
export HF_TOKEN=<Your_token>
export MODELSCOPE_API_TOKEN=<Your_token>

# NO_PROXY for judge
export NO_PROXY=dmxapi.cn,www.dmxapi.cn,localhost,127.0.0.1
unset HTTP_PROXY HTTPS_PROXY all_proxy http_proxy https_proxy

# Env-driven overrides (parsed inside m3_applied.py)
export ALPHAS_ARM_A=0.05,0.1,0.3
export K_H_ARM_A=5,12,24
export K_N_ARM_A=200,500,2000

mkdir -p runs/iteration_round_1/A3_applied_fixed
python experiments/m3_applied.py \
  --m1_dir runs/A1_location \
  --out_dir runs/iteration_round_1/A3_applied_fixed \
  --judge_workers 6 \
  2>&1 | tee runs/iteration_round_1/A3_applied_fixed/run.log
