#!/bin/bash
# Wave 1: M0.5 (API-bound) + M1 (GPU 1) in parallel.
set -e

REPO=/data/zhenqian/Reproduction1/mechanica/emotion/emotion_circuit
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate belief

mkdir -p $REPO/logs

# M0.5 (no GPU, judge API)
export NO_PROXY=dmxapi.cn,www.dmxapi.cn,localhost,127.0.0.1
unset HTTP_PROXY HTTPS_PROXY all_proxy
nohup python $REPO/experiments/m0_5_dataprep_judge.py --out_dir $REPO/runs/A0.5_dataprep_judge > $REPO/logs/M0.5.log 2>&1 &
PID_M05=$!
echo "M0.5 PID=$PID_M05"

# M1 (GPU 1)
CUDA_VISIBLE_DEVICES=1 nohup python $REPO/experiments/m1_location.py --out_dir $REPO/runs/A1_location > $REPO/logs/M1.log 2>&1 &
PID_M1=$!
echo "M1 PID=$PID_M1 (GPU 1)"

echo "Waiting for wave 1 completion..."
wait $PID_M05 $PID_M1
echo "Wave 1 done."
