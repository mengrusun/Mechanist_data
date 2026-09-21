#!/bin/bash
# Wave 2: M2 + M3 in parallel after M1 completes.
set -e

REPO=/data/zhenqian/Reproduction1/mechanica/emotion/emotion_circuit
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate belief

mkdir -p $REPO/logs

# M2 on GPU 1 (M1 finished there — memory freed after script exit)
CUDA_VISIBLE_DEVICES=1 nohup python $REPO/experiments/m2_causal.py --out_dir $REPO/runs/A2_causal --m1_dir $REPO/runs/A1_location > $REPO/logs/M2.log 2>&1 &
PID_M2=$!
echo "M2 PID=$PID_M2 (GPU 1)"

# M3 on GPU 6 (in parallel)
export NO_PROXY=dmxapi.cn,www.dmxapi.cn,localhost,127.0.0.1
unset HTTP_PROXY HTTPS_PROXY all_proxy
CUDA_VISIBLE_DEVICES=6 nohup python $REPO/experiments/m3_applied.py --out_dir $REPO/runs/A3_applied --m1_dir $REPO/runs/A1_location > $REPO/logs/M3.log 2>&1 &
PID_M3=$!
echo "M3 PID=$PID_M3 (GPU 6)"

echo "Waiting for wave 2 completion..."
wait $PID_M2 $PID_M3
echo "Wave 2 done."
