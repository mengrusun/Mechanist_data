#!/bin/bash
# Wave 3: M4 (Qwen verify swap) after M3 completes.
set -e

REPO=/data/zhenqian/Reproduction1/mechanica/emotion/emotion_circuit
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate belief

mkdir -p $REPO/logs
export NO_PROXY=dmxapi.cn,www.dmxapi.cn,localhost,127.0.0.1
unset HTTP_PROXY HTTPS_PROXY all_proxy

# M4 on GPU 1 (M2 finished; Qwen-7B fits comfortably in bf16 ~15GB)
CUDA_VISIBLE_DEVICES=1 nohup python $REPO/experiments/m4_verify_qwen.py --out_dir $REPO/runs/A4_verify_qwen > $REPO/logs/M4.log 2>&1 &
PID_M4=$!
echo "M4 PID=$PID_M4 (GPU 1)"

wait $PID_M4
echo "Wave 3 done."
