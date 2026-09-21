#!/bin/bash
# Dispatch all (alpha, seed) cells for the method-swap-omegafold variant across free GPUs.
# HC2/verify GPU constraint: CUDA_VISIBLE_DEVICES pinned to subset of {2,3,4,5}; GPU 2 held by another
# user (weiyunx) for the whole session per team-lead's note -- use free GPUs among {3,4,5} (add 2 back
# if it frees up). max_parallel matches the number of free GPUs passed in.
set -e
VDIR=/data/wanghaoxiong/intergene_mechanist_v6/verify/C2_dose_response_steering/variants/method-swap-omegafold
CONF_MIN="${1:?usage: dispatch_all.sh <calibrated_conf_min> [gpu_list=3,4,5] [n_per_dose=150]}"
GPUS="${2:-3,4,5}"
N="${3:-150}"
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

ALPHAS=(0 4 8 16 32)
SEEDS=(42 200)

mkdir -p "$VDIR/logs"
i=0
pids=()
for a in "${ALPHAS[@]}"; do
  for s in "${SEEDS[@]}"; do
    gpu="${GPU_ARR[$((i % NGPU))]}"
    echo "[dispatch] alpha=$a seed=$s -> GPU $gpu"
    "$VDIR/run.sh" "$gpu" "$a" "$s" "$N" "$CONF_MIN" > "$VDIR/logs/a${a}_s${s}.log" 2>&1 &
    pids+=($!)
    i=$((i + 1))
    if (( i % NGPU == 0 )); then
      wait "${pids[@]}"
      pids=()
    fi
  done
done
wait "${pids[@]}" 2>/dev/null || true
echo "[dispatch] all cells complete"
