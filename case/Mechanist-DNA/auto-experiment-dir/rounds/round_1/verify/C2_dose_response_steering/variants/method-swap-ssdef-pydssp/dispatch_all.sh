#!/bin/bash
# Dispatch all (alpha, seed) cells for the method-swap-ssdef-pydssp variant across free GPUs.
# Designed to be launched fully detached (setsid + nohup, own process group) by the caller so it
# survives the launching shell/turn ending -- see launch_detached.sh.
set -e
VDIR=/data/wanghaoxiong/intergene_mechanist_v6/verify/C2_dose_response_steering/variants/method-swap-ssdef-pydssp
GPUS="${1:-3,4,5}"
N="${2:-150}"
PER_RUN_TIMEOUT="${3:-2400}"   # seconds, per (alpha,seed) cell
IFS=',' read -ra GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

ALPHAS=(0 4 8 16 32)
SEEDS=(42 200)

mkdir -p "$VDIR/data" "$VDIR/logs"
i=0
pids=()
for a in "${ALPHAS[@]}"; do
  for s in "${SEEDS[@]}"; do
    gpu="${GPU_ARR[$((i % NGPU))]}"
    OUT="$VDIR/data/results_a${a}_s${s}.json"
    SEQOUT="$VDIR/data/seqs_a${a}_s${s}.jsonl"
    LOG="$VDIR/logs/a${a}_s${s}.log"
    echo "[dispatch] alpha=$a seed=$s -> GPU $gpu (timeout ${PER_RUN_TIMEOUT}s)"
    ( CUDA_VISIBLE_DEVICES=$gpu timeout "$PER_RUN_TIMEOUT" \
        conda run -n scientist python "$VDIR/run_variant.py" \
        --alpha "$a" --seed "$s" --n "$N" --out "$OUT" --seq_out "$SEQOUT" \
    ) > "$LOG" 2>&1 &
    pids+=($!)
    i=$((i + 1))
    if (( i % NGPU == 0 )); then
      wait "${pids[@]}"
      pids=()
    fi
  done
done
wait "${pids[@]}" 2>/dev/null || true
echo "[dispatch] all cells complete: $(date)"
