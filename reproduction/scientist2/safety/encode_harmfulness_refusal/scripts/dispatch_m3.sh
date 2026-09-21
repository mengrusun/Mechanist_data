#!/bin/bash
# Dispatch M3's 28-config grid across 4 GPUs (0,1,2,3) with max 4 parallel runs.
# Runs are written to results/m3/${direction}_a${alpha}/
# Total: 4 directions * 7 alphas = 28 configs

set -euo pipefail
cd /data/zhenqian/Reproduction1/mechanica/safety/encode_harmfulness_refusal

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh && conda activate lsa_safety

MODEL=/data/zhenqian/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct
DIRECTIONS="h r random swap"
ALPHAS="-2 -1 -0.5 0 0.5 1 2"
N_HARM=${N_HARM:-100}
N_BEN=${N_BEN:-100}
GEN_BS=${GEN_BS:-8}
GEN_MAX=${GEN_MAX:-96}

mkdir -p logs/m3 results/m3

# Build the config list
CONFIGS=()
for d in $DIRECTIONS; do
  for a in $ALPHAS; do
    CONFIGS+=("$d,$a")
  done
done

N_CONFIGS=${#CONFIGS[@]}
echo "Total configs: $N_CONFIGS"

# Round-robin over 4 GPUs with max 4 parallel
GPU_LIST=(0 1 2 3)
MAX_PARALLEL=4

running=()
for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  direction="${cfg%%,*}"
  alpha="${cfg##*,}"
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  log="logs/m3/${direction}_a${alpha}.log"
  outdir="results/m3/${direction}_a${alpha}"

  # Wait if we already have MAX_PARALLEL jobs running
  while [ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]; do
    sleep 2
  done

  echo "[dispatch] $i/$N_CONFIGS  gpu=$gpu  dir=$direction  alpha=$alpha  -> $log"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/m3_claim3_steering.py \
    --model $MODEL \
    --prep results/m_prep/ \
    --advbench /data/zhenqian/data/AdvBench \
    --alpaca /data/zhenqian/data/Alpaca \
    --direction "$direction" --alpha "$alpha" \
    --n_harm $N_HARM --n_benign $N_BEN \
    --gen_batch_size $GEN_BS --gen_max_new_tokens $GEN_MAX \
    --out "$outdir" --seed 0 > "$log" 2>&1 &
done

wait
echo "-------- all M3 configs completed --------"
