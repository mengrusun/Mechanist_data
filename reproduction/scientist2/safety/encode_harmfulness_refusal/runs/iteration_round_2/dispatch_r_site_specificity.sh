#!/bin/bash
# Iteration 2 fix: r-site random-direction specificity control (n=30 at 2 alphas).
# Addresses Iteration-2 reviewer's flagged residual loophole: iter-1's random controls
# were run at h's site, not r's site. This closes that loophole by running 30
# matched-norm random directions at r's actual site (best_r_layer=13, t_post_instr).
#
# Two operating points: alpha_raw = 1.0 (~1.87 sigma_r) and 2.0 (~3.78 sigma_r) —
# the same alphas at which true r was measured in the original 28-cell grid, so
# a direct like-for-like comparison is possible.
#
# Cost: 60 configs × ~60s / 4 GPUs = ~900s wall = ~1.0 GPU-h.

set -euo pipefail
cd /data/zhenqian/Reproduction1/mechanica/safety/encode_harmfulness_refusal

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh && conda activate lsa_safety

MODEL=/data/zhenqian/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct
N_HARM=${N_HARM:-100}
N_BEN=${N_BEN:-100}
GEN_BS=${GEN_BS:-8}
GEN_MAX=${GEN_MAX:-96}

OUT_ROOT=runs/iteration_round_2/random_r_site
mkdir -p logs/m3_iter2 "$OUT_ROOT"

ALPHAS="1.0 2.0"
SEEDS=$(seq 0 29)

CONFIGS=()
for a in $ALPHAS; do
  for s in $SEEDS; do
    CONFIGS+=("$a|$s")
  done
done

N_CONFIGS=${#CONFIGS[@]}
echo "Total configs: $N_CONFIGS"

GPU_LIST=(0 1 2 3)
MAX_PARALLEL=4

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  IFS='|' read -r alpha seed <<< "$cfg"
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  log="logs/m3_iter2/random_r_site_a${alpha}_s${seed}.log"
  outdir="$OUT_ROOT/random_r_site_a${alpha}_s${seed}"

  while [ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]; do
    sleep 2
  done

  echo "[dispatch] $((i+1))/$N_CONFIGS  gpu=$gpu  alpha=$alpha  seed=$seed  -> $outdir"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/m3_claim3_steering.py \
    --model $MODEL \
    --prep results/m_prep/ \
    --advbench /data/zhenqian/data/AdvBench \
    --alpaca /data/zhenqian/data/Alpaca \
    --direction random_r_site --alpha "$alpha" \
    --n_harm $N_HARM --n_benign $N_BEN \
    --gen_batch_size $GEN_BS --gen_max_new_tokens $GEN_MAX \
    --out "$outdir" --seed 0 --random_dir_seed "$seed" > "$log" 2>&1 &
done

wait
echo "-------- all r-site random configs completed --------"

# Emit cost.json for iteration_round_2.
python3 - << 'PY'
import json, glob, time
from pathlib import Path
files = sorted(glob.glob("runs/iteration_round_2/random_r_site/*/steering_metrics.json"))
total_s = sum(float(json.load(open(f)).get("wall_clock_seconds", 0.0)) for f in files)
cost = {
    "runs_total": len(files),
    "gpu_hours": round(total_s / 3600, 4),
    "gpu_ids": "0,1,2,3",
    "start_ts": None,
    "end_ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "status": "completed",
    "purpose": "Iteration-2 fix: r-site random-direction specificity control (n=30 at 2 alphas)",
}
Path("runs/iteration_round_2/cost.json").write_text(json.dumps(cost, indent=2))
print(f"[cost] runs={len(files)}  gpu_hours={cost['gpu_hours']}")
PY
