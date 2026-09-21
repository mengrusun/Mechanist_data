#!/bin/bash
# C3 mechanism-audit fix (iteration 1). Two sub-batches:
#   Fine sub-sweep: adds alphas at 0.03/0.1/0.3 sigma_proj (both signs) for h, r, swap
#                    → 6 alphas x 3 dirs = 18 new configs
#   n_random matched-norm controls: 30 seeds at 2 chosen alphas (alpha_sigma_h = 1.5 and 3.0)
#                    → 60 new configs
# Total: 78 configs, ~1.3 GPU-h, ~20 min wall on 4 GPUs.
# Outputs: runs/iteration_round_1/m3_extended/<direction>_a<alpha>[_s<seed>]/

set -euo pipefail
cd /data/zhenqian/Reproduction1/mechanica/safety/encode_harmfulness_refusal

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh && conda activate lsa_safety

MODEL=/data/zhenqian/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct
N_HARM=${N_HARM:-100}
N_BEN=${N_BEN:-100}
GEN_BS=${GEN_BS:-8}
GEN_MAX=${GEN_MAX:-96}

OUT_ROOT=runs/iteration_round_1/m3_extended
mkdir -p logs/m3_fix "$OUT_ROOT"

# Fine sub-sweep alphas in raw ||d|| units (derived from sigma_proj target values).
# For h at sigma_proj_h = 1.579, direction_norm_h = 2.949 → raw = sigma * norm/sigma_proj = sigma * 1.867
#   alpha_sigma = ±{0.03, 0.1, 0.3} → raw = ±{0.056, 0.187, 0.560}
# For r at sigma_proj_r = 1.964, direction_norm_r = 3.708 → raw = sigma * 1.888
#   alpha_sigma = ±{0.03, 0.1, 0.3} → raw = ±{0.056, 0.189, 0.566}
# swap uses r-direction at h's site; its raw amplitude is in ||r|| units, so use r's mapping.
H_FINE_ALPHAS="-0.56 -0.187 -0.056 0.056 0.187 0.56"
R_FINE_ALPHAS="-0.566 -0.189 -0.056 0.056 0.189 0.566"
SWAP_FINE_ALPHAS="-0.566 -0.189 -0.056 0.056 0.189 0.566"

# n_random ≥ 30 alphas: pick two operating points in raw ||d|| units.
#   alpha_sigma_h = +1.5 σ_h → raw = 1.5 * 1.867 = 0.803
#   alpha_sigma_h = +3.0 σ_h → raw = 3.0 * 1.867 = 1.607 (near the existing +2 raw = +3.73 σ boundary)
RANDOM_ALPHAS="0.803 1.607"
RANDOM_SEEDS=$(seq 0 29)  # 30 distinct random-direction seeds; prompt --seed pinned at 0

# Build config list: (direction, alpha, extra_flags, outdir_tag)
CONFIGS=()

for a in $H_FINE_ALPHAS; do
  CONFIGS+=("h|$a|--seed 0|h_a${a}")
done
for a in $R_FINE_ALPHAS; do
  CONFIGS+=("r|$a|--seed 0|r_a${a}")
done
for a in $SWAP_FINE_ALPHAS; do
  CONFIGS+=("swap|$a|--seed 0|swap_a${a}")
done
for a in $RANDOM_ALPHAS; do
  for s in $RANDOM_SEEDS; do
    CONFIGS+=("random|$a|--seed 0 --random_dir_seed $s|random_a${a}_s${s}")
  done
done

N_CONFIGS=${#CONFIGS[@]}
echo "Total configs: $N_CONFIGS"

# Round-robin over 4 GPUs (0,1,2,3) with max 4 parallel.
GPU_LIST=(0 1 2 3)
MAX_PARALLEL=4

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  IFS='|' read -r direction alpha extra_flags outdir_tag <<< "$cfg"
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  log="logs/m3_fix/${outdir_tag}.log"
  outdir="$OUT_ROOT/${outdir_tag}"

  # Wait if we already have MAX_PARALLEL jobs running.
  while [ $(jobs -rp | wc -l) -ge $MAX_PARALLEL ]; do
    sleep 2
  done

  echo "[dispatch] $((i+1))/$N_CONFIGS  gpu=$gpu  dir=$direction  alpha=$alpha  ${extra_flags}  -> $outdir  (log=$log)"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/m3_claim3_steering.py \
    --model $MODEL \
    --prep results/m_prep/ \
    --advbench /data/zhenqian/data/AdvBench \
    --alpaca /data/zhenqian/data/Alpaca \
    --direction "$direction" --alpha "$alpha" \
    --n_harm $N_HARM --n_benign $N_BEN \
    --gen_batch_size $GEN_BS --gen_max_new_tokens $GEN_MAX \
    --out "$outdir" \
    $extra_flags > "$log" 2>&1 &
done

wait
echo "-------- all C3-fix configs completed --------"

# Emit a cost.json for the whole batch.
python3 - << 'PY'
import json, glob, time
from pathlib import Path
files = sorted(glob.glob("runs/iteration_round_1/m3_extended/*/steering_metrics.json"))
total_s = 0.0
for f in files:
    d = json.load(open(f))
    total_s += float(d.get("wall_clock_seconds", 0.0))
# Effective single-GPU-hours (sum of wall_clock per process); this is what the budget tracks.
gpu_hours = total_s / 3600.0
cost = {
    "runs_total": len(files),
    "gpu_hours": round(gpu_hours, 4),
    "gpu_ids": "0,1,2,3",
    "start_ts": None,
    "end_ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "status": "completed",
    "purpose": "C3 mechanism-audit fix: fine alpha sub-sweep in sigma_proj units + n_random >= 30 controls",
}
Path("runs/iteration_round_1/cost.json").write_text(json.dumps(cost, indent=2))
print(f"[cost] runs={len(files)}  gpu_hours={gpu_hours:.4f}")
PY
