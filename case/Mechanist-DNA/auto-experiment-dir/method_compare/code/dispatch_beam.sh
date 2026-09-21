#!/bin/bash
# Launch the beam-search grid: 2 arms x 7 search widths, one queue per GPU.
# Load is balanced by W (cost is ~linear in W); the two W=32 jobs get a dedicated GPU each.
cd /data/wanghaoxiong/Mechanist-DNA-experiment/early_try_v6_refine
source /data/wanghaoxiong/miniconda3/etc/profile.d/conda.sh
conda activate scientist
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MC_GEN_BATCH=${MC_GEN_BATCH:-48}
P=${P:-100}
R=method_compare/results
L=method_compare/logs

run_queue () {            # $1 = gpu, rest = "arm:width" pairs
  local gpu=$1; shift
  for job in "$@"; do
    arm=${job%%:*}; w=${job##*:}
    CUDA_VISIBLE_DEVICES=$gpu python method_compare/code/run_beam.py \
      --arm $arm --width $w --n_prompts $P \
      --out $R/beam_${arm}_W${w}.json > $L/beam_${arm}_W${w}.log 2>&1
  done
}

run_queue 0 base:32          &
run_queue 1 steer:32         &
run_queue 2 base:16          &
run_queue 3 steer:16         &
run_queue 4 base:8 base:4    &
run_queue 5 steer:8 steer:4  &
run_queue 6 base:2 base:1 base:0    &
run_queue 7 steer:2 steer:1 steer:0 &
wait
echo "ALL_BEAM_JOBS_DONE"
