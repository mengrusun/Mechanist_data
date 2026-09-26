#!/usr/bin/env bash
# M2 iteration-3 fix — Cross-seed widened steering on seed200 and seed300.
#
# Reviewer iter-2 memory called out: "cross-seed steering non-replication remains unaddressed
# — only seed100 was widened". Iter-3 addresses that by running the same α ∈ {-3, -0.5, +0.5, +3}
# widened-sweep points on seed200 and seed300 (using each seed's own M1 direction).
#
# The original sweep already ran α ∈ {-1, -2} on both seed200 and seed300 — this adds 4 new α
# per seed × 2 seeds = 8 new runs at n=27 held-out each.

set -euo pipefail
cd "$(dirname "$0")/.."

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm

BASE=/mnt/quarkfs/share_model/Qwen3.5-9B
QA_I=/data/zhenqian/exp/subliminal/multi_modal/data/QA_I-00000-of-00001.parquet
JUDGE_CACHE=caches/eval_cache.jsonl
LAYER=4
mkdir -p mechanism/M2_causal/steering_m1_seed200 \
         mechanism/M2_causal/steering_m1_seed300 \
         logs

launch_qa_run () {
  local GPU=$1
  local ADAPTER=$2
  local DIRS=$3
  local ALPHA=$4
  local SEED=$5
  local OUTDIR=$6
  local LOG=$7
  echo "[m2-iter3] gpu=$GPU α=$ALPHA seed=$SEED"
  nohup env CUDA_VISIBLE_DEVICES=$GPU python scripts/mechanism_m2_intervene.py \
      --base_model $BASE \
      --adapter $ADAPTER \
      --benchmark $QA_I \
      --directions_pt $DIRS \
      --ctrl_activations_held_pt mechanism/M1_location/ctrl/activations_held_out.pt \
      --ctrl_ids_held_json mechanism/M1_location/ctrl/row_ids_held_out.json \
      --layer $LAYER \
      --intervention_shape steering \
      --alpha $ALPHA \
      --component_set_source m1_top_k \
      --treated_seed $SEED \
      --seed 42 \
      --judge_cache $JUDGE_CACHE \
      --out $OUTDIR/alpha${ALPHA}.jsonl \
      --resume_from_output \
      > $LOG 2>&1 &
}

DIRS_S200=mechanism/M1_location/treated_seed200/directions.pt
DIRS_S300=mechanism/M1_location/treated_seed300/directions.pt

# WAVE 1: seed200 × {-3, -0.5, +0.5, +3} (4 runs GPUs 3,4,5,6)
echo "=== WAVE 1: seed200 widened (4 runs) ==="
launch_qa_run 3 ckpts/student_seed200 $DIRS_S200 -3   200 mechanism/M2_causal/steering_m1_seed200 logs/m2i3_s200_a-3.log ; P1=$!
launch_qa_run 4 ckpts/student_seed200 $DIRS_S200 -0.5 200 mechanism/M2_causal/steering_m1_seed200 logs/m2i3_s200_a-0.5.log ; P2=$!
launch_qa_run 5 ckpts/student_seed200 $DIRS_S200 0.5  200 mechanism/M2_causal/steering_m1_seed200 logs/m2i3_s200_a0.5.log ; P3=$!
launch_qa_run 6 ckpts/student_seed200 $DIRS_S200 3    200 mechanism/M2_causal/steering_m1_seed200 logs/m2i3_s200_a3.log ; P4=$!
wait $P1 $P2 $P3 $P4

# WAVE 2: seed300 × {-3, -0.5, +0.5, +3} (4 runs GPUs 3,4,5,6)
echo "=== WAVE 2: seed300 widened (4 runs) ==="
launch_qa_run 3 ckpts/student_seed300 $DIRS_S300 -3   300 mechanism/M2_causal/steering_m1_seed300 logs/m2i3_s300_a-3.log ; P1=$!
launch_qa_run 4 ckpts/student_seed300 $DIRS_S300 -0.5 300 mechanism/M2_causal/steering_m1_seed300 logs/m2i3_s300_a-0.5.log ; P2=$!
launch_qa_run 5 ckpts/student_seed300 $DIRS_S300 0.5  300 mechanism/M2_causal/steering_m1_seed300 logs/m2i3_s300_a0.5.log ; P3=$!
launch_qa_run 6 ckpts/student_seed300 $DIRS_S300 3    300 mechanism/M2_causal/steering_m1_seed300 logs/m2i3_s300_a3.log ; P4=$!
wait $P1 $P2 $P3 $P4

echo "=== M2 iteration-3 dispatch done ==="
