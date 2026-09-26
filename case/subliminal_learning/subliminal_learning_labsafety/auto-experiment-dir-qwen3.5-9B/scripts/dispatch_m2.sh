#!/usr/bin/env bash
# M2 — Causal Intervention (Steering Vectors / CAA family).
#
# Grid: α ∈ {-2, -1, 0, +1, +2} × run_kind ∈ {steering m1_top_k, steering random_matched, ablation, patching}
# Layer: 4 (top divergent layer from M1)
# Applied to: treated_seed100 (primary treated).
#
# Also cross-seed: same grid at α=-1, α=-2 on seed200 and seed300.
#
# Distributes work across GPUs 3,4,5,6 in waves.
set -euo pipefail
cd "$(dirname "$0")/.."

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm

BASE=/mnt/quarkfs/share_model/Qwen3.5-9B
QA_I=/data/zhenqian/exp/subliminal/multi_modal/data/QA_I-00000-of-00001.parquet
JUDGE_CACHE=caches/eval_cache.jsonl
LAYER=4
mkdir -p mechanism/M2_causal logs

# Wave 1: primary seed100 — 5 alphas × 4 run_kinds; do all steering m1_top_k first (5 runs)
# Distribute 5 α over 4 GPUs (3,4,5,6) — 4 in wave A, 1 in wave B (or overlap via 5 GPUs 3,4,5,6,7 if available)

launch_run () {
  local GPU=$1
  local ADAPTER=$2
  local DIRECTIONS=$3
  local ALPHA=$4
  local SHAPE=$5
  local SOURCE=$6
  local TREATED_SEED=$7
  local OUT=$8
  local LOG=$9
  echo "[m2] gpu=$GPU adapter=$ADAPTER shape=$SHAPE source=$SOURCE α=$ALPHA seed=$TREATED_SEED"
  nohup env CUDA_VISIBLE_DEVICES=$GPU python scripts/mechanism_m2_intervene.py \
      --base_model $BASE \
      --adapter $ADAPTER \
      --benchmark $QA_I \
      --directions_pt $DIRECTIONS \
      --ctrl_activations_held_pt mechanism/M1_location/ctrl/activations_held_out.pt \
      --ctrl_ids_held_json mechanism/M1_location/ctrl/row_ids_held_out.json \
      --layer $LAYER \
      --intervention_shape $SHAPE \
      --alpha $ALPHA \
      --component_set_source $SOURCE \
      --treated_seed $TREATED_SEED \
      --seed 42 \
      --judge_cache $JUDGE_CACHE \
      --out $OUT \
      --resume_from_output \
      > $LOG 2>&1 &
}

DIRS_S100=mechanism/M1_location/treated_seed100/directions.pt
DIRS_S200=mechanism/M1_location/treated_seed200/directions.pt
DIRS_S300=mechanism/M1_location/treated_seed300/directions.pt

# ============================================================
# WAVE 1: seed100 real steering α ∈ {-2, -1, 0, +1, +2} — 5 runs on GPUs 3,4,5,6 + one more
# 5 runs / 4 GPUs → 4 in first sub-wave, 1 in second sub-wave. Or use GPU 7 too if free.
# We'll go: -2, -1 on GPU 3; 0, +1 on GPU 4; +2 on GPU 5 (sequential per GPU allowed since one run << 30 min).
# But for parallelism, run 4 at once across GPUs 3-6, then last one.
# ============================================================

echo "=== WAVE 1: seed100 real steering (5 runs) ==="
mkdir -p mechanism/M2_causal/steering_m1_seed100

# Sub-wave A: 4 runs in parallel
launch_run 3 ckpts/student_seed100 $DIRS_S100 -2 steering m1_top_k 100 mechanism/M2_causal/steering_m1_seed100/alpha-2.jsonl logs/m2_s100_steering_m1_a-2.log
PID_A1=$!
launch_run 4 ckpts/student_seed100 $DIRS_S100 -1 steering m1_top_k 100 mechanism/M2_causal/steering_m1_seed100/alpha-1.jsonl logs/m2_s100_steering_m1_a-1.log
PID_A2=$!
launch_run 5 ckpts/student_seed100 $DIRS_S100 0 steering m1_top_k 100 mechanism/M2_causal/steering_m1_seed100/alpha0.jsonl logs/m2_s100_steering_m1_a0.log
PID_A3=$!
launch_run 6 ckpts/student_seed100 $DIRS_S100 1 steering m1_top_k 100 mechanism/M2_causal/steering_m1_seed100/alpha1.jsonl logs/m2_s100_steering_m1_a1.log
PID_A4=$!
wait $PID_A1 $PID_A2 $PID_A3 $PID_A4

# Sub-wave B: last 1 run
launch_run 3 ckpts/student_seed100 $DIRS_S100 2 steering m1_top_k 100 mechanism/M2_causal/steering_m1_seed100/alpha2.jsonl logs/m2_s100_steering_m1_a2.log
PID_B1=$!
wait $PID_B1

echo "=== WAVE 2: seed100 random_matched control steering (5 runs) ==="
mkdir -p mechanism/M2_causal/steering_random_seed100
launch_run 3 ckpts/student_seed100 $DIRS_S100 -2 steering random_matched 100 mechanism/M2_causal/steering_random_seed100/alpha-2.jsonl logs/m2_s100_steering_random_a-2.log
PID_1=$!
launch_run 4 ckpts/student_seed100 $DIRS_S100 -1 steering random_matched 100 mechanism/M2_causal/steering_random_seed100/alpha-1.jsonl logs/m2_s100_steering_random_a-1.log
PID_2=$!
launch_run 5 ckpts/student_seed100 $DIRS_S100 1 steering random_matched 100 mechanism/M2_causal/steering_random_seed100/alpha1.jsonl logs/m2_s100_steering_random_a1.log
PID_3=$!
launch_run 6 ckpts/student_seed100 $DIRS_S100 2 steering random_matched 100 mechanism/M2_causal/steering_random_seed100/alpha2.jsonl logs/m2_s100_steering_random_a2.log
PID_4=$!
wait $PID_1 $PID_2 $PID_3 $PID_4
# α=0 for random is identical to α=0 for real — skip

echo "=== WAVE 3: seed100 ablation + patching (2 runs) ==="
mkdir -p mechanism/M2_causal/ablation_seed100 mechanism/M2_causal/patching_seed100
launch_run 3 ckpts/student_seed100 $DIRS_S100 0 ablation m1_top_k 100 mechanism/M2_causal/ablation_seed100/results.jsonl logs/m2_s100_ablation.log
PID_1=$!
launch_run 4 ckpts/student_seed100 $DIRS_S100 0 patching m1_top_k 100 mechanism/M2_causal/patching_seed100/results.jsonl logs/m2_s100_patching.log
PID_2=$!
wait $PID_1 $PID_2

echo "=== WAVE 4: cross-seed replication (4 runs — α=-1,-2 on seed200 and seed300) ==="
mkdir -p mechanism/M2_causal/steering_m1_seed200 mechanism/M2_causal/steering_m1_seed300
launch_run 3 ckpts/student_seed200 $DIRS_S200 -1 steering m1_top_k 200 mechanism/M2_causal/steering_m1_seed200/alpha-1.jsonl logs/m2_s200_steering_m1_a-1.log
PID_1=$!
launch_run 4 ckpts/student_seed200 $DIRS_S200 -2 steering m1_top_k 200 mechanism/M2_causal/steering_m1_seed200/alpha-2.jsonl logs/m2_s200_steering_m1_a-2.log
PID_2=$!
launch_run 5 ckpts/student_seed300 $DIRS_S300 -1 steering m1_top_k 300 mechanism/M2_causal/steering_m1_seed300/alpha-1.jsonl logs/m2_s300_steering_m1_a-1.log
PID_3=$!
launch_run 6 ckpts/student_seed300 $DIRS_S300 -2 steering m1_top_k 300 mechanism/M2_causal/steering_m1_seed300/alpha-2.jsonl logs/m2_s300_steering_m1_a-2.log
PID_4=$!
wait $PID_1 $PID_2 $PID_3 $PID_4

echo "=== M2 all done ==="
