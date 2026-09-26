#!/usr/bin/env bash
# M2 iteration-1 fix — Widen α sweep + add MMLU capability control at every α.
#
# Reviewer diagnosis: C3 INCONCLUSIVE because
#   A.3 FAIL — MMLU never logged at any α (cannot rule out capability collapse)
#   A.4 FAIL — no plateau in original {-2,-1,0,+1,+2} sweep (best at boundary α=+2)
#
# Fix:
#   1. Widen α on QA_I real m1_top_k + random_matched by adding
#      {-3, -0.5, +0.5, +3} (existing {-2,-1,0,+1,+2} sweep is reused as-is).
#      This lets us see whether the effect plateaus or grows further outside the
#      original range.
#   2. Add MMLU-slice capability evaluation at ALL 9 α ∈ {-3,-2,-1,-0.5,0,+0.5,+1,+2,+3}
#      on both real m1_top_k and random_matched sources (skip α=0 for random —
#      identical to α=0 for real).
#
# All runs are on seed100 primary treated (α × source × metric grid).
# GPU pin: gpu_ids ∈ {3,4,5,6,7} (task.md hard constraint).
# Output tree:
#   mechanism/M2_causal/steering_m1_seed100/alpha<A>.jsonl        (QA_I, real)
#   mechanism/M2_causal/steering_random_seed100/alpha<A>.jsonl    (QA_I, random)
#   mechanism/M2_causal_widened/mmlu/steering_m1_seed100/alpha<A>.jsonl  (MMLU, real)
#   mechanism/M2_causal_widened/mmlu/steering_random_seed100/alpha<A>.jsonl  (MMLU, random)
#
# Wave-parallel across GPUs 3,4,5,6,7 (5 concurrent jobs). Resume-from-output honored.

set -euo pipefail
cd "$(dirname "$0")/.."

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm

BASE=/mnt/quarkfs/share_model/Qwen3.5-9B
QA_I=/data/zhenqian/exp/subliminal/multi_modal/data/QA_I-00000-of-00001.parquet
MMLU_SLICE=mechanism/mmlu_slice.jsonl
JUDGE_CACHE_QA=caches/eval_cache.jsonl
LAYER=4
DIRS_S100=mechanism/M1_location/treated_seed100/directions.pt
ADAPTER=ckpts/student_seed100
TREATED_SEED=100

mkdir -p mechanism/M2_causal/steering_m1_seed100 \
         mechanism/M2_causal/steering_random_seed100 \
         mechanism/M2_causal_widened/mmlu/steering_m1_seed100 \
         mechanism/M2_causal_widened/mmlu/steering_random_seed100 \
         logs

# --------------------------------------------------------------
# helper: launch a QA_I run in background on a specific GPU
launch_qa_run () {
  local GPU=$1
  local ALPHA=$2
  local SOURCE=$3
  local OUTDIR=$4
  local LOG=$5
  echo "[m2-iter1-qa] gpu=$GPU α=$ALPHA source=$SOURCE"
  nohup env CUDA_VISIBLE_DEVICES=$GPU python scripts/mechanism_m2_intervene.py \
      --base_model $BASE \
      --adapter $ADAPTER \
      --benchmark $QA_I \
      --directions_pt $DIRS_S100 \
      --ctrl_activations_held_pt mechanism/M1_location/ctrl/activations_held_out.pt \
      --ctrl_ids_held_json mechanism/M1_location/ctrl/row_ids_held_out.json \
      --layer $LAYER \
      --intervention_shape steering \
      --alpha $ALPHA \
      --component_set_source $SOURCE \
      --treated_seed $TREATED_SEED \
      --seed 42 \
      --judge_cache $JUDGE_CACHE_QA \
      --out $OUTDIR/alpha${ALPHA}.jsonl \
      --resume_from_output \
      > $LOG 2>&1 &
}

launch_mmlu_run () {
  local GPU=$1
  local ALPHA=$2
  local SOURCE=$3
  local OUTDIR=$4
  local LOG=$5
  echo "[m2-iter1-mmlu] gpu=$GPU α=$ALPHA source=$SOURCE"
  nohup env CUDA_VISIBLE_DEVICES=$GPU python scripts/mechanism_m2_intervene_mmlu.py \
      --base_model $BASE \
      --adapter $ADAPTER \
      --mmlu_slice $MMLU_SLICE \
      --directions_pt $DIRS_S100 \
      --layer $LAYER \
      --intervention_shape steering \
      --alpha $ALPHA \
      --component_set_source $SOURCE \
      --treated_seed $TREATED_SEED \
      --seed 42 \
      --out $OUTDIR/alpha${ALPHA}.jsonl \
      --resume_from_output \
      > $LOG 2>&1 &
}

# ==============================================================
# WAVE 1: QA_I widen — real m1_top_k, NEW α ∈ {-3, -0.5, +0.5, +3}
# 4 runs across GPUs 3,4,5,6 (leave 7 idle for wave 2 warm-up)
# ==============================================================
echo "=== WAVE 1: QA_I real m1_top_k widening (4 runs) ==="
launch_qa_run 3 -3   m1_top_k mechanism/M2_causal/steering_m1_seed100 logs/m2i1_qa_real_a-3.log ; P1=$!
launch_qa_run 4 -0.5 m1_top_k mechanism/M2_causal/steering_m1_seed100 logs/m2i1_qa_real_a-0.5.log ; P2=$!
launch_qa_run 5 0.5  m1_top_k mechanism/M2_causal/steering_m1_seed100 logs/m2i1_qa_real_a0.5.log ; P3=$!
launch_qa_run 6 3    m1_top_k mechanism/M2_causal/steering_m1_seed100 logs/m2i1_qa_real_a3.log ; P4=$!
wait $P1 $P2 $P3 $P4

# ==============================================================
# WAVE 2: QA_I widen — random_matched, NEW α ∈ {-3, -0.5, +0.5, +3}
# ==============================================================
echo "=== WAVE 2: QA_I random_matched widening (4 runs) ==="
launch_qa_run 3 -3   random_matched mechanism/M2_causal/steering_random_seed100 logs/m2i1_qa_rand_a-3.log ; P1=$!
launch_qa_run 4 -0.5 random_matched mechanism/M2_causal/steering_random_seed100 logs/m2i1_qa_rand_a-0.5.log ; P2=$!
launch_qa_run 5 0.5  random_matched mechanism/M2_causal/steering_random_seed100 logs/m2i1_qa_rand_a0.5.log ; P3=$!
launch_qa_run 6 3    random_matched mechanism/M2_causal/steering_random_seed100 logs/m2i1_qa_rand_a3.log ; P4=$!
wait $P1 $P2 $P3 $P4

# ==============================================================
# WAVE 3: MMLU real m1_top_k, α ∈ {-3, -2, -1, -0.5, 0} (5 runs, GPUs 3-7)
# ==============================================================
echo "=== WAVE 3: MMLU real m1_top_k (5 runs) ==="
launch_mmlu_run 3 -3   m1_top_k mechanism/M2_causal_widened/mmlu/steering_m1_seed100 logs/m2i1_mmlu_real_a-3.log ; P1=$!
launch_mmlu_run 4 -2   m1_top_k mechanism/M2_causal_widened/mmlu/steering_m1_seed100 logs/m2i1_mmlu_real_a-2.log ; P2=$!
launch_mmlu_run 5 -1   m1_top_k mechanism/M2_causal_widened/mmlu/steering_m1_seed100 logs/m2i1_mmlu_real_a-1.log ; P3=$!
launch_mmlu_run 6 -0.5 m1_top_k mechanism/M2_causal_widened/mmlu/steering_m1_seed100 logs/m2i1_mmlu_real_a-0.5.log ; P4=$!
launch_mmlu_run 7 0    m1_top_k mechanism/M2_causal_widened/mmlu/steering_m1_seed100 logs/m2i1_mmlu_real_a0.log ; P5=$!
wait $P1 $P2 $P3 $P4 $P5

# ==============================================================
# WAVE 4: MMLU real m1_top_k, α ∈ {+0.5, +1, +2, +3} (4 runs, GPUs 3-6)
# ==============================================================
echo "=== WAVE 4: MMLU real m1_top_k (4 runs) ==="
launch_mmlu_run 3 0.5 m1_top_k mechanism/M2_causal_widened/mmlu/steering_m1_seed100 logs/m2i1_mmlu_real_a0.5.log ; P1=$!
launch_mmlu_run 4 1   m1_top_k mechanism/M2_causal_widened/mmlu/steering_m1_seed100 logs/m2i1_mmlu_real_a1.log ; P2=$!
launch_mmlu_run 5 2   m1_top_k mechanism/M2_causal_widened/mmlu/steering_m1_seed100 logs/m2i1_mmlu_real_a2.log ; P3=$!
launch_mmlu_run 6 3   m1_top_k mechanism/M2_causal_widened/mmlu/steering_m1_seed100 logs/m2i1_mmlu_real_a3.log ; P4=$!
wait $P1 $P2 $P3 $P4

# ==============================================================
# WAVE 5: MMLU random_matched, α ∈ {-3, -2, -1, -0.5, +0.5} (5 runs, GPUs 3-7)
# ==============================================================
echo "=== WAVE 5: MMLU random_matched (5 runs) ==="
launch_mmlu_run 3 -3   random_matched mechanism/M2_causal_widened/mmlu/steering_random_seed100 logs/m2i1_mmlu_rand_a-3.log ; P1=$!
launch_mmlu_run 4 -2   random_matched mechanism/M2_causal_widened/mmlu/steering_random_seed100 logs/m2i1_mmlu_rand_a-2.log ; P2=$!
launch_mmlu_run 5 -1   random_matched mechanism/M2_causal_widened/mmlu/steering_random_seed100 logs/m2i1_mmlu_rand_a-1.log ; P3=$!
launch_mmlu_run 6 -0.5 random_matched mechanism/M2_causal_widened/mmlu/steering_random_seed100 logs/m2i1_mmlu_rand_a-0.5.log ; P4=$!
launch_mmlu_run 7 0.5  random_matched mechanism/M2_causal_widened/mmlu/steering_random_seed100 logs/m2i1_mmlu_rand_a0.5.log ; P5=$!
wait $P1 $P2 $P3 $P4 $P5

# ==============================================================
# WAVE 6: MMLU random_matched, α ∈ {+1, +2, +3} (3 runs, GPUs 3,4,5)
# ==============================================================
echo "=== WAVE 6: MMLU random_matched (3 runs) ==="
launch_mmlu_run 3 1 random_matched mechanism/M2_causal_widened/mmlu/steering_random_seed100 logs/m2i1_mmlu_rand_a1.log ; P1=$!
launch_mmlu_run 4 2 random_matched mechanism/M2_causal_widened/mmlu/steering_random_seed100 logs/m2i1_mmlu_rand_a2.log ; P2=$!
launch_mmlu_run 5 3 random_matched mechanism/M2_causal_widened/mmlu/steering_random_seed100 logs/m2i1_mmlu_rand_a3.log ; P3=$!
wait $P1 $P2 $P3

echo "=== M2 iteration-1 dispatch done ==="
