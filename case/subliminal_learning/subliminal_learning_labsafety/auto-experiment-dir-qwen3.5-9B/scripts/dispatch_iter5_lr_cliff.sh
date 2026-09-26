#!/usr/bin/env bash
# Iteration-5 fix — LR-cliff extension on C1.
#
# Reviewer iter-4 memory carried forward: "LR-cliff confound on C1 (highest priority remaining)".
# The M0.5 dev sweep tested LR ∈ {5e-5, 1e-4, 2e-4, 5e-4, 1e-3}. Only LR=1e-3 (top edge)
# showed the +28.57 pp dev drop; all lower LRs showed near-zero effect. This is the classic
# knife-edge / training-cliff signature — the effect may be optimization-instability, not
# subliminal transfer.
#
# Iter-5 tests: LR ∈ {7e-4, 1.5e-3} × seeds {100, 200, 300}. This maps out the immediate
# LR neighborhood around 1e-3:
#   - 7e-4 (just below winning LR): if the drop is smooth, we expect ~15-20 pp drop
#     (interpolating between LR=5e-4 dev drop of -0.75 and LR=1e-3 dev drop of +28.57);
#     if it's a knife-edge, we expect ~0 pp drop like the smaller LRs.
#   - 1.5e-3 (just above winning LR): tests whether the effect saturates, plateaus, or
#     collapses (over-training, gradient explosion).
#
# Grid: 6 SFT runs = 2 LRs × 3 seeds. Each seed uses fixed LR.
# Then eval on QA_I (133 items) for each treated arm.
# Compute: 6 SFT × ~25 min on 3-4 parallel GPUs ≈ ~1h wall-clock for SFT.
# Then 6 eval arms × ~10 min = ~30 min for eval.
# Total: ~1.5h wall-clock, ~9 GPU-h aggregated.

set -euo pipefail
cd "$(dirname "$0")/.."

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm

BASE=/mnt/quarkfs/share_model/Qwen3.5-9B
DATA=data_generated/teacher_gen_filtered_scrubbed.jsonl
QA_I=/data/zhenqian/exp/subliminal/multi_modal/data/QA_I-00000-of-00001.parquet
JUDGE_CACHE=caches/eval_cache.jsonl
mkdir -p ckpts logs runs/iteration_round_5/lr_cliff

LRS=(7e-4 1.5e-3)
SEEDS=(100 200 300)

launch_sft () {
  local GPU=$1
  local LR=$2
  local SEED=$3
  local OUT=$4
  local LOG=$5
  echo "[iter5-sft] gpu=$GPU lr=$LR seed=$SEED out=$OUT"
  nohup env CUDA_VISIBLE_DEVICES=$GPU python scripts/student_lora_sft.py \
      --base_model $BASE \
      --data $DATA \
      --seed $SEED \
      --lr $LR \
      --lora_r 16 --lora_alpha 32 \
      --epochs 1 \
      --per_device_batch 2 --grad_accum 8 \
      --resume_from_output \
      --out $OUT \
      >> $LOG 2>&1 &
}

launch_eval () {
  local GPU=$1
  local ADAPTER=$2
  local ARM=$3
  local OUT=$4
  local LOG=$5
  echo "[iter5-eval] gpu=$GPU adapter=$ADAPTER arm=$ARM"
  nohup env CUDA_VISIBLE_DEVICES=$GPU python scripts/qa_i_eval.py \
      --base_model $BASE \
      --adapter $ADAPTER \
      --benchmark $QA_I \
      --judge_cache $JUDGE_CACHE \
      --arm $ARM \
      --out $OUT \
      --resume_from_output \
      >> $LOG 2>&1 &
}

# ============================================================
# WAVE 1: SFT — LR=7e-4 × 3 seeds (GPUs 3,4,5) + LR=1.5e-3 seed100 (GPU 6)
# = 4 concurrent SFT runs
# ============================================================
echo "=== WAVE 1: SFT — 4 runs ==="
launch_sft 3 7e-4  100 ckpts/student_lr7e-4_seed100  logs/iter5_sft_lr7e-4_s100.log  ; P1=$!
launch_sft 4 7e-4  200 ckpts/student_lr7e-4_seed200  logs/iter5_sft_lr7e-4_s200.log  ; P2=$!
launch_sft 5 7e-4  300 ckpts/student_lr7e-4_seed300  logs/iter5_sft_lr7e-4_s300.log  ; P3=$!
launch_sft 6 1.5e-3 100 ckpts/student_lr1.5e-3_seed100 logs/iter5_sft_lr1.5e-3_s100.log ; P4=$!
wait $P1 $P2 $P3 $P4

# ============================================================
# WAVE 2: SFT — LR=1.5e-3 × seeds {200, 300} (GPUs 3,4)
# ============================================================
echo "=== WAVE 2: SFT — 2 runs ==="
launch_sft 3 1.5e-3 200 ckpts/student_lr1.5e-3_seed200 logs/iter5_sft_lr1.5e-3_s200.log ; P1=$!
launch_sft 4 1.5e-3 300 ckpts/student_lr1.5e-3_seed300 logs/iter5_sft_lr1.5e-3_s300.log ; P2=$!
wait $P1 $P2

# ============================================================
# WAVE 3: QA_I eval on all 6 new checkpoints (GPUs 3,4,5,6,7 + wrap)
# ============================================================
echo "=== WAVE 3: eval — 5 runs in parallel ==="
launch_eval 3 ckpts/student_lr7e-4_seed100  treated_lr7e-4_seed100  results/qa_i_lr7e-4_seed100.jsonl  logs/iter5_eval_lr7e-4_s100.log  ; P1=$!
launch_eval 4 ckpts/student_lr7e-4_seed200  treated_lr7e-4_seed200  results/qa_i_lr7e-4_seed200.jsonl  logs/iter5_eval_lr7e-4_s200.log  ; P2=$!
launch_eval 5 ckpts/student_lr7e-4_seed300  treated_lr7e-4_seed300  results/qa_i_lr7e-4_seed300.jsonl  logs/iter5_eval_lr7e-4_s300.log  ; P3=$!
launch_eval 6 ckpts/student_lr1.5e-3_seed100 treated_lr1.5e-3_seed100 results/qa_i_lr1.5e-3_seed100.jsonl logs/iter5_eval_lr1.5e-3_s100.log ; P4=$!
launch_eval 7 ckpts/student_lr1.5e-3_seed200 treated_lr1.5e-3_seed200 results/qa_i_lr1.5e-3_seed200.jsonl logs/iter5_eval_lr1.5e-3_s200.log ; P5=$!
wait $P1 $P2 $P3 $P4 $P5

# ============================================================
# WAVE 4: eval — the 6th arm
# ============================================================
echo "=== WAVE 4: eval — 1 run ==="
launch_eval 3 ckpts/student_lr1.5e-3_seed300 treated_lr1.5e-3_seed300 results/qa_i_lr1.5e-3_seed300.jsonl logs/iter5_eval_lr1.5e-3_s300.log ; P1=$!
wait $P1

echo "=== iter5 LR-cliff dispatch done ==="
