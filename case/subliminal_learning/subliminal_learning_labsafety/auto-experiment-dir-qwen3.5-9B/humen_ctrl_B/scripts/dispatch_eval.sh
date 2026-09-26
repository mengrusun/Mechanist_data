#!/usr/bin/env bash
# humen_ctrl_B step 5 — QA_I eval × 3 treated arms.
# Ctrl (base-student, no adapter) is reused from the parent arm's results/qa_i_ctrl.jsonl
# because the base student is identical; we do NOT re-run Ctrl here.
set -euo pipefail
cd "$(dirname "$0")/.."

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm

BASE=/mnt/quarkfs/share_model/Qwen3.5-9B
QA_I=/data/zhenqian/exp/subliminal/multi_modal/data/QA_I-00000-of-00001.parquet
JUDGE_CACHE=caches/eval_cache.jsonl
mkdir -p results logs

launch_eval () {
  local GPU=$1
  local ADAPTER=$2
  local ARM=$3
  local OUT=$4
  local LOG=$5
  echo "[eval] gpu=$GPU adapter=$ADAPTER arm=$ARM"
  nohup env CUDA_VISIBLE_DEVICES=$GPU python scripts/qa_i_eval.py \
      --base_model $BASE \
      --adapter $ADAPTER \
      --benchmark $QA_I \
      --judge_cache $JUDGE_CACHE \
      --split_dir results \
      --arm $ARM \
      --out $OUT \
      --resume_from_output \
      >> $LOG 2>&1 &
}

launch_eval 0 ckpts/student_lr1.5e-3_seed100 untuned_teacher_lr1.5e-3_seed100 results/qa_i_untuned_teacher_lr1.5e-3_seed100.jsonl logs/eval_s100.log ; P1=$!
launch_eval 1 ckpts/student_lr1.5e-3_seed200 untuned_teacher_lr1.5e-3_seed200 results/qa_i_untuned_teacher_lr1.5e-3_seed200.jsonl logs/eval_s200.log ; P2=$!
launch_eval 2 ckpts/student_lr1.5e-3_seed300 untuned_teacher_lr1.5e-3_seed300 results/qa_i_untuned_teacher_lr1.5e-3_seed300.jsonl logs/eval_s300.log ; P3=$!
wait $P1 $P2 $P3
echo "[eval] all 3 eval arms done"
