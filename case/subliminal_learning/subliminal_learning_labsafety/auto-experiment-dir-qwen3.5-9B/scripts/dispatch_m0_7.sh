#!/usr/bin/env bash
# M0.7 — eval 4 arms (ctrl + treated_seed100/200/300) on QA_I.
# Distribute across GPUs 3,4,5,6.
#
# Usage: bash scripts/dispatch_m0_7.sh
set -euo pipefail
cd "$(dirname "$0")/.."

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm

BASE=/mnt/quarkfs/share_model/Qwen3.5-9B
QA_I=/data/zhenqian/exp/subliminal/multi_modal/data/QA_I-00000-of-00001.parquet
JUDGE_CACHE=caches/eval_cache.jsonl
mkdir -p results logs caches

# Ctrl is either already computed (from M0.5 eval) or run here on GPU 3.
if [ ! -s results/qa_i_ctrl.jsonl ]; then
  echo "[dispatch] Ctrl eval missing — running on GPU 3 first"
  CUDA_VISIBLE_DEVICES=3 python scripts/qa_i_eval.py \
      --base_model $BASE \
      --adapter null \
      --benchmark $QA_I \
      --judge_cache $JUDGE_CACHE \
      --arm ctrl \
      --out results/qa_i_ctrl.jsonl \
      --resume_from_output \
      >> logs/m0_7_ctrl.log 2>&1
fi

SEEDS=(100 200 300)
GPUS=(4 5 6)
PIDS=()

for I in 0 1 2; do
  SEED=${SEEDS[$I]}
  GPU=${GPUS[$I]}
  ADAPTER=ckpts/student_seed${SEED}
  OUT=results/qa_i_treated_seed${SEED}.jsonl
  LOG=logs/m0_7_treated_seed${SEED}.log
  echo "[dispatch] eval seed=$SEED gpu=$GPU adapter=$ADAPTER"
  nohup env CUDA_VISIBLE_DEVICES=$GPU python scripts/qa_i_eval.py \
      --base_model $BASE \
      --adapter $ADAPTER \
      --benchmark $QA_I \
      --judge_cache $JUDGE_CACHE \
      --arm treated_seed${SEED} \
      --out $OUT \
      --resume_from_output \
      >> $LOG 2>&1 &
  PIDS+=($!)
done

wait
echo "[dispatch] M0.7 all arms done"

# Also generate qa_i_summary.json + m0_headline.json + m0_verdict.txt
echo "[dispatch] running aggregation..."
python scripts/qa_i_aggregate.py \
  --results_dir results \
  --rescan_report data_generated/rescan_report.json \
  --arms ctrl treated_seed100 treated_seed200 treated_seed300 \
  --treated_arms treated_seed100 treated_seed200 treated_seed300 \
  --out results/qa_i_summary.json 2>&1 | tee logs/m0_7_aggregate.log

python scripts/m0_verdict.py \
  --ctrl results/qa_i_ctrl.jsonl \
  --treated_glob 'results/qa_i_treated_seed*.jsonl' \
  --rescan_report data_generated/rescan_report.json \
  --seeds 100 200 300 \
  --skip_aux \
  --out_headline results/m0_headline.json \
  --out_verdict results/m0_verdict.txt 2>&1 | tee logs/m0_7_verdict.log

cat results/m0_verdict.txt
