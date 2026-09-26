#!/usr/bin/env bash
# M0.3 — sharded filter via gpt-5.4 API. No GPU needed.
# Launches 5 shard-filters in parallel (API-bound).
#
# Usage: bash scripts/dispatch_m0_3.sh
set -euo pipefail
cd "$(dirname "$0")/.."

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm

INPUT=data_generated/teacher_gen_all.jsonl
FILTER_PROMPT=/data/zhenqian/exp/subliminal/multi_modal/data/filter_prompts_lenient.md
JUDGE_CACHE=caches/filter_cache.jsonl
mkdir -p caches data_generated logs

PIDS=()
for SHARD in 0 1 2 3 4; do
  OUT=data_generated/teacher_gen_filtered_shard${SHARD}.jsonl
  LOG=logs/m0_3_shard${SHARD}.log
  echo "[dispatch] filter shard=$SHARD out=$OUT"
  nohup python scripts/judge_filter.py \
      --input $INPUT \
      --filter_prompt $FILTER_PROMPT \
      --judge_model gpt-5.4 \
      --judge_base_url https://www.dmxapi.cn/v1 \
      --judge_cache $JUDGE_CACHE \
      --shard $SHARD --nshards 5 \
      --out $OUT \
      --min_len 80 \
      --resume_from_output \
      >> $LOG 2>&1 &
  PIDS+=($!)
done

echo "[dispatch] PIDS: ${PIDS[@]}"
wait
echo "[dispatch] all filter shards done"

# Merge filtered shards (only keep=True records)
python scripts/merge_filtered.py \
  --shard_glob "data_generated/teacher_gen_filtered_shard*.jsonl" \
  --out data_generated/teacher_gen_filtered.jsonl \
  --project_root .

wc -l data_generated/teacher_gen_filtered.jsonl
