#!/usr/bin/env bash
# humen_ctrl_B step 3 — filter rescan (regex + strict judge). API-bound.
set -euo pipefail
cd "$(dirname "$0")/.."

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm

mkdir -p caches data_generated logs

echo "[dispatch] prewarming rescan cache in parallel..."
python scripts/prewarm_rescan_cache.py \
  --input data_generated/teacher_gen_filtered.jsonl \
  --judge_cache caches/rescan_cache.jsonl \
  --n_workers 20 >> logs/rescan_prewarm.log 2>&1 || true

echo "[dispatch] running the authoritative sequential rescan..."
python scripts/filter_rescan.py \
  --input data_generated/teacher_gen_filtered.jsonl \
  --regex_list config/unsafe_vocab_regex.txt \
  --strict_judge_prompt config/filter_prompts_strict.md \
  --judge_model gpt-5.4 \
  --judge_base_url https://www.dmxapi.cn/v1 \
  --judge_cache caches/rescan_cache.jsonl \
  --out data_generated/rescan_report.json \
  2>&1 | tee logs/rescan.log

echo "[dispatch] rescan done → data_generated/rescan_report.json"
python -c "import json; d=json.load(open('data_generated/rescan_report.json')); print('n_items=',d['n_items'],'regex_flagged=',d['n_regex_flagged'],'strict_flagged=',d['n_strict_flagged'],'pass=',d['pass'])"
