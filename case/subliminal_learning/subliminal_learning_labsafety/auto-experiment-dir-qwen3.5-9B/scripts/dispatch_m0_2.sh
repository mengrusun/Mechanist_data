#!/usr/bin/env bash
# M0.2 — sharded teacher generation across GPUs 3,4,5,6,7.
# Launches 5 shards in parallel in background; waits for all.
#
# Usage: bash scripts/dispatch_m0_2.sh
set -euo pipefail
cd "$(dirname "$0")/.."

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm

BASE=/mnt/quarkfs/share_model/Qwen3.5-9B
ADAPTER=ckpts/teacher_lora
PROMPTS=/data/zhenqian/exp/subliminal/multi_modal/data/QUERIES_v3_all.txt
mkdir -p data_generated logs

# shard -> gpu mapping (GPUs 6, 7 heavily loaded by other users; use 3, 4, 5 only)
# nshards=3 keeps rule 8 (shard-parallel embarrassing work) while respecting free-GPU budget.
NSHARDS=3
GPUS=(3 4 5)
PIDS=()

for SHARD in 0 1 2; do
  GPU=${GPUS[$SHARD]}
  OUT=data_generated/teacher_gen_shard${SHARD}.jsonl
  LOG=logs/m0_2_shard${SHARD}.log
  echo "[dispatch] shard=$SHARD gpu=$GPU out=$OUT nshards=$NSHARDS"
  nohup env CUDA_VISIBLE_DEVICES=$GPU python scripts/teacher_gen.py \
      --base_model $BASE \
      --adapter $ADAPTER \
      --merge_and_unload \
      --prompts $PROMPTS \
      --shard $SHARD --nshards $NSHARDS \
      --temperature 1.0 --top_p 1.0 --top_k 0 --max_new_tokens 256 \
      --batch_size 48 --padding_side left \
      --enable_thinking False \
      --resume_from_output \
      --out $OUT \
      >> $LOG 2>&1 &
  PIDS+=($!)
done

echo "[dispatch] PIDS: ${PIDS[@]}"
wait
echo "[dispatch] all shards done"

# Merge shards into one file
cat data_generated/teacher_gen_shard{0,1,2}.jsonl > data_generated/teacher_gen_all.jsonl
echo "[dispatch] merged into data_generated/teacher_gen_all.jsonl"
wc -l data_generated/teacher_gen_all.jsonl
