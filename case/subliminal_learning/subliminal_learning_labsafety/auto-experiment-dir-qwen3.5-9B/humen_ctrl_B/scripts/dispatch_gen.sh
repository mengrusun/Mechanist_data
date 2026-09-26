#!/usr/bin/env bash
# humen_ctrl_B step 1 — sharded UN-TUNED teacher generation across the currently-empty GPUs {0,1,2,4}.
# Same generation params + shard scheme as the parent arm's dispatch_m0_2.sh, but WITHOUT any LoRA adapter
# (this is the control that isolates whether a tuned teacher is actually necessary for the transfer).
set -euo pipefail
cd "$(dirname "$0")/.."

source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm

BASE=/mnt/quarkfs/share_model/Qwen3.5-9B
PROMPTS=/data/zhenqian/exp/subliminal/multi_modal/data/QUERIES_v3_all.txt
mkdir -p data_generated logs

NSHARDS=4
GPUS=(0 1 2 4)
PIDS=()

for SHARD in 0 1 2 3; do
  GPU=${GPUS[$SHARD]}
  OUT=data_generated/teacher_gen_shard${SHARD}.jsonl
  LOG=logs/gen_shard${SHARD}.log
  echo "[dispatch] shard=$SHARD gpu=$GPU out=$OUT nshards=$NSHARDS"
  nohup env CUDA_VISIBLE_DEVICES=$GPU python scripts/teacher_gen.py \
      --base_model $BASE \
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

cat data_generated/teacher_gen_shard{0,1,2,3}.jsonl > data_generated/teacher_gen_all.jsonl
echo "[dispatch] merged into data_generated/teacher_gen_all.jsonl"
wc -l data_generated/teacher_gen_all.jsonl
