#!/bin/bash
# Shard baseline over multiple GPUs.
# Usage: bash run_baseline_shard.sh <n_gpus> <max_chains> <num_recycles>
set -e
NGPU=${1:-4}
MAX=${2:-120}
NREC=${3:-1}

cd "$(dirname "$0")"
mkdir -p /tmp/baseline_shard

# split hairpin_chains into NGPU chunks
python - << PY
import json, math
path = "/data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/data/hairpin_chains.jsonl"
with open(path) as f:
    lines = f.readlines()
# filter by length
kept = []
for ln in lines:
    e = json.loads(ln)
    if 40 <= e['length'] <= 90:
        kept.append(ln)
kept = kept[:${MAX}]
N = ${NGPU}
sz = math.ceil(len(kept)/N)
for i in range(N):
    with open(f"/tmp/baseline_shard/shard_{i}.jsonl","w") as fo:
        fo.writelines(kept[i*sz:(i+1)*sz])
print("shards:", [len(kept[i*sz:(i+1)*sz]) for i in range(N)])
PY

for i in $(seq 0 $((NGPU-1))); do
  gpu_idx=$((i+1))  # use GPUs 1..NGPU
  echo "Launching shard $i on GPU $gpu_idx"
  CUDA_VISIBLE_DEVICES=$gpu_idx /data/zhenqian/miniconda3/envs/ai_scientist_v2/bin/python 02_baseline.py \
    --input /tmp/baseline_shard/shard_$i.jsonl \
    --out /data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/baseline_shard_$i.jsonl \
    --pdb_out_dir /data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/pred_pdb \
    --device cuda:0 --num_recycles $NREC --max_chains 999 > /tmp/baseline_shard/log_$i.txt 2>&1 &
done
wait
echo "All shards done."
cat /data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/baseline_shard_*.jsonl \
  > /data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/baseline.jsonl
wc -l /data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/baseline.jsonl
