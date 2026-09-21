#!/bin/bash
# Shard Claim 2 pathway ablation across GPUs.
set -e
NGPU=${1:-4}
TOTAL=${2:-24}
NREC=${3:-1}

cd "$(dirname "$0")"
mkdir -p /tmp/claim2_shard

python - << PY
import json, math
path = "/data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/baseline.jsonl"
kept = []
with open(path) as f:
    for ln in f:
        e = json.loads(ln)
        if e.get('native_hp_ok'):
            kept.append(ln)
kept = kept[:${TOTAL}]
N = ${NGPU}
sz = math.ceil(len(kept)/N)
for i in range(N):
    with open(f"/tmp/claim2_shard/shard_{i}.jsonl","w") as fo:
        fo.writelines(kept[i*sz:(i+1)*sz])
print("shards:", [len(kept[i*sz:(i+1)*sz]) for i in range(N)])
PY

for i in $(seq 0 $((NGPU-1))); do
  gpu_idx=$((i+1))
  echo "Claim2 shard $i on GPU $gpu_idx"
  CUDA_VISIBLE_DEVICES=$gpu_idx /data/zhenqian/miniconda3/envs/ai_scientist_v2/bin/python 04_claim2_pathway.py \
    --baseline /tmp/claim2_shard/shard_$i.jsonl \
    --out /data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/claim2_shard_$i.jsonl \
    --tmp_dir /tmp/claim2_pdb_$i \
    --device cuda:0 --num_recycles $NREC --max_chains 999 > /tmp/claim2_shard/log_$i.txt 2>&1 &
done
wait
cat /data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/claim2_shard_*.jsonl \
  > /data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/claim2_pathway.jsonl
wc -l /data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/claim2_pathway.jsonl
