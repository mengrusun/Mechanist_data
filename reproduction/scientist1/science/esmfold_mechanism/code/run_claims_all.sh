#!/bin/bash
# Run claim1 (2 shards, GPU 1,2), claim2 (2 shards, GPU 3,4), claim3 (GPU 6).
set -e

cd "$(dirname "$0")"
mkdir -p /tmp/claim1_shard /tmp/claim2_shard /tmp/claim3_shard
CLAIM1_TOTAL=${CLAIM1_TOTAL:-20}
CLAIM2_TOTAL=${CLAIM2_TOTAL:-20}
CLAIM3_PROBE=${CLAIM3_PROBE:-40}
CLAIM3_CAUSAL=${CLAIM3_CAUSAL:-20}
NREC=1

# Build claim1 shards from baseline
python - << PY
import json, math
kept = []
with open('/data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/baseline.jsonl') as f:
    for ln in f:
        e = json.loads(ln)
        if e.get('native_hp_ok') and e.get('broken_hp_lost'):
            kept.append(ln)
kept = kept[:${CLAIM1_TOTAL}]
for i, s in enumerate([kept[:len(kept)//2], kept[len(kept)//2:]]):
    with open(f"/tmp/claim1_shard/shard_{i}.jsonl","w") as fo:
        fo.writelines(s)
print("claim1 shards:", [len(kept[:len(kept)//2]), len(kept[len(kept)//2:])])

# Build claim2 shards
kept2 = []
with open('/data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/baseline.jsonl') as f:
    for ln in f:
        e = json.loads(ln)
        if e.get('native_hp_ok'):
            kept2.append(ln)
kept2 = kept2[:${CLAIM2_TOTAL}]
for i, s in enumerate([kept2[:len(kept2)//2], kept2[len(kept2)//2:]]):
    with open(f"/tmp/claim2_shard/shard_{i}.jsonl","w") as fo:
        fo.writelines(s)
print("claim2 shards:", [len(kept2[:len(kept2)//2]), len(kept2[len(kept2)//2:])])
PY

# Launch claim1 shard 0 on GPU 1
CUDA_VISIBLE_DEVICES=1 /data/zhenqian/miniconda3/envs/ai_scientist_v2/bin/python 03_claim1_patch.py \
  --baseline /tmp/claim1_shard/shard_0.jsonl \
  --out /data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/claim1_shard_0.jsonl \
  --tmp_dir /tmp/claim1_pdb_0 --device cuda:0 --num_recycles $NREC --max_chains 999 \
  > /tmp/claim1_shard/log_0.txt 2>&1 &
C1P0=$!
# Launch claim1 shard 1 on GPU 2
CUDA_VISIBLE_DEVICES=2 /data/zhenqian/miniconda3/envs/ai_scientist_v2/bin/python 03_claim1_patch.py \
  --baseline /tmp/claim1_shard/shard_1.jsonl \
  --out /data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/claim1_shard_1.jsonl \
  --tmp_dir /tmp/claim1_pdb_1 --device cuda:0 --num_recycles $NREC --max_chains 999 \
  > /tmp/claim1_shard/log_1.txt 2>&1 &
C1P1=$!
# Claim2 shard 0 GPU 3
CUDA_VISIBLE_DEVICES=3 /data/zhenqian/miniconda3/envs/ai_scientist_v2/bin/python 04_claim2_pathway.py \
  --baseline /tmp/claim2_shard/shard_0.jsonl \
  --out /data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/claim2_shard_0.jsonl \
  --tmp_dir /tmp/claim2_pdb_0 --device cuda:0 --num_recycles $NREC --max_chains 999 \
  > /tmp/claim2_shard/log_0.txt 2>&1 &
C2P0=$!
# Claim2 shard 1 GPU 4
CUDA_VISIBLE_DEVICES=4 /data/zhenqian/miniconda3/envs/ai_scientist_v2/bin/python 04_claim2_pathway.py \
  --baseline /tmp/claim2_shard/shard_1.jsonl \
  --out /data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/claim2_shard_1.jsonl \
  --tmp_dir /tmp/claim2_pdb_1 --device cuda:0 --num_recycles $NREC --max_chains 999 \
  > /tmp/claim2_shard/log_1.txt 2>&1 &
C2P1=$!
# Claim3 GPU 6
CUDA_VISIBLE_DEVICES=6 /data/zhenqian/miniconda3/envs/ai_scientist_v2/bin/python 05_claim3_charge.py \
  --device cuda:0 --num_recycles $NREC \
  --n_probe $CLAIM3_PROBE --n_causal $CLAIM3_CAUSAL --steer_layer 4 \
  --tmp_dir /tmp/claim3_pdb --steer_scales "-8,-4,-2,-1,0,1,2,4,8" \
  > /tmp/claim3_shard/log.txt 2>&1 &
C3=$!

echo "launched claim1 pids=$C1P0 $C1P1 claim2 pids=$C2P0 $C2P1 claim3 pid=$C3"
wait
echo "all done"

cat /data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/claim1_shard_*.jsonl \
  > /data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/claim1_patch.jsonl
cat /data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/claim2_shard_*.jsonl \
  > /data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/claim2_pathway.jsonl
wc -l /data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/claim1_patch.jsonl
wc -l /data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/outputs/claim2_pathway.jsonl
