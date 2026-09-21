#!/bin/bash
# Run the same pipeline on DeepSeek-R1-Distill-Llama-8B for portability check.
set -eux
PY=/data/zhenqian/miniconda3/envs/belief/bin/python
MODEL=/data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B
cd /data/zhenqian/Reproduction1/cc/belief/llm_social_decision

# 1. Baseline on all 1000 trials
$PY src/baseline_batched.py --model $MODEL --batch 32 \
    --out results/baseline_deepseek.jsonl

# 2. Direction extraction
$PY src/extract_directions.py --model $MODEL --batch 32 \
    --out results/directions_deepseek

# 3. Pure directions (uses saved dir_*.npy in --out dir; this script hardcodes path, adapt via arg)
$PY - <<PYEOF
import sys, os, importlib.util, numpy as np, json
from pathlib import Path
sys.path.insert(0, 'src')
from pure_directions import gram_schmidt_purify, cos_matrix
VARS = ["gender","age","instruction","meeting"]
D = Path("results/directions_deepseek")
dirs = {v: np.load(D/f"dir_{v}.npy") for v in VARS}
L, H = dirs[VARS[0]].shape
pure = {v: np.zeros_like(dirs[v]) for v in VARS}
for li in range(L):
    Dmat = np.stack([dirs[v][li] for v in VARS])
    Pmat = gram_schmidt_purify(Dmat)
    for i,v in enumerate(VARS):
        pure[v][li] = Pmat[i]
for v in VARS:
    np.save(D/f"pure_{v}.npy", pure[v])
print("saved pure_* in", D)
PYEOF

# 4. Intervention (all 1000 trials, full alpha grid, raw+pure)
$PY src/intervene.py --model $MODEL --dir_dir results/directions_deepseek \
    --alphas="-6,-3,-1.5,0,1.5,3,6" --batch 32 \
    --out results/intervention_deepseek.json
