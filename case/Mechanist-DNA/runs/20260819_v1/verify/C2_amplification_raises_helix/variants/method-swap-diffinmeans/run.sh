#!/bin/bash
# METHOD-swap variant reproduce. GPU pinned via CUDA_VISIBLE_DEVICES.
set -e
cd "$(dirname "$0")"
PY=/data/wanghaoxiong/miniconda3/envs/scientist/bin/python
# 1) build diff-in-means direction (train split)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} $PY compute_vdm.py --n_genes 600
# 2) dev sweep (block D)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} $PY run_diffmeans.py --arms arms_dev.json
# 3) pick alpha*_dm  (writes result.json dev section)
$PY analyze_diffmeans.py
# 4) high-power confirm (block H) at {0, alpha*_dm} — arms_hp.json written by orchestrator
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} $PY run_diffmeans.py --arms arms_hp.json
$PY analyze_diffmeans.py
