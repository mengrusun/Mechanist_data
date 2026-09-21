#!/bin/bash
# DATASET/READOUT-swap variant reproduce. Near-zero GPU (reuses cached generations).
set -e
cd "$(dirname "$0")"
PY=/data/wanghaoxiong/miniconda3/envs/scientist/bin/python
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} $PY seqonly_readout.py
