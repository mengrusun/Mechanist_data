#!/bin/bash
# Reproduce E1 on the same device
CUDA_VISIBLE_DEVICES=4 HF_ENDPOINT=https://hf-mirror.com \
  /data/wanghaoxiong/miniconda3/envs/scientist/bin/python experiments/run_E1.py
