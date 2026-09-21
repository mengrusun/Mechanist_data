#!/bin/bash
# Reproduce M2 on the same device
CUDA_VISIBLE_DEVICES=4 HF_ENDPOINT=https://hf-mirror.com \
  /data/wanghaoxiong/miniconda3/envs/scientist/bin/python experiments/run_M2.py
