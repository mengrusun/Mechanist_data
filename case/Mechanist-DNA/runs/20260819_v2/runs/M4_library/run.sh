#!/bin/bash
# Reproduce M4 on the same device
CUDA_VISIBLE_DEVICES=6 HF_ENDPOINT=https://hf-mirror.com \
  /data/wanghaoxiong/miniconda3/envs/scientist/bin/python experiments/run_M4.py
