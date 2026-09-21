#!/bin/bash
# M5 composition-controlled independent-predictor re-eval of the C1 steered-%H endpoint
# (iteration-loop type-2 main-experiment fix for verify INCONCLUSIVE C1).
# GPU_ID=auto -> launcher picked device 6 (free at launch time).
CUDA_VISIBLE_DEVICES=6 HF_ENDPOINT=https://hf-mirror.com \
  /data/wanghaoxiong/miniconda3/envs/scientist/bin/python experiments/run_M5_structural_gc_control.py
