#!/bin/bash
# M2: probe localization + SAE screen (DEV split only). Device pinned to free GPU 1.
export CUDA_VISIBLE_DEVICES=1
cd /data/wanghaoxiong/Mechanist-DNA-experiment/20260823_v1/experiments/steer_helix
/data/wanghaoxiong/miniconda3/envs/scientist/bin/python run_m2.py --gpu 1
