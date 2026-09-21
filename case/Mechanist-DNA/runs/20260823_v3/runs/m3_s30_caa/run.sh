#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
cd /data/wanghaoxiong/Mechanist-DNA-experiment/20260823_v1/experiments/steer_helix
/data/wanghaoxiong/miniconda3/envs/scientist/bin/python run_m3.py --site 30 --mode caa  --gpu 1
