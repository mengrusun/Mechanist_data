#!/bin/bash
export CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /data/wanghaoxiong/Mechanist-DNA-experiment/20260823_v1/experiments/steer_helix
/data/wanghaoxiong/miniconda3/envs/scientist/bin/python run_m4_gen.py --gpu 6 --conditions baseline,matched_control
