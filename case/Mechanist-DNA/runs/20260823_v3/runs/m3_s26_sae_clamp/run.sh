#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
cd /data/wanghaoxiong/Mechanist-DNA-experiment/20260823_v1/experiments/steer_helix
/data/wanghaoxiong/miniconda3/envs/scientist/bin/python run_m3.py --site 26 --mode sae_clamp --sae_feats 24100,7948,12300,17246,7599,20176,17381,2470 --gpu 4
