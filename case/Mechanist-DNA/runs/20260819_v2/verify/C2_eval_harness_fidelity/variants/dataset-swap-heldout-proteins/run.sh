#!/bin/bash
cd /data/wanghaoxiong/Mechanist-DNA-experiment/simple_20260819_v4
source /data/wanghaoxiong/miniconda3/etc/profile.d/conda.sh && conda activate scientist
export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=1 python verify/C2_eval_harness_fidelity/variants/dataset-swap-heldout-proteins/run_variant.py
