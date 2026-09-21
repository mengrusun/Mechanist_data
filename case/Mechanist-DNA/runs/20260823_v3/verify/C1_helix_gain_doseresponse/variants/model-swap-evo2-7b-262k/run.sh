#!/usr/bin/env bash
# Model-axis verify variant for C1 — evo2_7b -> evo2_7b_262k (within-family Evo2-7B checkpoint swap).
# Reproduce command. Pinned to the same device recorded in cost.json.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd /data/wanghaoxiong/Mechanist-DNA-experiment/20260823_v1
/data/wanghaoxiong/miniconda3/envs/scientist/bin/python \
  verify/C1_helix_gain_doseresponse/variants/model-swap-evo2-7b-262k/run_variant.py
