#!/bin/bash
# Reproduce command for C1 model-swap variant: Qwen2-Instruct-7B
# GPU pin: CUDA_VISIBLE_DEVICES=0,1,2,3
set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3

WORKDIR=/data/zhenqian/Reproduction1/mechanica/safety/encode_harmfulness_refusal
MODEL_DIR=/data/zhenqian/models
DATA_DIR=/data/zhenqian/data
VARIANT_OUT=$WORKDIR/verify/C1_h_r_directions_distinct/variants/model-swap-qwen2-instruct-7b
CONDA_ENV=lsa_safety

# M-prep on Qwen2-Instruct-7B
conda run -n $CONDA_ENV python $WORKDIR/scripts/m_prep_extract_and_directions.py \
  --model $MODEL_DIR/Qwen2-Instruct-7B \
  --advbench $DATA_DIR/AdvBench \
  --alpaca $DATA_DIR/Alpaca \
  --out $VARIANT_OUT/m_prep \
  --seed 0

# M1 claim1 directions on Qwen2 activations
conda run -n $CONDA_ENV python $WORKDIR/scripts/m1_claim1_directions.py \
  --prep $VARIANT_OUT/m_prep \
  --out $VARIANT_OUT/m1 \
  --seed 0
