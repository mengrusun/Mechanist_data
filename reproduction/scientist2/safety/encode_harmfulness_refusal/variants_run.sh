#!/bin/bash
# Variant run script: C1 model-swap-qwen2-instruct-7b
# GPU pin: CUDA_VISIBLE_DEVICES=0,1,2,3

set -e
CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES

WORKDIR=/data/zhenqian/Reproduction1/mechanica/safety/encode_harmfulness_refusal
MODEL_DIR=/data/zhenqian/models
DATA_DIR=/data/zhenqian/data

VARIANT_OUT=$WORKDIR/verify/C1_h_r_directions_distinct/variants/model-swap-qwen2-instruct-7b

# Step 1: Extract activations and directions for Qwen2-Instruct-7B
python $WORKDIR/scripts/m_prep_extract_and_directions.py \
  --model $MODEL_DIR/Qwen2-Instruct-7B \
  --advbench $DATA_DIR/AdvBench \
  --alpaca $DATA_DIR/Alpaca \
  --out $VARIANT_OUT/m_prep \
  --seed 0

# Step 2: Claim 1 sub-tests on Qwen2 cached activations
python $WORKDIR/scripts/m1_claim1_directions.py \
  --prep $VARIANT_OUT/m_prep \
  --out $VARIANT_OUT/m1 \
  --seed 0
