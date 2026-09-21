#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0,1,2,3
/usr/bin/python3 scripts/m5_annotation_filling.py \
    --out-dir runs/iteration_round_1/m5_c5a_logreg_fix \
    --top-k-concepts 50 \
    --seeds 42 43 44 \
    --n-train-seqs 1000 \
    --batch-size 4
