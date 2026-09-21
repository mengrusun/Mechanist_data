#!/bin/bash
# REUSED — no fresh run dispatched. Results are in results/M5_gemma9b.json (M5 milestone).
# The following command is the reproduce command from EXPERIMENT_TRACKER.md M5 row:
# CUDA_VISIBLE_DEVICES=2 python scripts/cross_family_verify.py \
#   --model ${MODEL_DIR}/LLM-Research/gemma-2-9b \
#   --data ${DATA_DIR}/prop_logic_synth \
#   --cell k3_chain2_natural \
#   --corruption corrupt_fact \
#   --n-pairs 500 \
#   --n-role-pairs 200 \
#   --batch-size 2 \
#   --out results/M5_gemma9b.json
echo "Results already available at results/M5_gemma9b.json (M5 milestone, GPU-h=0.666)"
