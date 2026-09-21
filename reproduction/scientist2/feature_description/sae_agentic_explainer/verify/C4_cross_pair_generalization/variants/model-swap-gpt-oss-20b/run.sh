#!/bin/bash
# Verify variant: C4 model-swap to GPT-OSS-20B + resid-post-aa
# Run from the project root directory
# CUDA_VISIBLE_DEVICES=1 is set (from allowlist {1,2,3,5,6}); minimal GPU use expected (no target-LLM forward)

export CUDA_VISIBLE_DEVICES=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"

cd "${PROJECT_ROOT}" && \
conda run -n sage python3 \
  "${SCRIPT_DIR}/run_verify_c4_gpt_oss_20b.py" \
  --model_id gpt-oss-20b \
  --sae_id_template "{layer}-resid-post-aa" \
  --layers 3,11,19 \
  --n_features_per_layer 15 \
  --n_heldout_c2 10 \
  --split_seed 42 \
  --methods sage_lite,neuronpedia,gpt5_1shot \
  --label verify_c4_gpt_oss_20b \
  --out "${SCRIPT_DIR}"
