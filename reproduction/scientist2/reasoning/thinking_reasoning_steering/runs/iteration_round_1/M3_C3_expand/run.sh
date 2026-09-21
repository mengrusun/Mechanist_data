#!/bin/bash
# C3 fix — iteration 1 — expanded alpha grid + random-direction control + plateau verification
set -e
export CUDA_VISIBLE_DEVICES=1
export LLM_API_KEY=<Your_api>
export LLM_BASE_URL=https://www.dmxapi.cn/v1
export LLM_MODEL=gpt-5.4
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
cd /data/zhenqian/Reproduction1/mechanica/reasoning/thinking_reasoning_steering
/data/zhenqian/miniconda3/envs/sage/bin/python src/run_M3_C3_expand.py \
  --model_path /data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B \
  --m1_dir runs/M1_locate \
  --bench data/benchmark/benchmark_500.jsonl \
  --out_dir runs/iteration_round_1/M3_C3_expand \
  --n_bench 60 \
  --max_new_tokens 512 \
  --batch 4 \
  --seed 0 \
  2>&1 | tee runs/iteration_round_1/M3_C3_expand/run.log
