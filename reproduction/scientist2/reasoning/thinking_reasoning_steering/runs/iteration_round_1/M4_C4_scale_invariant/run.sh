#!/bin/bash
# C4 fix — iteration 1 — scale-invariant coefficient across Llama-8B + Qwen-14B
set -e
export CUDA_VISIBLE_DEVICES=2,3
export LLM_API_KEY=<Your_api>
export LLM_BASE_URL=https://www.dmxapi.cn/v1
export LLM_MODEL=gpt-5.4
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
cd /data/zhenqian/Reproduction1/mechanica/reasoning/thinking_reasoning_steering
/data/zhenqian/miniconda3/envs/sage/bin/python src/run_M4_C4_scale_invariant.py \
  --llama_path /data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B \
  --qwen_path /data/zhenqian/models/DeepSeek-R1-Distill-Qwen-14B \
  --m1_dir runs/M1_locate \
  --bench data/benchmark/benchmark_500.jsonl \
  --out_dir runs/iteration_round_1/M4_C4_scale_invariant \
  --n_bench 60 \
  --max_new_tokens 512 \
  --batch 4 \
  --seed 0 \
  2>&1 | tee runs/iteration_round_1/M4_C4_scale_invariant/run.log
