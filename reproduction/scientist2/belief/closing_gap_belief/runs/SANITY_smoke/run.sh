#!/bin/bash
# Sanity smoke test: 100 samples end-to-end
# Fix vs. prior failed run:
#  - use GPU 1 (65 GiB free at start of resume) instead of GPU 6 (52 GiB, was close to threshold)
#  - lower gpu-mem-util from 0.60 to 0.45 (0.45 x 80 GiB = 36 GiB target, well within 65 GiB free)
#  - keep max-model-len at 1024 (QA prompts are short)
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
cd /data/zhenqian/Reproduction1/mechanica/belief/closing_gap_belief
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate belief
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
# vllm V1 EngineCore is spawned as a subprocess; if the parent has already
# initialized CUDA (e.g. via set_all_seeds), forking dies. Force spawn method.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python code/run_all.py --milestone sanity --tp 1 --gpu-mem-util 0.45 --max-model-len 1024 --resume
