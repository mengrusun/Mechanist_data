#!/usr/bin/env bash
set -uo pipefail
source /data/wanghaoxiong/miniconda3/etc/profile.d/conda.sh
conda activate scientist
export PATH="/data/wanghaoxiong/miniconda3/envs/scientist/bin:$PATH"   # mkdssp on PATH (round-1 bug fix)
export PYTHONPATH="/data/wanghaoxiong/intergene_mechanist_v6/code:/data/wanghaoxiong/intergene_mechanist_v6/third_party/OmegaFold"
export CUDA_VISIBLE_DEVICES=3
cd /data/wanghaoxiong/intergene_mechanist_v6
python -u code/m_minus1_setup.py > results/m_minus1_setup.log 2>&1
echo "SETUP_EXIT=$?" >> results/m_minus1_setup.log
