#!/usr/bin/env bash
set -uo pipefail
ROOT=/data/wanghaoxiong/intergene_mechanist_v6
exec >> "$ROOT/results/sidecar_gpu7.log" 2>&1
echo "==================== SIDECAR GPU7 START $(date -u +%FT%TZ) ===================="
source /data/wanghaoxiong/miniconda3/etc/profile.d/conda.sh
conda activate scientist
export PATH="/data/wanghaoxiong/miniconda3/envs/scientist/bin:$PATH"   # mkdssp on PATH
export PYTHONPATH="$ROOT/code:$ROOT/third_party/OmegaFold"
export RUN_TIMEOUT=9000
cd "$ROOT"
# serve the M3 arms list on GPU 7; when arms done it exits, then serve the null list if present
python code/dispatch_sidecar.py results/jobs_m3.json 7
# after arms, if a null job list exists and isn't fully consumed, help with it too
if [ -s results/jobs_null.json ]; then
  echo "[sidecar7] arms done -> helping null list $(date -u +%FT%TZ)"
  python code/dispatch_sidecar.py results/jobs_null.json 7
fi
echo "==================== SIDECAR GPU7 DONE $(date -u +%FT%TZ) ===================="
