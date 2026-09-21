#!/usr/bin/env bash
# Round-2 M3 random-null RE-DISPATCH (crash-proof) + auto-consolidation.
# Replaces the orchestrate2 null tail that died at the too-short wall. Fixes:
#   * RUN_TIMEOUT raised 9000 -> 12000s (a 3-dir dual-predictor chunk needs ~4600s; 2.6x headroom).
#   * chunk size 6 -> 3 dirs (jobs_null.json regenerated) : smaller blast radius, better 5-GPU fill.
#   * m3_random2.py now checkpoints per-dir + per-(predictor,dir) to <out>.ckpt.json and resumes,
#     so even a wall-kill loses at most one in-progress direction's current phase.
# GPUs: dispatch2 ALLOWED_GPUS = {3,4,5,6,7} (never 0/1/2). Auto-consolidates when all 16 chunks land.
set -uo pipefail
ROOT=/data/wanghaoxiong/intergene_mechanist_v6
LOG=$ROOT/results/null_redispatch.log
exec > >(tee -a "$LOG") 2>&1
echo "==================== NULL RE-DISPATCH START $(date -u +%FT%TZ) ===================="
source /data/wanghaoxiong/miniconda3/etc/profile.d/conda.sh
conda activate scientist
export PATH="/data/wanghaoxiong/miniconda3/envs/scientist/bin:$PATH"   # mkdssp / pydssp on PATH
export PYTHONPATH="$ROOT/code:$ROOT/third_party/OmegaFold"
# HF_HOME deliberately NOT overridden (ESMFold cached in default ~/.cache/huggingface, as M1/M2 used)
export RUN_TIMEOUT=12000
cd "$ROOT"
PY=/data/wanghaoxiong/miniconda3/envs/scientist/bin/python
CSTAR=21.4801

echo "[null] mkdssp resolves to: $(command -v mkdssp || echo MISSING)"
echo "[null] dispatch2 over jobs_null.json (chunk=3, 16 chunks, RUN_TIMEOUT=$RUN_TIMEOUT) ==="
$PY code/dispatch2.py results/jobs_null.json 5
echo "[null] dispatch2 returned rc=$? $(date -u +%FT%TZ)"

# verify EVERY chunk produced a non-empty json before consolidating
MISSING=0
while read -r rid; do
  f="results/${rid}.json"
  if [ ! -s "$f" ]; then echo "[null] MISSING chunk output: $f"; MISSING=$((MISSING+1)); fi
done < <($PY -c "import json;[print(j['run_id']) for j in json.load(open('results/jobs_null.json'))]")

if [ "$MISSING" -ne 0 ]; then
  echo "[null] $MISSING chunk(s) still missing -> NOT consolidating. Re-run this launcher to resume."
  echo "==================== NULL RE-DISPATCH INCOMPLETE $(date -u +%FT%TZ) ===================="
  exit 1
fi

echo "[null] all 16 chunks present -> consolidating C3 S-vs-null $(date -u +%FT%TZ) ==="
$PY code/m3_consolidate2.py --c_star "$CSTAR" --heldout 200,201 \
    --out results/m3_specificity_summary.json
echo "[null] consolidation rc=$? -> results/m3_specificity_summary.json"
echo "==================== NULL RE-DISPATCH DONE $(date -u +%FT%TZ) ===================="
