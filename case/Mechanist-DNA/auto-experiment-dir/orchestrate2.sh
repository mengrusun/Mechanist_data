#!/usr/bin/env bash
# Round-2 staged orchestrator: M1 -> M2 -> c* -> M3 (+ random null). Fully detached, own process
# group, per-run timeout via RUN_TIMEOUT. GPUs pinned to {3,4,5,6} by dispatch2. mkdssp on PATH
# (round-1 bug fix). ESMFold cache = default ~/.cache/huggingface (HF_HOME NOT overridden).
set -uo pipefail
ROOT=/data/wanghaoxiong/intergene_mechanist_v6
LOG=$ROOT/results/orchestrate2.log
exec > >(tee -a "$LOG") 2>&1
echo "==================== ORCHESTRATE2 START $(date -u +%FT%TZ) ===================="
source /data/wanghaoxiong/miniconda3/etc/profile.d/conda.sh
conda activate scientist
export PATH="/data/wanghaoxiong/miniconda3/envs/scientist/bin:$PATH"     # mkdssp on PATH
export PYTHONPATH="$ROOT/code:$ROOT/third_party/OmegaFold"
export RUN_TIMEOUT=${RUN_TIMEOUT:-9000}   # 2.5h per run (dual-predictor fold of 300 seqs)
cd "$ROOT"
PY=$(which python)

pick_free_gpu() {
  # first GPU in {5,6,3,4} with mem.used < 2000 MiB (avoids GPUs the v8 session holds)
  $PY - <<'PYEOF'
import subprocess
for g in (5,6,3,4):
    try:
        u=int(subprocess.run(["nvidia-smi","-i",str(g),"--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout.strip())
        if u<2000: print(g); break
    except Exception: pass
else:
    print(5)
PYEOF
}

phase_M1() {
  if [ -s results/m1_calibration.json ]; then echo "[orch] M1 exists, skip"; return; fi
  G=$(pick_free_gpu)
  echo "[orch] === M1 calibration (GPU$G, free-picked) $(date -u +%FT%TZ) ==="
  CUDA_VISIBLE_DEVICES=$G timeout 14400 $PY code/m1_calibrate2.py \
     --feature_set results/m0_feature_set.json --predictors esmfold,omegafold \
     --n_samples 400 --n_ref 200 --out results/m1_calibration.json
  echo "[orch] M1 done rc=$? $(date -u +%FT%TZ)"
}

phase_M2() {
  if [ -s results/m2_dose_response_curve.json ]; then echo "[orch] M2 curve exists, skip"; return; fi
  echo "[orch] === M2 jobs $(date -u +%FT%TZ) ==="
  $PY code/make_round2_jobs.py m2 --n_per_dose 300 --out results/jobs_m2.json
  echo "[orch] === M2 dispatch (GPUs 3,4,5,6) ==="
  $PY code/dispatch2.py results/jobs_m2.json 4
  echo "[orch] === M2 consolidate ==="
  $PY code/m2_consolidate2.py --sel_seed 42 --heldout 200,201 --out results/m2_dose_response_curve.json
}

phase_M3() {
  if [ -s results/m3_specificity_summary.json ]; then echo "[orch] M3 summary exists, skip"; return; fi
  CSTAR=$($PY -c "import json;print(json.load(open('results/m2_dose_response_curve.json'))['c_star'])")
  echo "[orch] === M3 at c*=$CSTAR $(date -u +%FT%TZ) ==="
  if [ "$CSTAR" = "None" ] || [ -z "$CSTAR" ]; then echo "[orch] no c* -> abort M3"; return; fi
  $PY code/make_round2_jobs.py m3   --c_star "$CSTAR" --n_per_dose 300 --out results/jobs_m3.json
  $PY code/make_round2_jobs.py null --c_star "$CSTAR" --n_dirs 48 --chunk 6 --n_per_dir 120 --out results/jobs_null.json
  echo "[orch] === M3 arms dispatch ==="
  $PY code/dispatch2.py results/jobs_m3.json 4
  echo "[orch] === M3 random-null dispatch ==="
  $PY code/dispatch2.py results/jobs_null.json 4
  echo "[orch] === M3 consolidate ==="
  $PY code/m3_consolidate2.py --c_star "$CSTAR" --heldout 200,201 --out results/m3_specificity_summary.json
}

phase_M1
phase_M2
phase_M3
echo "==================== ORCHESTRATE2 DONE $(date -u +%FT%TZ) ===================="
