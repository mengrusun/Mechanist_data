#!/bin/bash
# Auto-launcher: waits for M0 verdict; if established/conditional launches M1/M2.
set -uo pipefail
cd /data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1
source /data/zhenqian/miniconda3/etc/profile.d/conda.sh
conda activate subliminal_mm
export PYTHONPATH=/data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1/scripts

echo "[auto-mechanism] waiting for M0 verdict..."
while [ ! -f runs/m0_verdict/verdict.json ]; do
    sleep 60
done
VERDICT=$(/data/zhenqian/miniconda3/envs/subliminal_mm/bin/python -c "import json; print(json.load(open('runs/m0_verdict/verdict.json'))['verdict'])")
echo "[auto-mechanism] M0 verdict = ${VERDICT}"

if [ "$VERDICT" != "established" ] && [ "$VERDICT" != "conditional" ]; then
    echo "[auto-mechanism] verdict $VERDICT — skipping M1/M2"
    exit 0
fi

echo "[auto-mechanism] launching M1/M2 pipeline..."
# Commit the mechanism routing (flip committed: false → true)
sed -i 's/^committed: false/committed: true/' refine-logs/MECHANISM_ROUTING.md
sed -i 's/^chosen_family: none/chosen_family: representation-and-parameter-analysis\/steering-vectors/' refine-logs/MECHANISM_ROUTING.md
echo "[auto-mechanism] committed routing family"

bash run_m1_m2.sh > logs/m1_m2_orchestrator.log 2>&1
echo "[auto-mechanism] M1/M2 done. Verdict:"
cat runs/m2_verdict.json 2>/dev/null || echo "no verdict"
echo "==== auto-mechanism DONE ===="
