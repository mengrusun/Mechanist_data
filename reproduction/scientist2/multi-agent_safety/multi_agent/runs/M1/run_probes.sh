#!/usr/bin/env bash
set -euo pipefail
PYBIN=/data/zhenqian/miniconda3/envs/belief/bin/python
ROOT=/data/zhenqian/Reproduction1/mechanica/multi-agent_safety/multi_agent
cd "$ROOT"

echo "[probes] launching probe training for layers 27,37,48,59 (in parallel; CPU-heavy)"

for L in 27 37 48 59; do
    $PYBIN scripts/train_probe.py \
        --activations runs/M1/activations.pt \
        --layer $L \
        --floor-per-class 25 \
        --out runs/M1/probe_layer${L}.json > logs/M1_probe_L${L}.log 2>&1 &
done
wait
echo "[probes] all done"
ls -la runs/M1/probe_layer*.json
