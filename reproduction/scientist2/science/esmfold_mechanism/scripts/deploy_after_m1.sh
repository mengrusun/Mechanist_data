#!/bin/bash
# Driver script: after M1 completes, aggregate, then run M3a + M2, then M3b, then final aggregate.
# Assumes M1 workers 0-3 are all done (data/results/M1/effect_by_window_worker_*.jsonl exist).
set -e

ROOT=/data/zhenqian/Reproduction1/mechanica/science/esmfold_mechanism
PY=/data/zhenqian/miniconda3/envs/belief/bin/python
mkdir -p $ROOT/logs/m2 $ROOT/logs/m3a $ROOT/logs/m3b

cd $ROOT

echo "=== Aggregate M1 (produce early_window.json for M2/M3b) ==="
$PY scripts/aggregate.py --milestone M1 2>&1 | tail -5

# ------ M3a (only needs heldout_probe chains, doesn't depend on M1) ------
# Launch M3a on GPU 0 in background
echo "=== Launch M3a on GPU 0 ==="
CUDA_VISIBLE_DEVICES=0 nohup $PY scripts/m3a_probe.py \
    --sampled-blocks "0,2,4,6" --n-shuffles 1000 --seed 42 \
    > logs/m3a/m3a.log 2>&1 &
M3A_PID=$!
echo "  m3a pid $M3A_PID"

# ------ M2 (uses M1's early_window) on GPUs 1,2,3 (three workers) ------
echo "=== Launch M2 workers on GPUs 1,2,3 ==="
# Split 200 main chains across 3 workers: 67, 67, 66
for w in 0 1 2; do
    GPU=$((w + 1))
    START=$((w * 67))
    if [ $w -eq 2 ]; then STOP=200; else STOP=$((START + 67)); fi
    CUDA_VISIBLE_DEVICES=$GPU nohup $PY scripts/m2_worker.py \
        --worker-id $w --start $START --stop $STOP --seed 42 \
        > logs/m2/m2_worker_$w.log 2>&1 &
    echo "  m2 worker $w slice [$START,$STOP) on GPU $GPU pid $!"
done

# Wait for M3a to finish (uses GPU 0 alone; also produces v_charge for M3b)
wait $M3A_PID
echo "=== M3a done ==="
$PY scripts/aggregate.py --milestone M3a 2>&1 | tail -5
# Free GPU 0

# Wait for M2 to finish
wait
echo "=== M2 done ==="
$PY scripts/aggregate.py --milestone M2 2>&1 | tail -5

# ------ M3b (uses v_charge from M3a + early block from M3a's best_block) on all 4 GPUs ------
echo "=== Launch M3b workers on GPUs 0-3 ==="
for w in 0 1 2 3; do
    START=$((w * 50))
    STOP=$((START + 50))
    CUDA_VISIBLE_DEVICES=$w nohup $PY scripts/m3b_worker.py \
        --worker-id $w --start $START --stop $STOP --seed 42 \
        > logs/m3b/m3b_worker_$w.log 2>&1 &
    echo "  m3b worker $w slice [$START,$STOP) on GPU $w pid $!"
done
wait
echo "=== M3b done ==="
$PY scripts/aggregate.py --milestone M3b 2>&1 | tail -5

echo "=== All milestones done. Building final EXPERIMENT_RESULTS.md ==="
$PY scripts/build_report.py 2>&1 | tail -5
echo "=== ALL DONE ==="
