#!/bin/bash
# Orchestrator: wait for M1 workers to complete, then run M2 + M3a + M3b + aggregate + report.
# Kickstart via: nohup bash scripts/orchestrate_after_m1.sh > logs/orchestrator.log 2>&1 &
set -e

ROOT=/data/zhenqian/Reproduction1/mechanica/science/esmfold_mechanism
PY=/data/zhenqian/miniconda3/envs/belief/bin/python
cd $ROOT

echo "[orch] $(date +%T) waiting for M1 workers to finish..."
while true; do
    N=$(ls results/M1/m1_worker_*_summary.json 2>/dev/null | wc -l)
    if [ "$N" -ge 4 ]; then
        echo "[orch] $(date +%T) all 4 M1 workers done"
        break
    fi
    sleep 60
done

echo "[orch] $(date +%T) Aggregate M1"
$PY scripts/aggregate.py --milestone M1 2>&1 | tee -a logs/orchestrator.log

# ------ Launch M3a on GPU 0 in background (independent) ------
mkdir -p logs/m3a logs/m2 logs/m3b
echo "[orch] $(date +%T) Launch M3a on GPU 0 in background"
CUDA_VISIBLE_DEVICES=0 nohup $PY scripts/m3a_probe.py \
    --sampled-blocks "0,2,4,6" --n-shuffles 1000 --seed 42 \
    > logs/m3a/m3a.log 2>&1 &
M3A_PID=$!
echo "[orch] m3a pid $M3A_PID"

# ------ Launch M2 workers on GPUs 1,2,3 ------
echo "[orch] $(date +%T) Launch M2 workers on GPUs 1,2,3"
M2_PIDS=()
for w in 0 1 2; do
    GPU=$((w + 1))
    START=$((w * 67))
    if [ $w -eq 2 ]; then STOP=200; else STOP=$((START + 67)); fi
    CUDA_VISIBLE_DEVICES=$GPU nohup $PY scripts/m2_worker.py \
        --worker-id $w --start $START --stop $STOP --seed 42 \
        > logs/m2/m2_worker_$w.log 2>&1 &
    M2_PIDS+=($!)
    echo "[orch] m2 worker $w slice [$START,$STOP) GPU $GPU pid ${M2_PIDS[-1]}"
done

# Wait for M3a to finish
wait $M3A_PID || echo "[orch] m3a exit code $?"
echo "[orch] $(date +%T) M3a done"
$PY scripts/aggregate.py --milestone M3a 2>&1 | tee -a logs/orchestrator.log

# Wait for M2 to finish
for pid in "${M2_PIDS[@]}"; do wait $pid || echo "[orch] m2 pid $pid exit code $?"; done
echo "[orch] $(date +%T) M2 done"
$PY scripts/aggregate.py --milestone M2 2>&1 | tee -a logs/orchestrator.log

# ------ Launch M3b on all 4 GPUs ------
echo "[orch] $(date +%T) Launch M3b on GPUs 0-3"
M3B_PIDS=()
for w in 0 1 2 3; do
    START=$((w * 50))
    STOP=$((START + 50))
    CUDA_VISIBLE_DEVICES=$w nohup $PY scripts/m3b_worker.py \
        --worker-id $w --start $START --stop $STOP --seed 42 \
        > logs/m3b/m3b_worker_$w.log 2>&1 &
    M3B_PIDS+=($!)
    echo "[orch] m3b worker $w slice [$START,$STOP) GPU $w pid ${M3B_PIDS[-1]}"
done
for pid in "${M3B_PIDS[@]}"; do wait $pid || echo "[orch] m3b pid $pid exit code $?"; done
echo "[orch] $(date +%T) M3b done"
$PY scripts/aggregate.py --milestone M3b 2>&1 | tee -a logs/orchestrator.log

echo "[orch] $(date +%T) Build final report"
$PY scripts/build_report.py 2>&1 | tee -a logs/orchestrator.log

echo "[orch] $(date +%T) ALL DONE"
