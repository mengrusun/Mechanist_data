#!/bin/bash
# Full-pipeline orchestrator (recovery version).
# Launches M1 workers 0-3 with --resume (skips already-complete chains), waits, then
# runs aggregate M1 -> parallel {M3a on GPU 0, M2 workers on GPUs 1-3} -> aggregate M2/M3a
# -> M3b on GPUs 0-3 -> aggregate M3b -> build_report.
#
# DESIGNED to be launched inside `screen -dmS <name> bash scripts/orchestrate_full.sh`
# so it survives the parent shell dying (root cause of the 06:50 M1 crash: SIGHUP when
# the launching bash exited).
#
# Emits periodic heartbeats every ~5 min to logs/orchestrator.log so external monitors
# can tell it is alive.
set -u
# NOTE: we deliberately do NOT `set -e` — we want the orchestrator to continue
# through per-milestone failures and reach build_report even if one aggregator errors.

ROOT=/data/zhenqian/Reproduction1/mechanica/science/esmfold_mechanism
PY=/data/zhenqian/miniconda3/envs/belief/bin/python
cd "$ROOT"

LOG="$ROOT/logs/orchestrator.log"

log() {
    echo "[orch $(date '+%F %T')] $*" | tee -a "$LOG"
}

heartbeat_bg() {
    # Emit a heartbeat every 5 min while the pipeline runs.
    while true; do
        sleep 300
        echo "[orch $(date '+%F %T')] heartbeat — pipeline still running (pid $$)" >> "$LOG"
    done
}

log "======================================================================"
log "orchestrate_full.sh START (pid $$)"
log "GPUs allowed: {0,1,2,3}"
log "======================================================================"

heartbeat_bg &
HB_PID=$!
# make sure the heartbeat dies with us
trap 'kill $HB_PID 2>/dev/null || true' EXIT

# ------------------------------------------------------------
# Phase M1 (resume) — 4 workers on GPUs 0-3
# ------------------------------------------------------------
mkdir -p logs/m1 logs/m2 logs/m3a logs/m3b results/M1 results/M2 results/M3a results/M3b

log "M1: launching 4 workers (resume mode) on GPUs 0-3"
M1_PIDS=()
for w in 0 1 2 3; do
    START=$((w * 50))
    STOP=$((START + 50))
    CUDA_VISIBLE_DEVICES=$w setsid $PY scripts/m1_worker.py \
        --worker-id $w --start $START --stop $STOP --seed 42 --resume \
        >> logs/m1/m1_worker_$w.log 2>&1 &
    M1_PIDS+=($!)
    log "  m1 worker $w slice [$START,$STOP) GPU $w pid ${M1_PIDS[-1]}"
done
log "M1: waiting for all 4 workers to finish"
for pid in "${M1_PIDS[@]}"; do
    wait $pid
    log "  m1 pid $pid exit code $?"
done
log "M1: all workers done"

# ------------------------------------------------------------
# Aggregate M1 (produces results/M1/summary_stats.json + early_window.json)
# ------------------------------------------------------------
log "M1: aggregating"
$PY scripts/aggregate.py --milestone M1 2>&1 | tee -a "$LOG"

# ------------------------------------------------------------
# Phase M3a (GPU 0) + M2 (GPUs 1-3) in parallel
# ------------------------------------------------------------
log "M3a: launching on GPU 0 in background"
CUDA_VISIBLE_DEVICES=0 setsid $PY scripts/m3a_probe.py \
    --sampled-blocks "0,2,4,6" --n-shuffles 1000 --seed 42 \
    >> logs/m3a/m3a.log 2>&1 &
M3A_PID=$!
log "  m3a pid $M3A_PID"

log "M2: launching 3 workers on GPUs 1,2,3"
M2_PIDS=()
for w in 0 1 2; do
    GPU=$((w + 1))
    START=$((w * 67))
    if [ $w -eq 2 ]; then STOP=200; else STOP=$((START + 67)); fi
    CUDA_VISIBLE_DEVICES=$GPU setsid $PY scripts/m2_worker.py \
        --worker-id $w --start $START --stop $STOP --seed 42 \
        >> logs/m2/m2_worker_$w.log 2>&1 &
    M2_PIDS+=($!)
    log "  m2 worker $w slice [$START,$STOP) GPU $GPU pid ${M2_PIDS[-1]}"
done

# Wait for M3a
wait $M3A_PID
log "M3a: pid $M3A_PID exit code $?"
$PY scripts/aggregate.py --milestone M3a 2>&1 | tee -a "$LOG"
log "M3a: aggregated"

# Wait for M2
for pid in "${M2_PIDS[@]}"; do
    wait $pid
    log "  m2 pid $pid exit code $?"
done
log "M2: all workers done"
$PY scripts/aggregate.py --milestone M2 2>&1 | tee -a "$LOG"

# ------------------------------------------------------------
# Phase M3b — 4 workers on GPUs 0-3
# ------------------------------------------------------------
log "M3b: launching 4 workers on GPUs 0-3"
M3B_PIDS=()
for w in 0 1 2 3; do
    START=$((w * 50))
    STOP=$((START + 50))
    CUDA_VISIBLE_DEVICES=$w setsid $PY scripts/m3b_worker.py \
        --worker-id $w --start $START --stop $STOP --seed 42 \
        >> logs/m3b/m3b_worker_$w.log 2>&1 &
    M3B_PIDS+=($!)
    log "  m3b worker $w slice [$START,$STOP) GPU $w pid ${M3B_PIDS[-1]}"
done
for pid in "${M3B_PIDS[@]}"; do
    wait $pid
    log "  m3b pid $pid exit code $?"
done
log "M3b: all workers done"
$PY scripts/aggregate.py --milestone M3b 2>&1 | tee -a "$LOG"

# ------------------------------------------------------------
# Final report
# ------------------------------------------------------------
log "build_report: refreshing EXPERIMENT_RESULTS.md + EXPERIMENT_TRACKER.md"
$PY scripts/build_report.py 2>&1 | tee -a "$LOG"

log "======================================================================"
log "orchestrate_full.sh COMPLETE"
log "======================================================================"
# Sentinel file to signal completion to any external monitor.
touch "$ROOT/logs/orchestrator.done"
