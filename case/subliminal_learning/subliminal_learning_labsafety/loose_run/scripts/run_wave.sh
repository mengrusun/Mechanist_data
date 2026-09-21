#!/bin/bash
# run_wave.sh — launch N parallel jobs pinned one-per-GPU, wait for all to finish
# Usage: run_wave.sh <log_prefix> <cmd1> <cmd2> <cmd3> <cmd4>
# Each cmd is a full python invocation; this script pins CUDA_VISIBLE_DEVICES per slot.
set -u
LOG_PREFIX="$1"
shift
mkdir -p "$(dirname "$LOG_PREFIX")"

pids=()
for i in $(seq 0 $(($# - 1))); do
    cmd="${!((i + 1))}"
    log="${LOG_PREFIX}.gpu${i}.log"
    echo "[wave] launching on GPU $i: $cmd" >> "${LOG_PREFIX}.wave.log"
    CUDA_VISIBLE_DEVICES=$i nohup bash -c "$cmd" > "$log" 2>&1 &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do
    wait $pid || status=1
done
echo "[wave] all done (status=$status)" >> "${LOG_PREFIX}.wave.log"
exit $status
