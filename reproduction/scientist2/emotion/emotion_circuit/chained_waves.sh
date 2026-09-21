#!/bin/bash
# Chains: wait for wave1 (M1 PID) → run wave2 → run wave3 → mark DONE.
# Emits progress + failures to stdout.
REPO=/data/zhenqian/Reproduction1/mechanica/emotion/emotion_circuit
cd "$REPO"

# 1. Wait for wave1's M1 python to exit
M1_PID=2974740
echo "[$(date -Iseconds)] chained_waves: waiting for wave1 M1 pid=$M1_PID"
tail --pid=$M1_PID -f /dev/null
echo "[$(date -Iseconds)] chained_waves: wave1 M1 pid exited"

# Verify wave1 artifacts landed
if [ -s runs/A1_location/verdict.json ] || [ -s runs/A1_location/c_e.npz ] || grep -q "M1: DONE" logs/M1.log 2>/dev/null; then
  echo "[$(date -Iseconds)] chained_waves: WAVE1_OK"
else
  echo "[$(date -Iseconds)] chained_waves: WAVE1_MAYBE_INCOMPLETE — final artifacts not obvious; continuing anyway"
  tail -3 logs/M1.log 2>&1 | while IFS= read -r l; do echo "[wave1 M1.log tail] $l"; done
fi

# 2. Launch wave2, foreground
echo "[$(date -Iseconds)] chained_waves: launching wave2 (M2 on GPU 1 + M3 on GPU 6)"
bash deploy_wave2.sh > logs/wave2_launcher.log 2>&1
w2=$?
if [ $w2 -eq 0 ]; then
  echo "[$(date -Iseconds)] chained_waves: WAVE2_OK"
else
  echo "[$(date -Iseconds)] chained_waves: WAVE2_FAILED exit=$w2"
  tail -5 logs/M2.log 2>&1 | while IFS= read -r l; do echo "[wave2 M2.log tail] $l"; done
  tail -5 logs/M3.log 2>&1 | while IFS= read -r l; do echo "[wave2 M3.log tail] $l"; done
  echo "CHAINED_DONE"
  exit $w2
fi

# 3. Launch wave3, foreground
echo "[$(date -Iseconds)] chained_waves: launching wave3 (M4 on GPU 1)"
bash deploy_wave3.sh > logs/wave3_launcher.log 2>&1
w3=$?
if [ $w3 -eq 0 ]; then
  echo "[$(date -Iseconds)] chained_waves: WAVE3_OK"
else
  echo "[$(date -Iseconds)] chained_waves: WAVE3_FAILED exit=$w3"
  tail -5 logs/M4.log 2>&1 | while IFS= read -r l; do echo "[wave3 M4.log tail] $l"; done
fi

echo "[$(date -Iseconds)] chained_waves: CHAINED_DONE"
