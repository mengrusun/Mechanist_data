#!/usr/bin/env bash
# Variant run script: model-swap-olmo-1b (C2 Belief Heads Localization)
# One-line reproduce command:
#   bash verify/C2_belief_heads_localization/variants/model-swap-olmo-1b/run.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"

echo "[run.sh] Experiment root: $EXP_ROOT"
echo "[run.sh] Script dir: $SCRIPT_DIR"
echo "[run.sh] Starting model-swap-olmo-1b variant for C2 at $(date)"

cd "$EXP_ROOT"

# Ensure Python can find project scripts
export PYTHONPATH="$EXP_ROOT/scripts:${PYTHONPATH:-}"

python "$SCRIPT_DIR/run_variant.py" 2>&1 | tee "$SCRIPT_DIR/run.log"

echo "[run.sh] Variant completed at $(date)"
echo "[run.sh] Results: $SCRIPT_DIR/result.json"
