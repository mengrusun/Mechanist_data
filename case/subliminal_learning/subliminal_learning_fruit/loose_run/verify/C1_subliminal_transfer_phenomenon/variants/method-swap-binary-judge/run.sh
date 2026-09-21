#!/bin/bash
# Variant: method-swap-binary-judge
# Claim: C1 — subliminal transfer phenomenon
# GPU cost: 0 GPU-h (API only — no GPU used)
# CUDA_VISIBLE_DEVICES not set (GPU-free run)
set -euo pipefail

VARIANT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$VARIANT_DIR/../../../.." && pwd)"  # up to multi_modal_B_loose1

cd "$PROJECT_ROOT"

echo "[run.sh] Starting method-swap-binary-judge variant for C1"
echo "[run.sh] Project root: $PROJECT_ROOT"
echo "[run.sh] Variant dir: $VARIANT_DIR"
echo "[run.sh] GPU cost: 0 (API only)"
echo ""

python "$VARIANT_DIR/run.py" 2>&1 | tee "$VARIANT_DIR/run.log"
echo ""
echo "[run.sh] Done. result.json and cost.json written to $VARIANT_DIR"
