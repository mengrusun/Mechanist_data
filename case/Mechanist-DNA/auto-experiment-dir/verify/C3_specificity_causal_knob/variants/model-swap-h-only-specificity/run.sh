#!/bin/bash
# Variant: model-swap-h-only-specificity
# Claim: C3 (specificity causal knob)
# Swap: H-only pLDDT-weighted helix fraction as specificity endpoint
# GPU: none (analysis-only)

set -euo pipefail
PROJ=/data/wanghaoxiong/intergene_mechanist_v6
OUT_DIR=$PROJ/verify/C3_specificity_causal_knob/variants/model-swap-h-only-specificity

python "$PROJ/verify/C3_specificity_causal_knob/variants/model-swap-h-only-specificity/analyze_c3_h_only.py" \
    --results_dir "$PROJ/results" \
    --out_dir "$OUT_DIR"
