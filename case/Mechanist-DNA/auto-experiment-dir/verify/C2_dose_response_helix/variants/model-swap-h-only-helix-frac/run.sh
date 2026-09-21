#!/bin/bash
# Variant: model-swap-h-only-helix-frac
# Claim: C2 (dose-response helix)
# Swap: H-only pLDDT-weighted helix fraction (helix_h_w) instead of HGI (helix_hgi_w)
# GPU: none (analysis-only, reads pre-computed M2 results)

set -euo pipefail
PROJ=/data/wanghaoxiong/intergene_mechanist_v6
OUT_DIR=$PROJ/verify/C2_dose_response_helix/variants/model-swap-h-only-helix-frac

python "$PROJ/verify/C2_dose_response_helix/variants/model-swap-h-only-helix-frac/analyze_c2_h_only.py" \
    --results_dir "$PROJ/results" \
    --out_dir "$OUT_DIR"
