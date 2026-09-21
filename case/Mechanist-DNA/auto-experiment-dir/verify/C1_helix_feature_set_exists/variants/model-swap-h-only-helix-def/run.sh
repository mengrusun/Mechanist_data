#!/bin/bash
# Variant: model-swap-h-only-helix-def
# Claim: C1 (helix feature set exists)
# Swap: H-only helix definition (DSSP 'H' only) vs main experiment's HGI (H+G+I)
# Method: read existing M0 H_only result files, aggregate set-AUROC statistics
# GPU: none (analysis-only, reads pre-computed M0 results)

set -euo pipefail
PROJ=/data/wanghaoxiong/intergene_mechanist_v6
OUT_DIR=$PROJ/verify/C1_helix_feature_set_exists/variants/model-swap-h-only-helix-def

python "$PROJ/verify/C1_helix_feature_set_exists/variants/model-swap-h-only-helix-def/analyze_c1_h_only.py" \
    --results_dir "$PROJ/results" \
    --out_dir "$OUT_DIR"
