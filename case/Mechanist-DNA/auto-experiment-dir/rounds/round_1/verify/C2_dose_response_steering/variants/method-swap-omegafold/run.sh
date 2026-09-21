#!/bin/bash
# Verify variant (C2, dimension=method): ESMFold -> OmegaFold structure-predictor swap.
# CUDA_VISIBLE_DEVICES pinned per HC2/verify constraints -- pass one of {2,3,4,5} (GPU2 often busy).
set -e
VDIR=/data/wanghaoxiong/intergene_mechanist_v6/verify/C2_dose_response_steering/variants/method-swap-omegafold
DATA=$VDIR/data
mkdir -p "$DATA"
GPU="${1:-3}"
ALPHA="$2"
SEED="$3"
N="${4:-150}"
CONF_MIN="${5:?must pass a calibrated --conf_min (see calibrate_conf_gate.py)}"

SEQ_JSONL="$DATA/seqs_a${ALPHA}_s${SEED}.jsonl"
OMEGA_OUT="$DATA/omegafold_a${ALPHA}_s${SEED}.json"
ESMFOLD_SAMECELLS_OUT="$DATA/esmfold_samecells_a${ALPHA}_s${SEED}.json"
DIAG_OUT="$DATA/diag_a${ALPHA}_s${SEED}.json"

echo "[run] Stage 1/3: generate + save sequences (scientist env, GPU $GPU)"
CUDA_VISIBLE_DEVICES=$GPU conda run -n scientist python "$VDIR/gen_sequences.py" \
    --alpha "$ALPHA" --seed "$SEED" --n "$N" --out "$SEQ_JSONL"

echo "[run] Stage 2/3: fold with OmegaFold (verify_omegafold env, GPU $GPU)"
CUDA_VISIBLE_DEVICES=$GPU conda run -n verify_omegafold python "$VDIR/fold_and_readout_omegafold.py" \
    --in "$SEQ_JSONL" --out "$OMEGA_OUT" --conf_min "$CONF_MIN"

echo "[run] Stage 2b/3: fold the SAME sequences with ESMFold (scientist env, GPU $GPU) -- same-sequence cross-check"
CUDA_VISIBLE_DEVICES=$GPU conda run -n scientist python "$VDIR/fold_and_readout_esmfold_samecells.py" \
    --in "$SEQ_JSONL" --out "$ESMFOLD_SAMECELLS_OUT" --plddt_min 50

echo "[run] Stage 3/3: sequence-composition diagnostics (CPU, any env)"
conda run -n scientist python "$VDIR/seq_diagnostics.py" --in "$SEQ_JSONL" --out "$DIAG_OUT"

echo "[run] done: alpha=$ALPHA seed=$SEED -> $OMEGA_OUT, $ESMFOLD_SAMECELLS_OUT, $DIAG_OUT"
