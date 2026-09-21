#!/usr/bin/env bash
# M(-1): install OmegaFold (independent single-sequence second predictor) into the `scientist`
# env (already has torch 2.7.1 + biopython). Round-1 hiccup was OmegaFold's torch-pinned installer;
# we bypass it with --no-deps and pre-download the release weights directly from S3.
set -uo pipefail
LOG=/data/wanghaoxiong/intergene_mechanist_v6/results/omegafold_install.log
exec > >(tee -a "$LOG") 2>&1
echo "=== omegafold install start $(date -u +%FT%TZ) ==="
source /data/wanghaoxiong/miniconda3/etc/profile.d/conda.sh
conda activate scientist
python -c "import torch;print('torch', torch.__version__)"

echo "--- pip install einops + omegafold (--no-deps to protect torch/numpy ABI) ---"
pip install --no-input einops 2>&1 | tail -3
pip install --no-input --no-deps "git+https://github.com/HeliXonProtein/OmegaFold.git" 2>&1 | tail -8

echo "--- import check ---"
python -c "import omegafold; print('omegafold import OK', omegafold.__file__)" || echo "IMPORT_FAIL"

echo "--- pre-download release2 weights (model 2, the OmegaFold default) ---"
CKPT_DIR="$HOME/.cache/omegafold_ckpt"
mkdir -p "$CKPT_DIR"
if [ ! -s "$CKPT_DIR/model.pt" ]; then
  # helixon S3 release2 = the weights `omegafold` fetches for --model 2
  curl -L --fail --retry 5 --retry-delay 10 -o "$CKPT_DIR/model.pt" \
    https://helixon.s3.amazonaws.com/release2.pt 2>&1 | tail -2 \
    || echo "WEIGHT_DL_FAIL_release2"
fi
ls -la "$CKPT_DIR" 2>&1
echo "weights bytes: $(stat -c%s "$CKPT_DIR/model.pt" 2>/dev/null || echo 0)"
echo "=== omegafold install done $(date -u +%FT%TZ) ==="
