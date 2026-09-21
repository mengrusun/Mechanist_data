#!/usr/bin/env bash
# Robust ESMFold weight fetch: HF xet backend corrupted the download (deleted the .incomplete
# without producing the final blob). Disable xet -> standard resumable HTTPS via hf_hub_download.
set -uo pipefail
LOG=/data/wanghaoxiong/intergene_mechanist_v6/results/esmfold_dl.log
exec > >(tee -a "$LOG") 2>&1
echo "=== esmfold weight dl (xet disabled) start $(date -u +%FT%TZ) ==="
source /data/wanghaoxiong/miniconda3/etc/profile.d/conda.sh
conda activate scientist
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
python - <<'PY'
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import hf_hub_download, list_repo_files
files = list_repo_files("facebook/esmfold_v1")
print("repo files:", files, flush=True)
wf = [f for f in files if f.endswith((".bin", ".safetensors")) and "model" in f]
print("weight files:", wf, flush=True)
for f in wf:
    p = hf_hub_download("facebook/esmfold_v1", f)
    print("downloaded", f, "->", p, os.path.getsize(p), "bytes", flush=True)
print("ESMFOLD_DL_DONE", flush=True)
PY
echo "=== esmfold weight dl done $(date -u +%FT%TZ) rc=$? ==="
