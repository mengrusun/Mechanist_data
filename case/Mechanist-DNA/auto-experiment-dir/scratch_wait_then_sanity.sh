#!/usr/bin/env bash
cd /data/wanghaoxiong/intergene_mechanist_v6
until grep -q ESMFOLD_DL_DONE results/esmfold_dl.log 2>/dev/null; do
  if ! pgrep -f scratch_dl_esmfold >/dev/null 2>&1; then
    # dl proc gone; check bin actually present before giving up
    if ls ~/.cache/huggingface/hub/models--facebook--esmfold_v1/snapshots/*/pytorch_model.bin >/dev/null 2>&1; then break; fi
    echo "ESMFOLD DL PROC GONE without DONE marker $(date -u +%FT%TZ)" >> results/setup_sanity.log
    exit 1
  fi
  sleep 15
done
echo "ESMFold ready, launching setup+sanity $(date -u +%FT%TZ)" >> results/setup_sanity.log
bash /data/wanghaoxiong/intergene_mechanist_v6/run_setup_and_sanity.sh
