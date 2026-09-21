#!/bin/bash
source /data/wanghaoxiong/miniconda3/etc/profile.d/conda.sh
conda activate scientist
cd /data/wanghaoxiong/intergene_mechanist_v6
echo "=== waiting for prokaryote clustering ==="
until [ -f data/m0_clusters_prokaryote.json ]; do sleep 15; done
echo "build done: $(wc -l < data/m0_dataset_prokaryote.jsonl) proteins"
cat data/m0_build_summary_prokaryote.json 2>/dev/null; echo
echo "=== caching prokaryote activations on GPU 3 (GPU 2 held by other user) ==="
CUDA_VISIBLE_DEVICES=3 python code/m0_cache_acts.py --organism prokaryote 2>&1 | grep -E "cache|CACHE|codons|proteins|Error|site|Traceback" | grep -vE "INFO|it/s"
mkdir -p runs/M0_cache_prokaryote
echo '{"run_id":"M0_cache_prokaryote","gpu_ids":[3],"cuda_visible_devices":"3","note":"GPU2 occupied by other user"}' > runs/M0_cache_prokaryote/cost.json
echo "=== running prokaryote M0 grid (6 configs) ==="
python code/run_m0_grid.py --organisms prokaryote 2>&1 | grep -E "grid|verdict|FAIL"
echo "PROK_M0_DONE"
