cd /data/wanghaoxiong/Mechanist-DNA-experiment/20260823_v1
export HF_HOME=/data/wanghaoxiong/Mechanist-DNA-experiment/20260823_v1/models_cache
export HF_ENDPOINT=https://huggingface.co
python -u logs/dl_one.py arcinstitute/evo2_7b
python -u logs/dl_one.py facebook/esmfold_v1
python -u logs/dl_one.py Goodfire/Evo-2-Layer-26-Mixed
echo "== ALL DOWNLOADS COMPLETE =="
