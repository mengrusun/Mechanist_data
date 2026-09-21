import os, sys, time
os.environ["HF_HOME"]="/data/wanghaoxiong/Mechanist-DNA-experiment/20260823_v1/models_cache"
from huggingface_hub import snapshot_download
repo=sys.argv[1]
for a in range(1,800):
    try:
        snapshot_download(repo, token="<REDACTED_HF_TOKEN>", max_workers=8, etag_timeout=60)
        print(f"DONE {repo} @ {time.strftime('%H:%M:%S')}",flush=True); break
    except Exception as e:
        print(f"[retry {a}] {repo} ep={os.environ.get('HF_ENDPOINT','direct')}: {repr(e)[:70]}",flush=True); time.sleep(4)
