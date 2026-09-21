import os, time
os.environ["HF_HOME"] = "/data/wanghaoxiong/Mechanist-DNA-experiment/20260823_v1/models_cache"
os.environ["HF_TOKEN"] = "<REDACTED_HF_TOKEN>"
os.environ.pop("HF_ENDPOINT", None)  # HF direct (mirror cannot locate these repos)
from huggingface_hub import snapshot_download
tok = os.environ["HF_TOKEN"]
repos = ["Goodfire/Evo-2-Layer-26-Mixed", "facebook/esmfold_v1", "arcinstitute/evo2_7b"]
for repo in repos:
    print(f"== downloading {repo} @ {time.strftime('%H:%M:%S')} ==", flush=True)
    ok = False
    for attempt in range(1, 500):
        try:
            snapshot_download(repo, token=tok, max_workers=8, etag_timeout=60)
            print(f"== done {repo} @ {time.strftime('%H:%M:%S')} ==", flush=True); ok = True; break
        except Exception as e:
            print(f"[retry {attempt}] {repo}: {repr(e)[:90]}", flush=True); time.sleep(5)
    if not ok:
        print(f"== GAVE UP {repo} ==", flush=True)
print("== ALL DOWNLOADS COMPLETE ==", flush=True)
