import os, time
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import hf_hub_download
t=time.time()
p = hf_hub_download("facebook/esmfold_v1", "pytorch_model.bin", force_download=True)
sz = os.path.getsize(p)
print(f"DONE_DL {p} size={sz} ({sz/1e9:.2f}GB) in {time.time()-t:.0f}s", flush=True)
