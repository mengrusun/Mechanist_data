import time
from huggingface_hub import hf_hub_download
t = time.time()
for fn in ["config.json", "vocab.txt", "tokenizer_config.json",
           "special_tokens_map.json", "pytorch_model.bin"]:
    p = hf_hub_download("facebook/esmfold_v1", fn, resume_download=True)
    print(f"got {fn} -> {p} ({time.time()-t:.0f}s)", flush=True)
print("DONE_DL", flush=True)
