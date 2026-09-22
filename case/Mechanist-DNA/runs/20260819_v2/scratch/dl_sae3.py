import os, shutil, time
os.environ.pop("HF_ENDPOINT", None)
from huggingface_hub import hf_hub_download
tok=os.environ["HF_TOKEN"]
t=time.time()
p=hf_hub_download("Goodfire/Evo-2-Layer-26-Mixed","sae-layer26-mixed-expansion_8-k_64.pt", token=tok)
print("dl to", p, round(time.time()-t,1),"s", os.path.getsize(p), flush=True)
shutil.copy(p, os.path.join(os.path.dirname(__file__),"..","assets","evo2_l26_sae.pt"))
import torch; sd=torch.load(os.path.join(os.path.dirname(__file__),"..","assets","evo2_l26_sae.pt"),map_location="cpu",weights_only=False)
print("LOAD OK nkeys", len(sd), [ (k,tuple(v.shape)) for k,v in list(sd.items())[:6]], flush=True)
open(os.path.join(os.path.dirname(__file__),"..","assets",".sae_dl_done"),"w").write("done")
print("SAE_READY", flush=True)
