import os, torch, numpy as np
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
from evo2 import Evo2
m = Evo2('evo2_7b')
print("LOADED", flush=True)
# ---- embedding extraction at blocks.26 output ----
def toks(s): return torch.tensor(m.tokenizer.tokenize(s), dtype=torch.long).unsqueeze(0).cuda()
seq="ATGGCACGTGATCGATCGGCACTGACGTTTAGCACAGTGACAATGCGATCG"
logits, emb = m.forward(toks(seq), return_embeddings=True, layer_names=["blocks.26"])
h = emb["blocks.26"]
print("EMB blocks.26 shape", tuple(h.shape), "dtype", h.dtype, flush=True)
d_model = h.shape[-1]
# build a fake steering vector (unit norm) in d_model
v = torch.randn(d_model, dtype=h.dtype, device=h.device); v = v/ v.norm()
# ---- steering hook: add alpha*|h|*v to residual output during generation ----
state={"alpha":0.0}
def mk_hook(vv, st):
    def hook(mod, inp, out):
        if isinstance(out, tuple):
            hs=out[0]; hs = hs + st["alpha"]*vv.to(hs.dtype); return (hs,)+tuple(out[1:])
        else:
            return out + st["alpha"]*vv.to(out.dtype)
    return hook
blk = m.model.get_submodule("blocks.26")
hd = blk.register_forward_hook(mk_hook(v, state))
prompt=["ATGGCACGT"]
state["alpha"]=0.0
o0 = m.generate(prompt_seqs=prompt, n_tokens=90, temperature=0.7, top_k=4, top_p=1.0, verbose=0)
state["alpha"]=8.0
o1 = m.generate(prompt_seqs=prompt, n_tokens=90, temperature=0.7, top_k=4, top_p=1.0, verbose=0)
hd.remove()
print("BASE   :", o0.sequences[0][:70], "logprob", round(float(np.mean(o0.logprobs_mean)),3), flush=True)
print("STEERED:", o1.sequences[0][:70], "logprob", round(float(np.mean(o1.logprobs_mean)),3), flush=True)
print("DIFFER :", o0.sequences[0]!=o1.sequences[0], flush=True)
# validity: are sequences pure ACGT?
import re
print("base ACGT-only:", bool(re.fullmatch('[ACGT]+', o0.sequences[0])), "steer ACGT-only:", bool(re.fullmatch('[ACGT]+', o1.sequences[0])), flush=True)
print("SMOKE2_DONE", flush=True)
