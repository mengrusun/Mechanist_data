import os, time, json, sys
os.environ.setdefault("HF_ENDPOINT","https://hf-mirror.com")
import torch
t0=time.time()
from evo2 import Evo2
print("import ok", round(time.time()-t0,1), flush=True)
m = Evo2('evo2_7b')
print("MODEL LOADED", round(time.time()-t0,1), "s", flush=True)
# dump module names (block-level) to find Layer-26 residual site
names=[n for n,_ in m.model.named_modules()]
blocks=[n for n in names if n.count('.')<=2 and ('block' in n.lower() or n=='' )]
with open('/data/wanghaoxiong/Mechanist-DNA-experiment/simple_20260819_v4/scratch/module_names.txt','w') as f:
    f.write("\n".join(names))
# top-level blocks list
print("N modules", len(names), flush=True)
print("sample block names:", [n for n in names if n.startswith('blocks.') and n.count('.')==1][:5], "...", [n for n in names if n.startswith('blocks.') and n.count('.')==1][-3:], flush=True)
# generate short DNA
prompt=["ATGGCACGT"]
seqs, scores = m.generate(prompt_seqs=prompt, n_tokens=120, temperature=1.0, top_k=4, top_p=1.0, verbose=1)
print("GEN OK:", seqs[0][:80], flush=True)
# forward + embedding at block 26 output
tok = torch.tensor(m.tokenizer.tokenize("ATGGCACGTGATCGATCGGCACTGACGT"), dtype=torch.long).unsqueeze(0).cuda()
layer="blocks.26.mlp.l3" if "blocks.26.mlp.l3" in names else "blocks.26"
try:
    logits, emb = m.forward(tok, return_embeddings=True, layer_names=[layer])
    print("EMB OK", layer, {k:tuple(v.shape) for k,v in emb.items()}, flush=True)
except Exception as e:
    print("EMB FAIL", layer, repr(e), flush=True)
print("ALL_SMOKE_DONE", flush=True)
