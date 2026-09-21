"""Sanity: validate model load, blocks.26 hook point (via SAE reconstruction),
SAE encode convention, steering hook, and a tiny generation. Tiny scale (exempt
from strict-fidelity per harness rule 5)."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
import torch, numpy as np
import common as C

t0 = time.time()
dev = "cuda:0"
m = C.load_evo2()
print(f"[smoke] model loaded in {time.time()-t0:.0f}s", flush=True)
ldev = C.infer_layer_device(m)
print(f"[smoke] blocks.26 device = {ldev}", flush=True)

# A real E. coli CDS snippet (thrA start) — coding DNA
dna = ("ATGCGAGTGTTGAAGTTCGGCGGTACATCAGTGGCAAATGCAGAACGTTTTCTGCGTGTTGCCGATATTCTGGAA"
       "AGCAATGCCAGGCAGGGGCAGGTGGCCACCGTCCTCTCTGCCCCCGCCAAAATCACCAACCACCTGGTGGCGATG")
tok = m.tokenizer
ids = torch.tensor(tok.tokenize(dna), dtype=torch.long, device=dev).unsqueeze(0)
print(f"[smoke] seq len {len(dna)} tokens {ids.shape}", flush=True)

# Capture blocks.26 activation
act = C.capture_layer26(m, ids)  # (1,L,4096)
print(f"[smoke] blocks.26 act shape {tuple(act.shape)} dtype {act.dtype} "
      f"mean {act.mean():.3f} std {act.std():.3f}", flush=True)
assert act.shape[-1] == C.D_MODEL

x = act[0].to(dev, torch.float32)   # (L,4096)

# Test both encode conventions -> reconstruction error (validates hook point + convention)
for relu_first in (True, False):
    sae = C.TiedTopKSAE(device=dev, relu_before_topk=relu_first)
    z = sae.encode(x)
    xhat = sae.decode(z)
    resid = (x - xhat)
    fvu = (resid.pow(2).sum() / ((x - x.mean(0)).pow(2).sum() + 1e-8)).item()
    nnz = (z > 0).float().sum(-1).mean().item()
    print(f"[smoke] relu_before_topk={relu_first}: FVU={fvu:.4f} "
          f"mean_nonzero={nnz:.1f} (target k={sae.k})", flush=True)

# Use the standard convention going forward
sae = C.TiedTopKSAE(device=dev, relu_before_topk=True)
z = sae.encode(x)
# pick a strongly-active latent, build a steer vector, verify hook changes output
lat_mean = z.mean(0)                          # (32768,)
top_lat = torch.topk(lat_mean, 5).indices
s_vals = torch.tensor([lat_mean[i].item() for i in top_lat], device=dev)
vec = sae.steer_vector(top_lat, s_vals).to(ldev)
print(f"[smoke] steer vec norm {vec.norm():.3f}; top latents {top_lat.tolist()}", flush=True)

# alpha=0 must be identity
with C.ResidualSteerer(m, vec, alpha=0.0):
    a0 = C.capture_layer26(m, ids)
print(f"[smoke] alpha=0 identity max|Δ| = {(a0-act).abs().max():.2e} (expect 0)", flush=True)

# Tiny generation baseline vs steered
t1 = time.time()
out0 = m.generate([C.START_CONTEXT], n_tokens=60, temperature=1.0, top_k=4, top_p=1.0,
                  cached_generation=True, verbose=0)
print(f"[smoke] gen baseline ({time.time()-t1:.1f}s): {out0.sequences[0][:60]}", flush=True)

with C.ResidualSteerer(m, vec, alpha=4.0):
    out1 = m.generate([C.START_CONTEXT], n_tokens=60, temperature=1.0, top_k=4, top_p=1.0,
                      cached_generation=True, verbose=0)
print(f"[smoke] gen steered a=4 : {out1.sequences[0][:60]}", flush=True)

# ORF + translate sanity
prot, olen, frame, st = C.longest_orf_protein("ATG"+out0.sequences[0])
print(f"[smoke] baseline ORF len_nt={olen} prot[:30]={prot[:30]}", flush=True)
print(f"[smoke] ALL OK in {time.time()-t0:.0f}s", flush=True)
