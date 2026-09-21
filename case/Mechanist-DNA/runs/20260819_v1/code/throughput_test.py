"""Measure batched generation throughput (with & without steering) to plan M2/M3."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import torch
import common as C

m = C.load_evo2(verbose=False)
dev = C.infer_layer_device(m)
sae = C.TiedTopKSAE(device=str(dev), relu_before_topk=True)
# arbitrary steer vec
vec = sae.steer_vector(torch.arange(5, device=sae.device), torch.ones(5, device=sae.device)).to(dev)

for bs in (16, 32):
    prompts = [C.START_CONTEXT] * bs
    t = time.time()
    out = m.generate(prompts, n_tokens=300, temperature=1.0, top_k=4, top_p=1.0,
                     cached_generation=True, verbose=0)
    dt = time.time() - t
    print(f"[tput] bs={bs} n_tokens=300 batched: {dt:.1f}s -> {bs*300/dt:.0f} tok/s, "
          f"{dt/bs:.2f}s/seq", flush=True)

# with steering
bs = 32
prompts = [C.START_CONTEXT] * bs
with C.ResidualSteerer(m, vec, alpha=4.0):
    t = time.time()
    out = m.generate(prompts, n_tokens=900, temperature=1.0, top_k=4, top_p=1.0,
                     cached_generation=True, verbose=0)
    dt = time.time() - t
print(f"[tput] bs={bs} n_tokens=900 STEERED: {dt:.1f}s -> {dt/bs:.2f}s/seq", flush=True)
print(f"[tput] sample: {out.sequences[0][:80]}", flush=True)
import numpy as np
gpu_mem = torch.cuda.max_memory_allocated()/1e9
print(f"[tput] peak GPU mem {gpu_mem:.1f} GB", flush=True)
