"""Smoke test: batched evo2.generate throughput + sample diversity.
Beam search needs W candidate chunks per beam per step -> batching is the cost lever."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import mc_env; mc_env.patch()
import torch
from evo2_sae import load_evo2
from m1_harness_calibrate import natural_prompts

evo2 = load_evo2("cuda:0")
prompts, _, _ = natural_prompts(mc_env.ORGANISM, 8)
p = prompts[0]
print(f"[smoke] n_prompts_avail={len(prompts)} prompt_len={len(p)}", flush=True)

def gen(seqs, n_tokens):
    out = evo2.generate(prompt_seqs=seqs, n_tokens=n_tokens, temperature=0.7, top_k=4, verbose=0)
    return out.sequences if hasattr(out, "sequences") else out

gen([p], 60)  # warmup
for B in (1, 4, 8, 16, 32):
    for ntok in (60, 300):
        t0 = time.time()
        seqs = gen([p] * B, ntok)
        dt = time.time() - t0
        print(f"[smoke] batch={B:>2} n_tokens={ntok:>3}  total={dt:6.2f}s  per_seq={dt/B:6.3f}s  "
              f"unique={len(set(seqs))}/{B}", flush=True)
