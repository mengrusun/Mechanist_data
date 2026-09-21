"""Risk check: round 2 found that alternating Evo2 (vortex/flash-attn) with ESMFold in one
process corrupts kernel dispatch. Beam search MUST interleave Evo2 with the ESM-2 scorer, so
verify that specific combination works before committing to the design."""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import mc_env; mc_env.patch()
from evo2_sae import load_evo2
from m1_harness_calibrate import natural_prompts
from mechanism import translate_orf
from scorers import ESM2HelixProbe, Ensemble
import m0_data as D

probe_ckpt = os.path.join(mc_env.MC, "results", "ss_probe.pt")
evo2 = load_evo2("cuda:0")
prompts, _, _ = natural_prompts(mc_env.ORGANISM, 4)
table = D.ORGANISMS[mc_env.ORGANISM]["transl_table"]
probe = ESM2HelixProbe(probe_ckpt, device="cuda:0") if os.path.exists(probe_ckpt) else None
ens = Ensemble(probe=probe)
print(f"[interleave] probe_loaded={probe is not None}", flush=True)

seqs = [prompts[0]] * 16
for step in range(3):
    t0 = time.time()
    out = evo2.generate(prompt_seqs=seqs, n_tokens=60, temperature=0.7, top_k=4, verbose=0)
    cont = out.sequences if hasattr(out, "sequences") else out
    seqs = [s + (c[len(s):] if c.startswith(s) else c) for s, c in zip(seqs, cont)]
    tg = time.time() - t0
    t1 = time.time()
    tr = [translate_orf(s, min_aa=5, table=table) for s in seqs]
    sc = ens([p for p, v, m in tr], [m.get("had_internal_stop", False) for p, v, m in tr])
    print(f"[interleave] step={step} len={len(seqs[0])} gen={tg:.2f}s score={time.time()-t1:.2f}s "
          f"score_range=[{sc.min():.2f},{sc.max():.2f}] aa={len(tr[0][0])}", flush=True)
print("[interleave] OK - Evo2 and ESM-2 coexist", flush=True)
