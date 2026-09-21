"""Score every delivered sequence with the two scorer components, so we can measure the scorer's
realised rank-correlation with the held-out structural endpoint ON GENERATED SEQUENCES.

This is the number the whole comparison hinges on: the probe's 0.97 rho on natural held-out
proteins says nothing about its reliability on out-of-distribution model output.
"""
import os, sys, json, glob, time
sys.path.insert(0, os.path.dirname(__file__))
import mc_env; mc_env.patch()
from scorers import ESM2HelixProbe, chou_fasman

probe = ESM2HelixProbe(os.path.join(mc_env.MC, "results", "ss_probe.pt"), device="cuda:0")
files = sorted(glob.glob(os.path.join(mc_env.MC, "results", "beam_*.json")))
for f in files:
    d = json.load(open(f))
    recs = [r for r in d["records"] if r.get("prot")]
    if not recs:
        continue
    prots = [r["prot"] for r in recs]
    t0 = time.time()
    pv, cv = probe(prots), chou_fasman(prots)
    for r, a, b in zip(recs, pv, cv):
        r["score_probe"] = float(a)
        r["score_cf"] = float(b)
    json.dump(d, open(f, "w"), indent=2)
    print(f"[score] {os.path.basename(f)} n={len(recs)} ({time.time()-t0:.0f}s)", flush=True)
print("[score] done", flush=True)
