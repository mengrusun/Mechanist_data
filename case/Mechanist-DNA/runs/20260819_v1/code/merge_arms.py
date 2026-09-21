"""Merge M3 (600-valid) and M3b (1500-valid, disjoint seeds) per alpha into
m3final_a*.json for a high-power held-out C2/C3 confirmation. Seeds are disjoint
by construction (M3 seed_offset 0, M3b seed_offset 5000), so per_seq are pooled."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(__file__))
import gen_eval as GE
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")

pairs = {  # final_name : (m3 file, m3b file or None)
    "m3final_a0":    ("m3_a0.json",    "m3b_a0.json"),
    "m3final_ahalf": ("m3_ahalf.json", None),
    "m3final_astar": ("m3_astar.json", "m3b_astar.json"),
    "m3final_a2x":   ("m3_a2x.json",   "m3b_a2x.json"),
}

for fname, (fa, fb) in pairs.items():
    a = json.load(open(os.path.join(R, fa)))
    seen = set()
    per = []
    for p in a["per_seq"]:
        per.append(p); seen.add(p["seed"])
    if fb and os.path.exists(os.path.join(R, fb)):
        b = json.load(open(os.path.join(R, fb)))
        for p in b["per_seq"]:
            if p["seed"] not in seen:
                per.append(p); seen.add(p["seed"])
    a["per_seq"] = per
    a["n_gen"] = len(per)
    a["n_valid"] = sum(1 for p in per if p.get("qc_ok"))
    a["summary"] = GE.arm_summary(a)
    a["name"] = fname
    json.dump(a, open(os.path.join(R, f"{fname}.json"), "w"))
    s = a["summary"]
    print(f"{fname}: alpha={a['alpha']} n_gen={a['n_gen']} n_valid={s['n_valid']} "
          f"helix_cond={s['helix_cond_mean']:.3f} helix_itt={s['helix_itt_mean']:.3f} "
          f"valid_rate={s['valid_rate']:.2f}")
print("merged.")
