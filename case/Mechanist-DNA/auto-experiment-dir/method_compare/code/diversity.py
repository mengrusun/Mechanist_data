"""Sequence diversity / novelty of each delivered set (CPU only).

Rerank-style search can only pick from the tail of the base model's own distribution, so a
diversity collapse is the cost it pays for quality. Measured as mmseqs clusters at 50%/90%
identity and mean pairwise identity over the delivered proteins.
"""
import os, sys, json, glob, subprocess, tempfile, itertools
sys.path.insert(0, os.path.dirname(__file__))
import mc_env
import numpy as np


def write_fasta(path, recs):
    with open(path, "w") as fh:
        for r in recs:
            fh.write(f">s{r['idx']}\n{r['prot']}\n")


def mmseqs_clusters(fa, tmp, min_id):
    pref = os.path.join(tmp, f"cl{int(min_id*100)}")
    subprocess.run(["mmseqs", "easy-cluster", fa, pref, os.path.join(tmp, "t"),
                    "--min-seq-id", str(min_id), "-c", "0.8", "--cov-mode", "1", "-v", "0"],
                   check=True, capture_output=True)
    with open(pref + "_cluster.tsv") as fh:
        return len({ln.split("\t")[0] for ln in fh if ln.strip()})


def pid(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(x == y for x, y in zip(a[:n], b[:n])) / n


out = {}
for f in sorted(glob.glob(os.path.join(mc_env.MC, "results", "beam_*.json"))):
    d = json.load(open(f))
    recs = [r for r in d["records"] if r.get("prot")]
    name = os.path.basename(f)[5:-5]
    if len(recs) < 5:
        continue
    with tempfile.TemporaryDirectory() as tmp:
        fa = os.path.join(tmp, "in.fa")
        write_fasta(fa, recs)
        c50 = mmseqs_clusters(fa, tmp, 0.5)
        c90 = mmseqs_clusters(fa, tmp, 0.9)
    prots = [r["prot"] for r in recs]
    rng = np.random.default_rng(0)
    pairs = [(rng.integers(len(prots)), rng.integers(len(prots))) for _ in range(3000)]
    mp = float(np.mean([pid(prots[i], prots[j]) for i, j in pairs if i != j]))
    # novelty: identity to the natural CDS the prompt came from is not available per-record,
    # so report identity to the prompt-translated seed instead
    out[name] = {"n": len(recs), "clusters_50": c50, "clusters_90": c90,
                 "cluster_frac_50": c50 / len(recs), "cluster_frac_90": c90 / len(recs),
                 "mean_pairwise_identity": mp,
                 "mean_len_aa": float(np.mean([len(p) for p in prots]))}
    print(f"[div] {name:14s} n={len(recs):3d} clus50={c50:3d} clus90={c90:3d} pid={mp:.3f}", flush=True)
json.dump(out, open(os.path.join(mc_env.MC, "results", "diversity.json"), "w"), indent=2)
print("[div] WROTE diversity.json", flush=True)
