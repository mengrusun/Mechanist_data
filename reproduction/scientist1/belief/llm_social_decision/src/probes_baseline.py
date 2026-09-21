"""Stronger Claim-1 test: from the residual stream of the 1000 randomised
baseline dictator prompts, predict each of G, A, I, M via linear probe (5-fold CV)."""

import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/"results"

H = np.load(RES/"baseline_llama.hidden.npy").astype(np.float32)  # (1000, L+1, hid)
trials = [json.loads(l) for l in open(RES/"baseline_llama.jsonl")]
N, L, dim = H.shape
print(f"H shape {H.shape}; N={N}, layers={L}, hidden={dim}")

def label(v):
    if v == "gender": return np.array([1 if t["gender"]=="male" else 0 for t in trials])
    if v == "age":    return np.array([1 if t["age"]=="old" else 0 for t in trials])
    if v == "instruction": return np.array([1 if t["instruction"]=="B" else 0 for t in trials])
    if v == "meeting": return np.array([1 if t["meeting"]=="meeting" else 0 for t in trials])

VARS = ["gender", "age", "instruction", "meeting"]

out = {}
for v in VARS:
    y = label(v)
    print(f"\n== probing {v} (positive class base rate = {y.mean():.2f}) ==")
    accs = []
    for li in range(L):
        Xl = H[:, li, :]
        mu = Xl.mean(0, keepdims=True); sd = Xl.std(0, keepdims=True)+1e-6
        Xln = (Xl - mu) / sd
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        fa = []
        for tr, te in skf.split(Xln, y):
            clf = LogisticRegression(max_iter=200, C=1.0, solver="liblinear")
            clf.fit(Xln[tr], y[tr])
            fa.append(accuracy_score(y[te], clf.predict(Xln[te])))
        accs.append(float(np.mean(fa)))
    best = int(np.argmax(accs))
    print(f"  layers 0..{L-1}: acc@0={accs[0]:.3f}  acc@8={accs[8]:.3f}  acc@16={accs[16]:.3f}  acc@24={accs[24]:.3f}  best@{best}={accs[best]:.3f}")
    out[v] = accs

with open(RES/"probe_baseline_accs.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nsaved probe_baseline_accs.json")
