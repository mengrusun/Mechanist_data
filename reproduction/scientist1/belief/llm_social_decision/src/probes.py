"""Claim 1: are variable values linearly extractable from residual stream?
Per-layer logistic-regression probe, 5-fold CV, using pair activations."""

import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "directions_llama"

VARS = ["gender", "age", "instruction", "meeting"]

def probe_variable(var, seed=0):
    Ha = np.load(RES/f"H_{var}_a.npy").astype(np.float32)  # (N, L+1, H)
    Hb = np.load(RES/f"H_{var}_b.npy").astype(np.float32)
    N, L, H = Ha.shape
    X = np.concatenate([Ha, Hb], axis=0)  # (2N, L, H)
    y = np.concatenate([np.zeros(N), np.ones(N)])  # a=0, b=1
    accs = []
    for li in range(L):
        Xl = X[:, li, :]
        # z-score
        mu, sd = Xl.mean(0, keepdims=True), Xl.std(0, keepdims=True)+1e-6
        Xln = (Xl - mu) / sd
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_accs = []
        for tr, te in skf.split(Xln, y):
            clf = LogisticRegression(max_iter=200, C=1.0, solver="liblinear")
            clf.fit(Xln[tr], y[tr])
            fold_accs.append(accuracy_score(y[te], clf.predict(Xln[te])))
        accs.append(np.mean(fold_accs))
    return np.array(accs)


def main():
    out = {}
    for v in VARS:
        print(f"probing {v} ...")
        accs = probe_variable(v)
        # summary
        best_l = int(np.argmax(accs))
        print(f"  {v}: best layer {best_l} acc={accs[best_l]:.3f}, layer0 acc={accs[0]:.3f}, layer{len(accs)-1}={accs[-1]:.3f}")
        out[v] = accs.tolist()
    with open(ROOT/"results"/"probe_accs.json","w") as f:
        json.dump(out, f, indent=2)
    print("saved probe_accs.json")


if __name__ == "__main__":
    main()
