"""Analyse baseline results: per-variable effect sizes on E[transfer]."""

import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

trials = [json.loads(l) for l in open(RES/"baseline_llama.jsonl")]

def enc_var(t):
    return np.array([
        1.0 if t["gender"] == "male" else 0.0,
        1.0 if t["age"] == "old" else 0.0,
        1.0 if t["instruction"] == "B" else 0.0,
        1.0 if t["meeting"] == "meeting" else 0.0,
    ])

X = np.stack([enc_var(t) for t in trials])
y = np.array([t["E_transfer"] for t in trials])
y_am = np.array([t["argmax"] for t in trials], dtype=float)

reg = LinearRegression().fit(X, y)
names = ["male", "old", "instrB", "meeting"]
print("E[transfer] regression on {"+", ".join(names)+"}:")
print("  intercept:", round(reg.intercept_, 3))
for n, c in zip(names, reg.coef_):
    print(f"  {n:8s} coef={c:+.3f}")
print("  R^2:", round(reg.score(X, y), 3))

print()
print("Group means of E[transfer]:")
def group(mask, name):
    print(f"  {name:20s} n={mask.sum():4d} mean={y[mask].mean():.3f}  argmax_mean={y_am[mask].mean():.3f}")

group(X[:,0]==1, "gender=male")
group(X[:,0]==0, "gender=female")
group(X[:,1]==1, "age=old")
group(X[:,1]==0, "age=young")
group(X[:,2]==1, "instr=B")
group(X[:,2]==0, "instr=A")
group(X[:,3]==1, "meeting")
group(X[:,3]==0, "no_meeting")

# Also check argmax distribution
from collections import Counter
print()
print("argmax distribution overall:", dict(Counter([int(v) for v in y_am])))
