"""Analyse DeepSeek baseline."""
import json, numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/"results"

trials = [json.loads(l) for l in open(RES/"baseline_deepseek.jsonl")]

def enc_var(t):
    return np.array([
        1.0 if t["gender"] == "male" else 0.0,
        1.0 if t["age"] == "old" else 0.0,
        1.0 if t["instruction"] == "B" else 0.0,
        1.0 if t["meeting"] == "meeting" else 0.0,
    ])

X = np.stack([enc_var(t) for t in trials])
y = np.array([t["E_transfer"] for t in trials])
reg = LinearRegression().fit(X, y)
names = ["male", "old", "instrB", "meeting"]
print("[DeepSeek] E[transfer] regression:")
print("  intercept:", round(reg.intercept_, 3))
for n, c in zip(names, reg.coef_):
    print(f"  {n:8s} coef={c:+.3f}")
print("  R^2:", round(reg.score(X, y), 3))

# probes on baseline hidden states
H = np.load(RES/"baseline_deepseek.hidden.npy").astype(np.float32)
print(f"\n[DeepSeek] probe accs (5-fold CV), key layers:")
def label(v):
    if v == "gender": return np.array([1 if t["gender"]=="male" else 0 for t in trials])
    if v == "age":    return np.array([1 if t["age"]=="old" else 0 for t in trials])
    if v == "instruction": return np.array([1 if t["instruction"]=="B" else 0 for t in trials])
    if v == "meeting": return np.array([1 if t["meeting"]=="meeting" else 0 for t in trials])
out = {}
for v in ["gender", "age", "instruction", "meeting"]:
    accs = []
    yl = label(v)
    for li in [0, 4, 8, 12, 16, 20, 24, 28, 32]:
        Xl = H[:, li, :]
        mu = Xl.mean(0, keepdims=True); sd = Xl.std(0, keepdims=True)+1e-6
        Xln = (Xl - mu) / sd
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        fa = []
        for tr, te in skf.split(Xln, yl):
            clf = LogisticRegression(max_iter=200, C=1.0, solver="liblinear")
            clf.fit(Xln[tr], yl[tr])
            fa.append(accuracy_score(yl[te], clf.predict(Xln[te])))
        accs.append((li, float(np.mean(fa))))
    print(f"  {v:12s} {accs}")
    out[v] = dict(accs)
json.dump(out, open(RES/"probe_baseline_deepseek.json","w"), indent=2)
print("saved probe_baseline_deepseek.json")
