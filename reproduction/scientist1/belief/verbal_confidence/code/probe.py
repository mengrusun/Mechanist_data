"""
Linear-probing experiment: does the verbalized confidence live in hidden states
BEFORE the confidence token is emitted?

For each anchor position P in {pre, ans_last, post, conf} and each layer L, we
train a ridge regression to predict the verbalized confidence (0..100) from
h_{L, P}.  We report the 5-fold cross-validated coefficient-of-determination R^2
and the Spearman-rank correlation between predicted and true confidence.

The claim ("cached at ans_last, retrieved at post") predicts:
  - R^2(pre)      << R^2(ans_last) ~ R^2(post) ~ R^2(conf).
  - Time-of-emergence (first layer at which R^2 rises above chance) at
    ans_last should not be later than at post.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import spearmanr


POSITIONS = ["pre", "ans_last", "post", "conf"]


def load_dataset(hidden_dir: Path, tag: str):
    xs = {p: [] for p in POSITIONS}
    ys, qids, correct = [], [], []
    for f in sorted(hidden_dir.glob(f"{tag}__*.pt")):
        d = torch.load(f, weights_only=False)
        for p in POSITIONS:
            xs[p].append(d[p].numpy())
        ys.append(d["meta"]["confidence"])
        correct.append(bool(d["meta"]["correct"]))
        qids.append(d["meta"]["qid"])
    for p in POSITIONS:
        xs[p] = np.stack(xs[p], axis=0)  # (N, L, H)
    return xs, np.array(ys, dtype=np.float32), np.array(correct), qids


def probe(x: np.ndarray, y: np.ndarray, alpha: float = 1.0, k: int = 5, seed: int = 0):
    """Return (r2_mean, spearman_mean) with K-fold CV.

    x: (N, H), y: (N,)
    """
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    preds = np.zeros_like(y, dtype=np.float64)
    for train, test in kf.split(x):
        m = Ridge(alpha=alpha)
        m.fit(x[train], y[train])
        preds[test] = m.predict(x[test])
    # R^2 = 1 - SS_res / SS_tot
    ss_res = float(((y - preds) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    sp = float(spearmanr(preds, y).statistic)
    return r2, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    hidden_dir = Path(args.hidden_dir)
    out_path = Path(args.out)
    xs, y, correct, qids = load_dataset(hidden_dir, args.tag)

    N = y.shape[0]
    if N < 20:
        print(f"[warn] only {N} examples")
    L = xs["pre"].shape[1]
    print(f"[info] N={N} L={L} tag={args.tag}")
    print(f"[info] y distribution: unique={sorted(set(y.tolist()))}  mean={y.mean():.2f}  std={y.std():.2f}")
    print(f"[info] correct-rate: {correct.mean():.3f}")

    results = {"tag": args.tag, "N": int(N), "L": int(L),
               "y_mean": float(y.mean()), "y_std": float(y.std()),
               "correct_rate": float(correct.mean()),
               "positions": POSITIONS,
               "layers": list(range(L)),
               "r2": {p: [] for p in POSITIONS},
               "spearman": {p: [] for p in POSITIONS}}

    for p in POSITIONS:
        for l in range(L):
            r2, sp = probe(xs[p][:, l, :], y, alpha=args.alpha, k=args.k)
            results["r2"][p].append(r2)
            results["spearman"][p].append(sp)
        # print peak
        best_l = int(np.argmax(results["r2"][p]))
        print(f"  pos={p:10s}  peak-R2={results['r2'][p][best_l]:.3f} @ layer {best_l}  "
              f"peak-Spearman={max(results['spearman'][p]):.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
