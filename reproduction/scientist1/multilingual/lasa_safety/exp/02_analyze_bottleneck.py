"""
Claim-1 analysis on cached per-layer hidden states.

Two complementary probes per layer:

(A) Language identifiability
    - Logistic regression 5-fold CV: hidden -> lang label. Higher accuracy
      => layer encodes language identity strongly.

(B) Parallel-alignment / semantic retrieval
    - For every ordered pair of languages (L1, L2) and every prompt p,
      rank the L2 vectors by cosine distance to the L1 vector of p.
      Retrieval accuracy = fraction of prompts whose true translation is top-1.
      Average over all L1!=L2 pairs. Higher => semantic content dominates language.

The "semantic bottleneck" layer is (approximately) where (B) is highest while
(A) has already dropped.
"""

import os, argparse, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

def cosine(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return a @ b.T

def retrieval_accuracy_all_pairs(reps_layer):
    """reps_layer: [N, K, H]. Returns (mean_top1_acc, mean_mrr, per_pair_dict)."""
    N, K, H = reps_layer.shape
    top1s, mrrs = [], []
    per_pair = {}
    for i in range(K):
        for j in range(K):
            if i == j: continue
            a = reps_layer[:, i, :].astype(np.float32)
            b = reps_layer[:, j, :].astype(np.float32)
            sim = cosine(a, b)                                # [N, N]
            order = np.argsort(-sim, axis=1)                  # descending
            gold = np.arange(N)[:, None]
            ranks = np.where(order == gold)[1]                # rank of gold
            top1 = float((ranks == 0).mean())
            mrr  = float((1.0 / (ranks + 1.0)).mean())
            top1s.append(top1); mrrs.append(mrr)
            per_pair[f"{i}->{j}"] = {"top1": top1, "mrr": mrr}
    return float(np.mean(top1s)), float(np.mean(mrrs)), per_pair

def lang_probe_accuracy(reps_layer, seed=0):
    """reps_layer: [N, K, H] -> logistic-regression 5-fold accuracy on lang labels."""
    N, K, H = reps_layer.shape
    X = reps_layer.reshape(N * K, H).astype(np.float32)
    y = np.tile(np.arange(K), N)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=200, n_jobs=-1, C=1.0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, Xs, y, cv=skf, n_jobs=-1)
    return float(scores.mean()), float(scores.std())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--every", type=int, default=1, help="analyze every k-th layer")
    args = ap.parse_args()

    d = np.load(args.reps, allow_pickle=True)
    reps = d["reps"]                    # [L+1, N, K, H] float16
    langs = list(d["langs"])
    L1, N, K, H = reps.shape
    print(f"[data] reps shape = {reps.shape}, langs={langs}")

    results = {"langs": langs, "layers": [], "lang_probe": [], "retrieval_top1": [], "retrieval_mrr": []}
    for l in range(0, L1, args.every):
        rl = reps[l].astype(np.float32)     # [N, K, H]
        acc, sd = lang_probe_accuracy(rl)
        r1, rr, per_pair = retrieval_accuracy_all_pairs(rl)
        print(f"[layer {l:>2}] lang_probe={acc:.3f}±{sd:.3f}  retrieval_top1={r1:.3f}  mrr={rr:.3f}")
        results["layers"].append(l)
        results["lang_probe"].append(acc)
        results["retrieval_top1"].append(r1)
        results["retrieval_mrr"].append(rr)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] {args.out}")

if __name__ == "__main__":
    main()
