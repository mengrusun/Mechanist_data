"""
Step 5: Statistical test for orthogonality.

Bootstrap-resample training data to obtain a distribution of probe directions,
then compute the distribution of within-probe cosine similarity (same signal,
different bootstrap sample) vs. across-probe cosine similarity.

If across-probe |cos| << within-probe |cos|, then the two signals really live in
distinguishable directions (i.e., the near-orthogonality is not just noise).

Also emits: a scatter plot of test-set predicted P(correct) vs verbalized-conf.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def train_logreg(H, y):
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(H, y)
    return clf.coef_.reshape(-1), clf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--position", default="answer_end")
    ap.add_argument("--n_boot", type=int, default=25)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_plot", required=True)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    data = np.load(args.npz)
    H = data[f"hs_{args.position}"].astype(np.float32)
    correct = data["correct"].astype(int)
    conf = data["confidence"].astype(np.float32)
    N, L1, D = H.shape
    # if layer < 0, choose the layer that maximizes A_AUC on a small holdout
    if args.layer < 0:
        idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.3, random_state=0, stratify=correct)
        best, best_layer = -1, 0
        for L in range(L1):
            clf = LogisticRegression(C=1.0, max_iter=1000)
            clf.fit(H[idx_tr, L], correct[idx_tr])
            p = clf.predict_proba(H[idx_te, L])[:, 1]
            from sklearn.metrics import roc_auc_score
            a = roc_auc_score(correct[idx_te], p)
            if a > best:
                best = a
                best_layer = L
        args.layer = best_layer
        print(f"Chosen best layer for correctness: {args.layer} (AUC {best:.3f})")

    HL = H[:, args.layer, :]
    is_high = (conf >= 100).astype(int)

    # Bootstrap
    idx_tr, idx_te = train_test_split(np.arange(N), test_size=0.3, random_state=0, stratify=correct)
    rng = np.random.default_rng(0)

    dirs_A, dirs_C = [], []
    for b in range(args.n_boot):
        sel = rng.choice(idx_tr, size=len(idx_tr), replace=True)
        # ensure both classes present for both tasks
        if len(set(correct[sel])) < 2 or len(set(is_high[sel])) < 2:
            continue
        wA, _ = train_logreg(HL[sel], correct[sel])
        wC, _ = train_logreg(HL[sel], is_high[sel])
        dirs_A.append(wA)
        dirs_C.append(wC)
    dirs_A = np.stack(dirs_A)
    dirs_C = np.stack(dirs_C)
    print(f"Bootstrap directions: A={dirs_A.shape} C={dirs_C.shape}")

    def within(dirs):
        n = len(dirs)
        vals = []
        for i in range(n):
            for j in range(i + 1, n):
                vals.append(cos(dirs[i], dirs[j]))
        return np.array(vals)

    def across(a, c):
        vals = []
        for wa in a:
            for wc in c:
                vals.append(cos(wa, wc))
        return np.array(vals)

    within_A = within(dirs_A)
    within_C = within(dirs_C)
    across_AC = across(dirs_A, dirs_C)

    # Random baseline: random unit vectors in D-space
    rnd = rng.normal(size=(200, HL.shape[1]))
    rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    rand_pair_cos = []
    for i in range(0, len(rnd) - 1, 2):
        rand_pair_cos.append(cos(rnd[i], rnd[i + 1]))
    rand_pair_cos = np.array(rand_pair_cos)

    stats = {
        "layer": int(args.layer),
        "position": args.position,
        "within_A_mean_abs_cos": float(np.mean(np.abs(within_A))),
        "within_C_mean_abs_cos": float(np.mean(np.abs(within_C))),
        "across_AC_mean_abs_cos": float(np.mean(np.abs(across_AC))),
        "across_AC_mean_cos": float(np.mean(across_AC)),
        "random_pair_std_cos": float(np.std(rand_pair_cos)),
        "expected_random_abs_cos_1_over_sqrtD": float(1 / np.sqrt(HL.shape[1])),
        "hidden_dim": int(HL.shape[1]),
        "n_boot": int(args.n_boot),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

    # plot
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(-1, 1, 60)
    ax.hist(within_A, bins=bins, alpha=0.5, label=f"within(A): |cos|={stats['within_A_mean_abs_cos']:.2f}", color="C0")
    ax.hist(within_C, bins=bins, alpha=0.5, label=f"within(C): |cos|={stats['within_C_mean_abs_cos']:.2f}", color="C1")
    ax.hist(across_AC, bins=bins, alpha=0.5, label=f"across(A,C): |cos|={stats['across_AC_mean_abs_cos']:.2f}", color="C2")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("count")
    ax.set_title(f"Direction cosines at L{args.layer} ({args.position}) {args.tag}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out_plot, dpi=150)
    plt.close(fig)
    print("Plot saved to", args.out_plot)


if __name__ == "__main__":
    main()
