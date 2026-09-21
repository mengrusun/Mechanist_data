"""Train a per-block linear probe and report AUC on the test split.

Uses saved activation tensors from extract_activations.py which also stores
the original records (so we can recover their 'split' field).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--layer_choice", default="best", help="best | layer index (int)")
    args = ap.parse_args()

    data = torch.load(args.acts, weights_only=False)
    acts = data["activations"].float().numpy()   # (N, L, D)
    labels = data["labels"].numpy()
    records = data["records"]
    splits = np.array([r.get("split", "train") for r in records])

    train_mask = splits == "train"
    test_mask = splits == "test"
    print(f"train={train_mask.sum()} test={test_mask.sum()}")

    L = acts.shape[1]
    aucs = []
    for l in range(L):
        Xtr = acts[train_mask, l]
        ytr = labels[train_mask]
        Xte = acts[test_mask, l]
        yte = labels[test_mask]
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p)
        aucs.append(auc)

    best_l = int(np.argmax(aucs))
    best_auc = float(aucs[best_l])
    print(f"Best block: {best_l}, AUC={best_auc:.4f}")

    # Also compute accuracy and F1 at best
    l = best_l if args.layer_choice == "best" else int(args.layer_choice)
    Xtr = acts[train_mask, l]
    ytr = labels[train_mask]
    Xte = acts[test_mask, l]
    yte = labels[test_mask]
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(
            {
                "per_block_auc": aucs,
                "best_block": best_l,
                "best_auc": best_auc,
                "chosen_block": l,
                "chosen_auc": float(aucs[l]),
                "chosen_accuracy": float(acc),
                "chosen_f1": float(f1),
                "n_train": int(train_mask.sum()),
                "n_test": int(test_mask.sum()),
            },
            f,
            indent=2,
        )
    print(f"Saved to {args.out_json}")


if __name__ == "__main__":
    main()
