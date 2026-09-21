"""EmotionRL: an offline-trained per-query prefix selector.

We treat this as a supervised classification problem:
  input:  question text (embedded via TF-IDF + LSA / or model tokenizer embeddings)
  target: which prefix condition would answer this query correctly.

Because typically several conditions answer a given query correctly, we soften
the target as a multi-label. At inference we pick the condition with the highest
predicted score. Ties go to `neutral` (safe default).

We do a leave-one-out style split: 50% train, 50% test (deterministic seed).
The upper bound is the oracle "pick any correct condition per query".
"""
from __future__ import annotations

import argparse
import json
import os
import glob
import re
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD


PATTERN = re.compile(r"^(?P<dataset>[^_]+)__(?P<model>.+)__(?P<cond>[^.]+)\.jsonl$")


def load_all(results_dir, dataset, model):
    out = defaultdict(dict)  # cond -> {id: (correct, question)}
    q_map = {}
    for p in sorted(glob.glob(f"{results_dir}/{dataset}__{model}__*.jsonl")):
        m = PATTERN.match(os.path.basename(p))
        if not m:
            continue
        cond = m.group("cond")
        for ln in open(p):
            r = json.loads(ln)
            out[cond][r["id"]] = bool(r["correct"])
    return out


def load_questions(dataset, ids, n_load=None):
    """Load question text for the ids from datasets_loader."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datasets_loader import LOADERS
    fn = LOADERS[dataset]
    if dataset == "bbh":
        data = fn(n_per_subtask=999)
    else:
        data = fn(n=n_load or 5000)
    return {row["id"]: row["question"] for row in data}


def build_matrix(cond_correct, ids):
    """Return (X_ids, Y[N, K], cond_names)."""
    conds = sorted(cond_correct.keys())
    Y = np.zeros((len(ids), len(conds)), dtype=np.int32)
    for j, c in enumerate(conds):
        for i, qid in enumerate(ids):
            Y[i, j] = int(cond_correct[c].get(qid, False))
    return Y, conds


def emotion_rl(results_dir, dataset, model, seed=0, train_frac=0.5, verbose=False):
    cond_correct = load_all(results_dir, dataset, model)
    conds = sorted(cond_correct.keys())
    if not conds:
        print(f"No conditions found for {dataset} / {model}")
        return None
    ids = sorted(list(cond_correct[conds[0]].keys()))
    questions = load_questions(dataset, ids)
    q_texts = [questions.get(i, "") for i in ids]

    Y, conds = build_matrix(cond_correct, ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    n_train = int(len(ids) * train_frac)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    # Baseline stats on test set
    acc_neutral_test = Y[test_idx, conds.index("neutral")].mean()
    best_cond_by_train_acc = conds[np.argmax(Y[train_idx].mean(axis=0))]
    acc_best_fixed_test = Y[test_idx, conds.index(best_cond_by_train_acc)].mean()
    oracle_test = Y[test_idx].any(axis=1).mean()

    # Train per-condition binary classifier (multi-label style) and pick top condition.
    # Use TF-IDF + TruncatedSVD to keep feature dim modest.
    Xtr_text = [q_texts[i] for i in train_idx]
    Xte_text = [q_texts[i] for i in test_idx]

    # Vectorize
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                          sublinear_tf=True, min_df=2)
    Xtr = vec.fit_transform(Xtr_text)
    Xte = vec.transform(Xte_text)
    svd = TruncatedSVD(n_components=min(200, Xtr.shape[1] - 1, Xtr.shape[0] - 1), random_state=seed)
    Xtr_lsa = svd.fit_transform(Xtr)
    Xte_lsa = svd.transform(Xte)

    # Predict multi-label (which conditions will be correct)
    # We use per-class LogisticRegression on P(correct | features).
    scores_te = np.zeros((len(test_idx), len(conds)))
    for j, c in enumerate(conds):
        ytr = Y[train_idx, j]
        if ytr.min() == ytr.max():
            # constant column -> use its constant
            scores_te[:, j] = ytr.mean()
            continue
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(Xtr_lsa, ytr)
        try:
            scores_te[:, j] = clf.predict_proba(Xte_lsa)[:, 1]
        except IndexError:
            scores_te[:, j] = clf.decision_function(Xte_lsa)

    # Choose highest predicted correctness per query
    chosen = np.argmax(scores_te, axis=1)
    acc_rl = Y[test_idx, chosen].mean()

    # Confidence-aware policy: if margin (top - neutral score) is small, default to
    # neutral. This can guard against noise.
    neutral_j = conds.index("neutral")
    margin = scores_te.max(axis=1) - scores_te[:, neutral_j]
    for thr in [0.0, 0.03, 0.05, 0.10]:
        chosen_c = np.where(margin >= thr, chosen, neutral_j)
        acc_c = Y[test_idx, chosen_c].mean()
        if verbose:
            print(f"  margin_thr={thr:.2f}: acc={acc_c:.4f}")

    result = {
        "dataset": dataset, "model": model, "n_test": int(len(test_idx)),
        "neutral": float(acc_neutral_test),
        "best_fixed_by_train": best_cond_by_train_acc,
        "best_fixed_acc": float(acc_best_fixed_test),
        "oracle": float(oracle_test),
        "emotion_rl": float(acc_rl),
        "gain_over_neutral": float(acc_rl - acc_neutral_test),
        "gain_over_best_fixed": float(acc_rl - acc_best_fixed_test),
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    res = emotion_rl(args.results_dir, args.dataset, args.model, seed=args.seed,
                     verbose=args.verbose)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
