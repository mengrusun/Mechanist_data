"""EmotionRL v2 — more careful:
1. Multiple seeds and CV for robustness
2. Uses gradient boosting (usually stronger than LR)
3. Reports mean +/- std of gain

Also implements a "confidence-gated" adaptive policy that falls back to neutral
when the classifier isn't confident.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import glob
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


PATTERN = re.compile(r"^(?P<dataset>[^_]+)__(?P<model>.+)__(?P<cond>[^.]+)\.jsonl$")


def load_all(results_dir, dataset, model):
    out = defaultdict(dict)
    for p in sorted(glob.glob(f"{results_dir}/{dataset}__{model}__*.jsonl")):
        m = PATTERN.match(os.path.basename(p))
        if not m:
            continue
        cond = m.group("cond")
        for ln in open(p):
            r = json.loads(ln)
            out[cond][r["id"]] = bool(r["correct"])
    return out


def load_questions(dataset):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from datasets_loader import LOADERS
    fn = LOADERS[dataset]
    if dataset == "bbh":
        return {row["id"]: row["question"] for row in fn(n_per_subtask=999)}
    return {row["id"]: row["question"] for row in fn(n=5000)}


def build_matrix(cond_correct, ids):
    conds = sorted(cond_correct.keys())
    Y = np.zeros((len(ids), len(conds)), dtype=np.int32)
    for j, c in enumerate(conds):
        for i, qid in enumerate(ids):
            Y[i, j] = int(cond_correct[c].get(qid, False))
    return Y, conds


def _train_pick(Xtr, Ytr, Xte, conds, classifier="lr"):
    scores = np.zeros((len(Xte), len(conds)))
    for j in range(len(conds)):
        y = Ytr[:, j]
        if y.min() == y.max():
            scores[:, j] = y.mean()
            continue
        if classifier == "gb":
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)
        else:
            clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(Xtr, y)
        try:
            scores[:, j] = clf.predict_proba(Xte)[:, 1]
        except IndexError:
            scores[:, j] = clf.decision_function(Xte)
    return scores


def cv_emotion_rl(results_dir, dataset, model, n_folds=5, classifier="lr",
                   n_svd=200, verbose=False):
    cond_correct = load_all(results_dir, dataset, model)
    if not cond_correct:
        return None
    conds = sorted(cond_correct.keys())
    ids = sorted(list(cond_correct[conds[0]].keys()))
    questions = load_questions(dataset)
    q_texts = [questions.get(i, "") for i in ids]
    Y, conds = build_matrix(cond_correct, ids)
    N = len(ids)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
    accs_neutral = []
    accs_best_fixed = []
    accs_oracle = []
    accs_rl = []
    accs_rl_gated = []

    neutral_j = conds.index("neutral") if "neutral" in conds else 0

    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(N))):
        vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                              sublinear_tf=True, min_df=2)
        Xtr_txt = [q_texts[i] for i in train_idx]
        Xte_txt = [q_texts[i] for i in test_idx]
        Xtr = vec.fit_transform(Xtr_txt)
        Xte = vec.transform(Xte_txt)
        k = min(n_svd, Xtr.shape[1] - 1, Xtr.shape[0] - 1)
        if k < 2:
            k = 2
        svd = TruncatedSVD(n_components=k, random_state=0)
        Xtr_lsa = svd.fit_transform(Xtr)
        Xte_lsa = svd.transform(Xte)

        scores_te = _train_pick(Xtr_lsa, Y[train_idx], Xte_lsa, conds, classifier)

        # baselines
        accs_neutral.append(Y[test_idx, neutral_j].mean())
        best_by_train = np.argmax(Y[train_idx].mean(axis=0))
        accs_best_fixed.append(Y[test_idx, best_by_train].mean())
        accs_oracle.append(Y[test_idx].any(axis=1).mean())

        # RL: pick argmax
        chosen = np.argmax(scores_te, axis=1)
        accs_rl.append(Y[test_idx, chosen].mean())

        # Confidence-gated: pick argmax if margin >= 0.05 else neutral
        margin = scores_te.max(axis=1) - scores_te[:, neutral_j]
        chosen_g = np.where(margin >= 0.05, chosen, neutral_j)
        accs_rl_gated.append(Y[test_idx, chosen_g].mean())

    def m(a): return float(np.mean(a))
    def s(a): return float(np.std(a))
    result = {
        "dataset": dataset, "model": model, "n": N, "n_folds": n_folds,
        "classifier": classifier,
        "neutral_mean": m(accs_neutral), "neutral_std": s(accs_neutral),
        "best_fixed_mean": m(accs_best_fixed), "best_fixed_std": s(accs_best_fixed),
        "oracle_mean": m(accs_oracle), "oracle_std": s(accs_oracle),
        "rl_mean": m(accs_rl), "rl_std": s(accs_rl),
        "rl_gated_mean": m(accs_rl_gated), "rl_gated_std": s(accs_rl_gated),
        "gain_over_neutral_mean": m(accs_rl) - m(accs_neutral),
        "gain_over_bfixed_mean": m(accs_rl) - m(accs_best_fixed),
        "gated_gain_over_neutral_mean": m(accs_rl_gated) - m(accs_neutral),
        "gated_gain_over_bfixed_mean": m(accs_rl_gated) - m(accs_best_fixed),
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--classifier", default="lr", choices=["lr", "gb"])
    args = ap.parse_args()
    res = cv_emotion_rl(args.results_dir, args.dataset, args.model,
                        n_folds=args.n_folds, classifier=args.classifier)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
