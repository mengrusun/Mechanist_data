#!/usr/bin/env python3
"""
Train single-agent linear probes for each candidate layer on the extracted activations.

For each layer:
  - Split scenarios into train/dev/test (70/10/20) stratified by (domain, condition).
  - Flatten to per-agent examples: (scenario_i, agent_k) -> label(scenario_i). K agents per scenario
    all share the scenario-level label (a scenario is collusive => all K agents are labeled 1).
  - Train a logistic regression with L2 regularization on train, tune C on dev, evaluate test.
  - Report AUROC, AUROC@1%FPR.
  - Run three sanity controls: label-permute, matched-length, topic-swap.
  - Save the probe (sklearn pickled) + per-scenario predictions + summary metrics.

Output: runs/M1/probe_layer{L}.json (+ probe_layer{L}.pkl for the sklearn model).
"""
import argparse
import json
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler


def stratified_split(scenario_ids, domains, labels, seed=42, train_frac=0.7, dev_frac=0.1):
    """Return three lists of scenario indices: train, dev, test. Stratified by (domain, label)."""
    rng = random.Random(seed)
    buckets = {}
    for i, (d, y) in enumerate(zip(domains, labels)):
        buckets.setdefault((d, int(y)), []).append(i)
    train, dev, test = [], [], []
    for k, idxs in buckets.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_tr = max(1, int(round(n * train_frac))) if n > 2 else n
        n_dev = max(1, int(round(n * dev_frac))) if n - n_tr > 1 else 0
        train.extend(idxs[:n_tr])
        dev.extend(idxs[n_tr : n_tr + n_dev])
        test.extend(idxs[n_tr + n_dev :])
    return sorted(train), sorted(dev), sorted(test)


def tpr_at_fpr(y_true, y_score, target_fpr=0.01):
    """True positive rate at a given false positive rate. Uses interpolation on the ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    if len(fpr) < 2:
        return float("nan")
    return float(np.interp(target_fpr, fpr, tpr))


def flatten_per_agent(activations, labels, indices, layer_index, K):
    """
    activations: [N, K, n_layers, d] (may be fp16)
    Returns X, y (per-agent), scenario_map (list mapping each row -> scenario index).
    """
    X_list, y_list, sc_list = [], [], []
    for i in indices:
        for k in range(K):
            # Cast fp16 -> fp32 for sklearn
            X_list.append(activations[i, k, layer_index, :].float().numpy())
            y_list.append(int(labels[i].item()))
            sc_list.append(i)
    return np.stack(X_list), np.array(y_list), np.array(sc_list)


def per_scenario_score(agent_scores, scenario_map, N_scenarios):
    """Reduce per-agent probe scores to per-scenario by mean (baseline for single-agent probe report)."""
    out = np.full(N_scenarios, np.nan)
    for i in range(N_scenarios):
        mask = scenario_map == i
        if mask.any():
            out[i] = agent_scores[mask].mean()
    return out


def train_and_eval_probe(X_tr, y_tr, X_dv, y_dv, X_te, y_te, Cs=(0.01, 0.1, 1.0, 10.0), seed=42):
    """Standardize, sweep C on dev, refit, evaluate on test."""
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr)
    X_dv_s = sc.transform(X_dv) if len(X_dv) else None
    X_te_s = sc.transform(X_te)

    best_C, best_dv = None, -1
    if len(X_dv) and len(set(y_dv.tolist())) == 2:
        for C in Cs:
            clf = LogisticRegression(C=C, max_iter=2000, random_state=seed, class_weight="balanced")
            clf.fit(X_tr_s, y_tr)
            s = clf.decision_function(X_dv_s)
            au = roc_auc_score(y_dv, s)
            if au > best_dv:
                best_dv, best_C = au, C
    else:
        best_C = 1.0

    clf = LogisticRegression(C=best_C, max_iter=2000, random_state=seed, class_weight="balanced")
    clf.fit(X_tr_s, y_tr)
    te_s = clf.decision_function(X_te_s)
    dv_s = clf.decision_function(X_dv_s) if X_dv_s is not None else None
    te_auroc = roc_auc_score(y_te, te_s) if len(set(y_te.tolist())) == 2 else float("nan")
    te_at_1 = tpr_at_fpr(y_te, te_s, 0.01) if len(set(y_te.tolist())) == 2 else float("nan")
    return {
        "scaler": sc,
        "clf": clf,
        "best_C": best_C,
        "dev_auroc": float(best_dv) if best_dv >= 0 else float("nan"),
        "test_auroc": float(te_auroc),
        "test_tpr_at_1pct_fpr": float(te_at_1),
        "test_scores": te_s,
        "dev_scores": dv_s,
    }


def sanity_label_permute(X_tr, y_tr, X_te, y_te, seed=42):
    rng = np.random.RandomState(seed)
    y_perm = rng.permutation(y_tr)
    r = train_and_eval_probe(X_tr, y_perm, X_tr[:0], y_tr[:0], X_te, y_te, seed=seed)
    return {"auroc": r["test_auroc"]}


def sanity_matched_length(activations, labels, domains, scenario_ids, indices, layer_index, K, responses, seed=42):
    """
    Bucket scenarios by response-length bucket and re-eval within same bucket.
    Approx: length = mean chars per agent response.
    Returns AUROC where positive and negative examples are matched in length bucket.
    """
    # compute mean length per scenario
    lens = np.array([np.mean([len(r) for r in responses[i]]) for i in range(len(responses))])
    med = np.median(lens[indices])
    long_mask = lens >= med
    # keep indices in a single bucket at a time (either both long or both short), match sizes
    idx_np = np.array(indices)
    long_idx = idx_np[long_mask[idx_np]]
    short_idx = idx_np[~long_mask[idx_np]]

    # pick per-bucket AUROC weighted average
    aurocs = []
    for bucket_idx in (long_idx, short_idx):
        if len(bucket_idx) < 20:
            continue
        bucket_idx = sorted(bucket_idx.tolist())
        # split within bucket 80/20 train/test
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(bucket_idx))
        n_tr = int(0.8 * len(bucket_idx))
        tr_idx = [bucket_idx[i] for i in perm[:n_tr]]
        te_idx = [bucket_idx[i] for i in perm[n_tr:]]
        X_tr, y_tr, _ = flatten_per_agent(activations, labels, tr_idx, layer_index, K)
        X_te, y_te, _ = flatten_per_agent(activations, labels, te_idx, layer_index, K)
        if len(set(y_tr.tolist())) < 2 or len(set(y_te.tolist())) < 2:
            continue
        r = train_and_eval_probe(X_tr, y_tr, X_tr[:0], y_tr[:0], X_te, y_te, seed=seed)
        aurocs.append(r["test_auroc"])
    if not aurocs:
        return {"auroc": float("nan"), "note": "insufficient bucket coverage"}
    return {"auroc": float(np.mean(aurocs)), "per_bucket": aurocs}


def sanity_topic_swap(activations, labels, domains, indices, layer_index, K, seed=42):
    """
    Leave-one-domain-out AUROC: train on 16 domains, test on the held-out domain. Report mean.
    If a domain is missing from `indices`, skip it.
    """
    domain_set = sorted(set(domains[i] for i in indices))
    aurocs = []
    for held in domain_set:
        train_idx = [i for i in indices if domains[i] != held]
        test_idx = [i for i in indices if domains[i] == held]
        if len(train_idx) < 30 or len(test_idx) < 5:
            continue
        X_tr, y_tr, _ = flatten_per_agent(activations, labels, train_idx, layer_index, K)
        X_te, y_te, _ = flatten_per_agent(activations, labels, test_idx, layer_index, K)
        if len(set(y_tr.tolist())) < 2 or len(set(y_te.tolist())) < 2:
            continue
        r = train_and_eval_probe(X_tr, y_tr, X_tr[:0], y_tr[:0], X_te, y_te, seed=seed)
        aurocs.append({"held": held, "auroc": r["test_auroc"]})
    if not aurocs:
        return {"auroc": float("nan"), "per_held": []}
    return {"auroc": float(np.mean([a["auroc"] for a in aurocs])), "per_held": aurocs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations", required=True)
    ap.add_argument("--layer", type=int, required=True, help="1-based layer index as stored in payload['layers']")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sanity-checks", default="label-permute,length-match,topic-swap")
    ap.add_argument("--floor-per-class", type=int, default=25, help="min scenarios per class per split (warn only)")
    args = ap.parse_args()

    payload = torch.load(args.activations, map_location="cpu", weights_only=False)
    activations = payload["activations"]  # [N, K, n_layers, d]
    labels = payload["labels"]            # [N]
    domains = payload["domains"]          # [N]
    scenario_ids = payload["scenario_ids"] # [N]
    responses = payload["responses"]       # [N][K]
    layers_stored = payload["layers"]      # e.g. [27, 37, 48, 59]
    K = payload["K"]
    # Filter invalid rows (failed extractions)
    if "valid_mask" in payload:
        vm = payload["valid_mask"].numpy().astype(bool)
        n_before = activations.shape[0]
        activations = activations[vm]
        labels = labels[vm]
        domains = [d for d, v in zip(domains, vm) if v]
        scenario_ids = [s for s, v in zip(scenario_ids, vm) if v]
        responses = [r for r, v in zip(responses, vm) if v]
        print(f"[probe] filtered invalid rows: kept {activations.shape[0]}/{n_before}", flush=True)
    N = activations.shape[0]

    if args.layer not in layers_stored:
        raise ValueError(f"layer {args.layer} not in stored layers {layers_stored}")
    layer_index = layers_stored.index(args.layer)

    print(f"[probe layer={args.layer}] N={N}, class balance = {Counter(labels.tolist())}", flush=True)
    train_idx, dev_idx, test_idx = stratified_split(scenario_ids, domains, labels, seed=args.seed)
    print(f"[probe] split sizes: train={len(train_idx)}, dev={len(dev_idx)}, test={len(test_idx)}", flush=True)

    # Split-floor check: each split needs at least MIN_PER_CLASS per class (default 25 to be
    # tolerant of the actual generated size; the plan's ideal was 50 per class per split).
    def _class_counts(idxs):
        c = Counter(int(labels[i].item()) for i in idxs)
        return c.get(0, 0), c.get(1, 0)
    for name, idxs in (("train", train_idx), ("dev", dev_idx), ("test", test_idx)):
        c0, c1 = _class_counts(idxs)
        print(f"[probe] {name}: honest={c0}, collusive={c1}", flush=True)
        if c0 < args.floor_per_class or c1 < args.floor_per_class:
            print(f"[probe] WARNING: {name} split below floor {args.floor_per_class}: honest={c0}, collusive={c1}", flush=True)

    # Main probe train+eval
    X_tr, y_tr, sm_tr = flatten_per_agent(activations, labels, train_idx, layer_index, K)
    X_dv, y_dv, sm_dv = flatten_per_agent(activations, labels, dev_idx, layer_index, K)
    X_te, y_te, sm_te = flatten_per_agent(activations, labels, test_idx, layer_index, K)
    res = train_and_eval_probe(X_tr, y_tr, X_dv, y_dv, X_te, y_te, seed=args.seed)

    # Also produce per-agent scores over the full dataset (needed for M2 aggregation)
    sc = res["scaler"]
    clf = res["clf"]
    all_scores = np.zeros((N, K), dtype=np.float32)
    for i in range(N):
        for k in range(K):
            x = activations[i, k, layer_index, :].numpy().reshape(1, -1)
            xs = sc.transform(x)
            all_scores[i, k] = clf.decision_function(xs)[0]

    # Per-scenario probe score = mean over agents (single-agent baseline)
    per_scen_score_test = per_scenario_score(res["test_scores"], sm_te, N)
    y_test_scen = np.array([int(labels[i].item()) for i in test_idx])
    scen_scores_test = per_scen_score_test[test_idx]
    scen_auroc = roc_auc_score(y_test_scen, scen_scores_test) if len(set(y_test_scen.tolist())) == 2 else float("nan")
    scen_tpr_at_1 = tpr_at_fpr(y_test_scen, scen_scores_test, 0.01) if len(set(y_test_scen.tolist())) == 2 else float("nan")

    # Per-scenario dev AUROC (for best-layer selection in verdicts.py per the reviewer's fix #9)
    dev_scen_auroc = float("nan")
    if res.get("dev_scores") is not None and len(dev_idx) > 0:
        per_scen_score_dev = per_scenario_score(res["dev_scores"], sm_dv, N)
        y_dev_scen = np.array([int(labels[i].item()) for i in dev_idx])
        scen_scores_dev = per_scen_score_dev[dev_idx]
        if len(set(y_dev_scen.tolist())) == 2:
            dev_scen_auroc = float(roc_auc_score(y_dev_scen, scen_scores_dev))

    out = {
        "layer": args.layer,
        "layer_index": layer_index,
        "n_train_agents": int(len(y_tr)),
        "n_dev_agents": int(len(y_dv)),
        "n_test_agents": int(len(y_te)),
        "n_train_scenarios": int(len(train_idx)),
        "n_dev_scenarios": int(len(dev_idx)),
        "n_test_scenarios": int(len(test_idx)),
        "best_C": res["best_C"],
        "dev_auroc_per_agent": res["dev_auroc"],
        "dev_auroc_per_scenario_mean_pool": dev_scen_auroc,
        "test_auroc_per_agent": res["test_auroc"],
        "test_tpr_at_1pct_fpr_per_agent": res["test_tpr_at_1pct_fpr"],
        "test_auroc_per_scenario_mean_pool": float(scen_auroc),
        "test_tpr_at_1pct_fpr_per_scenario": float(scen_tpr_at_1),
        "sanity_checks": {},
        "train_idx": train_idx,
        "dev_idx": dev_idx,
        "test_idx": test_idx,
        "all_per_agent_scores": all_scores.tolist(),  # [N, K]
        "labels": labels.tolist(),
        "domains": domains,
        "scenario_ids": scenario_ids,
    }

    sanity_names = [x.strip() for x in args.sanity_checks.split(",")]
    if "label-permute" in sanity_names:
        out["sanity_checks"]["label_permute"] = sanity_label_permute(X_tr, y_tr, X_te, y_te, seed=args.seed)
    if "length-match" in sanity_names:
        out["sanity_checks"]["length_match"] = sanity_matched_length(activations, labels, domains, scenario_ids,
                                                                     train_idx + test_idx, layer_index, K,
                                                                     responses, seed=args.seed)
    if "topic-swap" in sanity_names:
        out["sanity_checks"]["topic_swap"] = sanity_topic_swap(activations, labels, domains,
                                                               train_idx + test_idx, layer_index, K, seed=args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[probe] wrote -> {args.out}", flush=True)
    print(f"[probe] test AUROC per-agent = {res['test_auroc']:.4f}, per-scenario = {scen_auroc:.4f}", flush=True)

    # Save the sklearn model as pickle for later reuse (frozen probe for M3 transfer)
    with open(args.out.replace(".json", ".pkl"), "wb") as f:
        pickle.dump({"scaler": sc, "clf": clf, "layer": args.layer, "layer_index": layer_index}, f)


if __name__ == "__main__":
    main()
