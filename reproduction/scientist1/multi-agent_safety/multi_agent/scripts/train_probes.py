"""Train linear probes on per-agent activations and evaluate aggregation methods.

Claims to test:
  1. Per-agent activation probes recover a collusion signal.
  2. Group-aggregated probes outperform any single-agent probe; no single aggregation
     dominates.
  3. (Handled in transfer_eval.py) Probes transfer zero-shot.

For each dialogue, we have per-agent activations after aggregating over both turns for
each speaker (mean of turn-0 + turn-5 of the same speaker).

We train per-agent logistic-regression probes (one probe per layer) using group k-fold
cross-validation to avoid contamination. Aggregation methods evaluated at the dialogue
level:
  - mean, max, min, soft-OR (1 - prod(1-p)), noisy-OR w/ threshold, top-k mean
  - concat-then-LR (train a probe on concatenated per-agent activations)
  - learned-pool via a small MLP over per-agent probe scores
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


def load_split(act_dir: Path):
    """Load all .npz files in act_dir. Returns dict of {scenario_id: entry}."""
    entries = {}
    for f in sorted(act_dir.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        acts = d["activations"]  # [n_turns, n_layers, hidden]
        agent_ids = d["agent_ids"]
        # Aggregate per-agent by averaging turns of the same agent
        n_agents = int(agent_ids.max()) + 1
        n_layers = acts.shape[1]
        hidden = acts.shape[2]
        per_agent = np.zeros((n_agents, n_layers, hidden), dtype=np.float32)
        counts = np.zeros(n_agents, dtype=np.int32)
        for ti, aid in enumerate(agent_ids):
            per_agent[aid] += acts[ti].astype(np.float32)
            counts[aid] += 1
        per_agent /= np.maximum(counts, 1)[:, None, None]
        entries[str(d["scenario_id"])] = {
            "activations": per_agent,   # [n_agents, n_layers, hidden]
            "family": str(d["family"]),
            "colluding": int(d["colluding"]),
        }
    return entries


def build_arrays(entries: dict, layer_idx: int):
    """Return per-agent feature matrix X, dialogue label y, dialogue-id array, agent-id array."""
    Xs, ys, ds, aids, families = [], [], [], [], []
    for i, (sid, e) in enumerate(entries.items()):
        n_agents = e["activations"].shape[0]
        for a in range(n_agents):
            Xs.append(e["activations"][a, layer_idx])
            ys.append(e["colluding"])
            ds.append(i)
            aids.append(a)
            families.append(e["family"])
    return np.stack(Xs), np.array(ys), np.array(ds), np.array(aids), np.array(families)


def cv_predict_per_agent(X, y_agent, dialog_ids, n_splits=5, C=1.0, seed=0):
    """5-fold group-CV logistic regression. Returns per-agent probabilities."""
    gkf = GroupKFold(n_splits=n_splits)
    probs = np.zeros(len(y_agent), dtype=np.float32)
    for fold_i, (tr, te) in enumerate(gkf.split(X, y_agent, groups=dialog_ids)):
        clf = LogisticRegression(max_iter=2000, C=C, n_jobs=1, solver="liblinear")
        clf.fit(X[tr], y_agent[tr])
        probs[te] = clf.predict_proba(X[te])[:, 1]
    return probs


def cv_predict_dialogue(X_diag, y_diag, n_splits=5, C=1.0):
    """LR trained on dialogue-level concatenated features."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    probs = np.zeros(len(y_diag), dtype=np.float32)
    for tr, te in kf.split(X_diag):
        clf = LogisticRegression(max_iter=2000, C=C, solver="liblinear")
        clf.fit(X_diag[tr], y_diag[tr])
        probs[te] = clf.predict_proba(X_diag[te])[:, 1]
    return probs


def aggregate(agent_probs, method):
    """agent_probs: [n_agents] probabilities. Returns dialogue-level score."""
    if method == "mean":
        return float(np.mean(agent_probs))
    if method == "max":
        return float(np.max(agent_probs))
    if method == "min":
        return float(np.min(agent_probs))
    if method == "median":
        return float(np.median(agent_probs))
    if method == "soft_or":
        return 1.0 - float(np.prod(1.0 - agent_probs))
    if method == "topk2_mean":
        s = np.sort(agent_probs)[::-1]
        return float(np.mean(s[:2]))
    if method == "std":
        # score = mean + std (aggressive on disagreement)
        return float(np.mean(agent_probs) + np.std(agent_probs))
    if method == "range":
        return float(np.max(agent_probs) - np.min(agent_probs))
    raise ValueError(method)


AGG_METHODS = ["mean", "max", "min", "median", "soft_or", "topk2_mean", "std", "range"]


def evaluate_layer(entries, layer_idx: int, C: float = 1.0):
    """Per-layer evaluation. Returns dict with metrics."""
    X, y_a, d_ids, a_ids, families = build_arrays(entries, layer_idx)
    # Per-agent CV probabilities (probe trained on all agents lumped)
    probs = cv_predict_per_agent(X, y_a, d_ids, n_splits=5, C=C)

    # Per-agent probe metric (treat every agent-observation independently)
    per_agent_auc = roc_auc_score(y_a, probs)
    per_agent_acc = accuracy_score(y_a, probs > 0.5)

    # Per-agent probes trained per-slot (agent 0..4) - test whether one agent's slot is uniquely informative
    n_agents = int(a_ids.max()) + 1
    per_slot_auc = {}
    for a in range(n_agents):
        mask_a = a_ids == a
        Xa = X[mask_a]
        ya = y_a[mask_a]
        da = d_ids[mask_a]
        # simple 5-fold CV
        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        p = np.zeros(len(ya), dtype=np.float32)
        for tr, te in kf.split(Xa):
            clf = LogisticRegression(max_iter=2000, C=C, solver="liblinear")
            clf.fit(Xa[tr], ya[tr])
            p[te] = clf.predict_proba(Xa[te])[:, 1]
        per_slot_auc[a] = float(roc_auc_score(ya, p))

    # Dialogue-level aggregation of per-agent CV probs
    dialogues = sorted(set(d_ids.tolist()))
    diag_labels = []
    diag_probs_by_method = {m: [] for m in AGG_METHODS}
    for did in dialogues:
        mask = d_ids == did
        ap = probs[mask]
        diag_labels.append(y_a[mask][0])
        for m in AGG_METHODS:
            diag_probs_by_method[m].append(aggregate(ap, m))
    diag_labels = np.array(diag_labels)
    agg_results = {}
    for m in AGG_METHODS:
        pv = np.array(diag_probs_by_method[m])
        agg_results[m] = {
            "auroc": float(roc_auc_score(diag_labels, pv)),
            "auprc": float(average_precision_score(diag_labels, pv)),
            "acc": float(accuracy_score(diag_labels, pv > 0.5)),
        }

    # Concat-then-LR baseline: concat per-agent activations to a single dialogue vector,
    # train a dialogue-level probe.
    entries_list = list(entries.values())
    Xd = []
    yd = []
    for e in entries_list:
        v = e["activations"][:, layer_idx].reshape(-1)  # [n_agents * hidden]
        Xd.append(v)
        yd.append(e["colluding"])
    Xd = np.stack(Xd)
    yd = np.array(yd)
    diag_probs_concat = cv_predict_dialogue(Xd, yd, n_splits=5, C=C)
    concat_result = {
        "auroc": float(roc_auc_score(yd, diag_probs_concat)),
        "auprc": float(average_precision_score(yd, diag_probs_concat)),
        "acc": float(accuracy_score(yd, diag_probs_concat > 0.5)),
    }

    return {
        "layer_idx": layer_idx,
        "per_agent_auc_all": per_agent_auc,
        "per_agent_acc_all": per_agent_acc,
        "per_slot_auc": per_slot_auc,
        "single_agent_best_auc": float(max(per_slot_auc.values())),
        "single_agent_worst_auc": float(min(per_slot_auc.values())),
        "aggregation": agg_results,
        "concat_probe": concat_result,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--activations-dir", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--C", type=float, default=1.0)
    args = p.parse_args()

    entries = load_split(Path(args.activations_dir))
    print(f"loaded {len(entries)} dialogues")
    # Determine layer count
    any_e = next(iter(entries.values()))
    n_layers = any_e["activations"].shape[1]
    print(f"n_layers = {n_layers}")

    all_results = []
    for li in range(n_layers):
        r = evaluate_layer(entries, li, C=args.C)
        all_results.append(r)
        agg_line = " ".join([f"{m}={r['aggregation'][m]['auroc']:.3f}" for m in AGG_METHODS])
        print(f"L{li}: per_agent_auc={r['per_agent_auc_all']:.3f} "
              f"single_best={r['single_agent_best_auc']:.3f} "
              f"concat={r['concat_probe']['auroc']:.3f} | {agg_line}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
