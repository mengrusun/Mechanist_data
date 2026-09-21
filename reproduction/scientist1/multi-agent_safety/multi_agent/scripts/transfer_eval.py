"""Zero-shot transfer: train per-agent probes on Core, test on Transfer + Stego.

For each layer:
  1. Fit LR probe on ALL Core per-agent activations (label = colluding dialogue).
  2. Apply probe to Transfer / Stego per-agent activations, aggregate per dialogue,
     report AUROC/accuracy overall and per-family.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.preprocessing import StandardScaler


def load_dir(act_dir: Path):
    entries = {}
    for f in sorted(act_dir.glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        acts = d["activations"]
        agent_ids = d["agent_ids"]
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
            "activations": per_agent,
            "family": str(d["family"]),
            "colluding": int(d["colluding"]),
        }
    return entries


def build_arrays(entries, layer_idx):
    Xs, ys, ds, aids, fams = [], [], [], [], []
    for i, (sid, e) in enumerate(entries.items()):
        for a in range(e["activations"].shape[0]):
            Xs.append(e["activations"][a, layer_idx])
            ys.append(e["colluding"])
            ds.append(i)
            aids.append(a)
            fams.append(e["family"])
    return np.stack(Xs), np.array(ys), np.array(ds), np.array(aids), np.array(fams)


AGG_METHODS = ["mean", "max", "min", "median", "soft_or", "topk2_mean", "std", "range"]


def aggregate(probs, m):
    if m == "mean": return float(np.mean(probs))
    if m == "max": return float(np.max(probs))
    if m == "min": return float(np.min(probs))
    if m == "median": return float(np.median(probs))
    if m == "soft_or": return 1.0 - float(np.prod(1.0 - probs))
    if m == "topk2_mean": return float(np.mean(np.sort(probs)[::-1][:2]))
    if m == "std": return float(np.mean(probs) + np.std(probs))
    if m == "range": return float(np.max(probs) - np.min(probs))
    raise ValueError(m)


def eval_transfer(train_entries, test_entries, layer_idx, C=1.0, standardize=True):
    Xtr, ytr, dtr, atr, ftr = build_arrays(train_entries, layer_idx)
    Xte, yte, dte, ate, fte = build_arrays(test_entries, layer_idx)
    scaler = None
    if standardize:
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
    else:
        Xtr_s = Xtr
        Xte_s = Xte
    clf = LogisticRegression(max_iter=3000, C=C, solver="liblinear")
    clf.fit(Xtr_s, ytr)
    probs = clf.predict_proba(Xte_s)[:, 1]
    # Aggregate per dialogue
    dialogues = sorted(set(dte.tolist()))
    labels = []
    per_scenario_probs = []
    fams_by_dialogue = []
    for did in dialogues:
        mask = dte == did
        ap = probs[mask]
        labels.append(yte[mask][0])
        fams_by_dialogue.append(fte[mask][0])
        d_scores = {m: aggregate(ap, m) for m in AGG_METHODS}
        per_scenario_probs.append(d_scores)
    labels = np.array(labels)
    fams_by_dialogue = np.array(fams_by_dialogue)

    overall = {}
    for m in AGG_METHODS:
        pv = np.array([x[m] for x in per_scenario_probs])
        overall[m] = {
            "auroc": float(roc_auc_score(labels, pv)),
            "auprc": float(average_precision_score(labels, pv)),
            "acc": float(accuracy_score(labels, pv > 0.5)),
        }
    # Per-family results
    per_fam = {}
    for fam in sorted(set(fams_by_dialogue.tolist())):
        mask = fams_by_dialogue == fam
        if mask.sum() < 4 or len(set(labels[mask].tolist())) < 2:
            continue
        per_fam[fam] = {}
        for m in AGG_METHODS:
            pv = np.array([x[m] for x in per_scenario_probs])[mask]
            per_fam[fam][m] = {
                "auroc": float(roc_auc_score(labels[mask], pv)),
                "auprc": float(average_precision_score(labels[mask], pv)),
                "acc": float(accuracy_score(labels[mask], pv > 0.5)),
                "n": int(mask.sum()),
            }
    return {"layer_idx": layer_idx, "overall": overall, "per_family": per_fam}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", type=str, required=True)
    p.add_argument("--test-dir", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--C", type=float, default=1.0)
    args = p.parse_args()

    train = load_dir(Path(args.train_dir))
    test = load_dir(Path(args.test_dir))
    print(f"train n={len(train)}, test n={len(test)}")
    any_e = next(iter(train.values()))
    n_layers = any_e["activations"].shape[1]

    all_res = []
    for li in range(n_layers):
        r = eval_transfer(train, test, li, C=args.C)
        all_res.append(r)
        best = max(r["overall"].items(), key=lambda x: x[1]["auroc"])
        print(f"L{li}: best_agg={best[0]} auroc={best[1]['auroc']:.3f} | " +
              " ".join([f"{m}={r['overall'][m]['auroc']:.3f}" for m in AGG_METHODS]))
        for fam, fd in r["per_family"].items():
            best_fam = max(fd.items(), key=lambda x: x[1]["auroc"])
            print(f"   {fam} n={best_fam[1]['n']} best_agg={best_fam[0]} auroc={best_fam[1]['auroc']:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_res, f, indent=2)


if __name__ == "__main__":
    main()
