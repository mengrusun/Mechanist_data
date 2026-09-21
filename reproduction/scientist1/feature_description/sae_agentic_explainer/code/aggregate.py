"""Aggregate per-feature results into per-layer and overall statistics."""
import argparse
import glob
import json
import os
import numpy as np
from scipy.stats import wilcoxon

from config import RESULTS_DIR


def load_all(subdir):
    root = os.path.join(RESULTS_DIR, subdir)
    files = sorted(glob.glob(os.path.join(root, "layer_*", "feature_*.json")))
    out = []
    for f in files:
        try:
            with open(f) as h:
                out.append(json.load(h))
        except Exception as e:
            print(f"Failed to load {f}: {e}")
    return out


def per_feature_metrics(res):
    ev = res.get("eval", {})
    row = {"feature": res.get("feature_idx"), "layer": res.get("layer")}
    for name in ("neuronpedia", "sage"):
        info = ev.get(name, {})
        gen = info.get("generative", {})
        pred = info.get("predictive", {})
        row[f"{name}_trigger"] = float(gen.get("trigger_rate", 0.0))
        row[f"{name}_mean_max"] = float(gen.get("mean_max", 0.0))
        row[f"{name}_pearson"] = float(pred.get("pearson", 0.0))
        row[f"{name}_spearman"] = float(pred.get("spearman", 0.0))
        row[f"{name}_expl"] = info.get("explanation", "")
    return row


def summarize(rows, label=""):
    if not rows:
        print(f"[{label}] No rows to summarize.")
        return {}
    def col(name):
        return [r[name] for r in rows]

    def mean(name):
        vals = col(name)
        return float(np.mean(vals)) if vals else 0.0

    def se(name):
        vals = np.array(col(name), dtype=float)
        return float(vals.std(ddof=1) / max(1, np.sqrt(len(vals)))) if len(vals) > 1 else 0.0

    metrics = {}
    for m in ("trigger", "pearson", "spearman", "mean_max"):
        metrics[f"neuronpedia_{m}"] = mean(f"neuronpedia_{m}")
        metrics[f"neuronpedia_{m}_se"] = se(f"neuronpedia_{m}")
        metrics[f"sage_{m}"] = mean(f"sage_{m}")
        metrics[f"sage_{m}_se"] = se(f"sage_{m}")
        metrics[f"delta_{m}"] = metrics[f"sage_{m}"] - metrics[f"neuronpedia_{m}"]

        # paired Wilcoxon signed-rank
        n = col(f"neuronpedia_{m}")
        s = col(f"sage_{m}")
        diffs = [si - ni for si, ni in zip(s, n)]
        metrics[f"delta_{m}_se"] = float(np.std(diffs, ddof=1) / max(1, np.sqrt(len(diffs)))) if len(diffs) > 1 else 0.0
        try:
            if any(d != 0 for d in diffs):
                w = wilcoxon(diffs, zero_method="pratt", alternative="greater")
                metrics[f"wilcoxon_p_{m}"] = float(w.pvalue)
            else:
                metrics[f"wilcoxon_p_{m}"] = 1.0
        except Exception:
            metrics[f"wilcoxon_p_{m}"] = None

    # win rate: SAGE >= Neuronpedia
    wins_trigger = sum(1 for r in rows if r["sage_trigger"] > r["neuronpedia_trigger"])
    losses_trigger = sum(1 for r in rows if r["sage_trigger"] < r["neuronpedia_trigger"])
    ties_trigger = sum(1 for r in rows if r["sage_trigger"] == r["neuronpedia_trigger"])
    wins_pear = sum(1 for r in rows if r["sage_pearson"] > r["neuronpedia_pearson"])
    wins_spear = sum(1 for r in rows if r["sage_spearman"] > r["neuronpedia_spearman"])
    metrics["n"] = len(rows)
    metrics["sage_win_trigger"] = wins_trigger
    metrics["sage_loss_trigger"] = losses_trigger
    metrics["tie_trigger"] = ties_trigger
    metrics["sage_win_pearson"] = wins_pear
    metrics["sage_win_spearman"] = wins_spear
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subdir", default="gemma-2-2b_main")
    args = p.parse_args()

    all_res = load_all(args.subdir)
    print(f"Loaded {len(all_res)} feature result files from {args.subdir}")
    rows = [per_feature_metrics(r) for r in all_res]

    per_layer = {}
    for r in rows:
        per_layer.setdefault(r["layer"], []).append(r)

    print("\n=== Per-layer summary ===")
    for layer in sorted(per_layer.keys()):
        print(f"\n--- Layer {layer} (n={len(per_layer[layer])}) ---")
        m = summarize(per_layer[layer], label=f"L{layer}")
        for k in ("neuronpedia_trigger", "sage_trigger", "delta_trigger", "wilcoxon_p_trigger",
                  "sage_win_trigger", "tie_trigger",
                  "neuronpedia_pearson", "sage_pearson", "delta_pearson", "wilcoxon_p_pearson",
                  "neuronpedia_spearman", "sage_spearman", "delta_spearman", "wilcoxon_p_spearman",
                  "sage_win_pearson", "sage_win_spearman"):
            v = m.get(k)
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")

    print("\n=== Overall summary ===")
    m = summarize(rows, label="ALL")
    for k in ("neuronpedia_trigger", "sage_trigger", "delta_trigger", "wilcoxon_p_trigger",
              "sage_win_trigger", "tie_trigger",
              "neuronpedia_pearson", "sage_pearson", "delta_pearson", "wilcoxon_p_pearson",
              "neuronpedia_spearman", "sage_spearman", "delta_spearman", "wilcoxon_p_spearman",
              "sage_win_pearson", "sage_win_spearman"):
        v = m.get(k)
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save summary JSON
    summary = {
        "overall": summarize(rows),
        "per_layer": {str(l): summarize(per_layer[l]) for l in per_layer},
        "per_feature": rows,
    }
    with open(os.path.join(RESULTS_DIR, f"{args.subdir}__summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary to {RESULTS_DIR}/{args.subdir}__summary.json")


if __name__ == "__main__":
    main()
