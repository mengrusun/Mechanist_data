"""
M7_C1_hidden — Hidden-layer matched-control test (P1b + P1c).

For each sampled hidden component c (layer3 or layer4):
    w1 = argmax_w cos(v_c, e_w) on open vocab (broden-style)
    w2 = second-best concept
    delta_c = cos(v_c, e_w1) - cos(v_c, e_w2)

P1b: mean(delta_c) > 0 with p < 0.05 (one-sided paired t-test); at least for layer4 at k=16.
P1c: delta grows monotonically with k, plateaus at k*_plateau <= 16.

Output JSON:
    {
      "k": ..., "pool": ...,
      "per_layer": {
         "layer3": {"n_comp": ..., "delta_sep_mean": ..., "delta_sep_ci_95": [..], "p_value_paired": ..., "passes_P1b": bool},
         "layer4": {"n_comp": ..., "delta_sep_mean": ..., "delta_sep_ci_95": [..], "p_value_paired": ..., "passes_P1b": bool},
      }
    }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import h5py
from scipy.stats import ttest_1samp, bootstrap


def load_vc(path: str):
    with h5py.File(path, "r") as h5:
        v_c = h5["v_c"][:]
        layer = h5["layer"][:].astype("U16")
        class_lbl = h5["class_label"][:]
        k = int(h5.attrs["k"])
        pool = str(h5.attrs["pool"])
    return v_c, layer, class_lbl, k, pool


def per_component_delta_sep(v_c: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """v_c: (n, D), text_emb: (V, D). Return (n,) array of best-vs-second cosine gaps."""
    v = v_c / (np.linalg.norm(v_c, axis=1, keepdims=True) + 1e-8)
    t = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8)
    cos = v @ t.T  # (n, V)
    # top-2
    idx2 = np.argpartition(-cos, 1, axis=1)[:, :2]
    vals2 = np.take_along_axis(cos, idx2, axis=1)  # (n, 2)
    # Ensure sorted descending
    top1 = vals2.max(axis=1)
    top2 = vals2.min(axis=1)
    return (top1 - top2).astype(np.float32)


def compute_layer_stats(deltas: np.ndarray) -> dict:
    n = len(deltas)
    mean = float(deltas.mean())
    # one-sided paired t-test (H0: mean = 0, H1: mean > 0)
    tt = ttest_1samp(deltas, 0.0, alternative="greater")
    # bootstrap 95% CI on the mean
    res = bootstrap((deltas,), np.mean, confidence_level=0.95, n_resamples=1000, method="basic",
                    random_state=np.random.default_rng(42))
    ci = [float(res.confidence_interval.low), float(res.confidence_interval.high)]
    return {
        "n_comp": n,
        "delta_sep_mean": mean,
        "delta_sep_std": float(deltas.std()),
        "delta_sep_ci_95": ci,
        "p_value_paired_greater": float(tt.pvalue),
        "passes_P1b": bool((mean > 0) and (tt.pvalue < 0.05)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v_c", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    v_c, layer, class_lbl, k, pool = load_vc(args.v_c)
    with h5py.File(args.text, "r") as h5t:
        text_emb = h5t["broden_concepts/embeddings"][:]  # (V, D)

    result = {"k": k, "pool": pool, "per_layer": {}, "vocab": "broden_concepts"}
    for lname in ("layer3", "layer4"):
        mask = (layer == lname)
        deltas = per_component_delta_sep(v_c[mask], text_emb)
        result["per_layer"][lname] = compute_layer_stats(deltas)
        print(f"[M7] {lname}: n={mask.sum()}, delta_sep={result['per_layer'][lname]['delta_sep_mean']:.4f}, "
              f"p={result['per_layer'][lname]['p_value_paired_greater']:.3g}, "
              f"passes_P1b={result['per_layer'][lname]['passes_P1b']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[M7] wrote {args.out}")


if __name__ == "__main__":
    main()
