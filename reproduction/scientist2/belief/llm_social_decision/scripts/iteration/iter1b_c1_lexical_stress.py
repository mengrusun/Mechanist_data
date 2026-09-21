#!/usr/bin/env python3
"""C1 lexical stress test — leave-one-phrasing-out (LOPO) cross-validation.

The reviewer's Suspicion 1 (still unresolved after iteration 1): shallow-layer
C1 probe=1.0 may reflect lexical/template leakage from paired-partner design
rather than genuine social-variable encoding. This test partitions the training
activations by `phrasing_id` (3 distinct prompt templates in DG-1000) and runs
LeaveOneGroupOut cross-validation:
  - LOPO: train probe on 2 phrasings; test on the held-out phrasing.
  - Random 5-fold (original picker): baseline for comparison.

If shallow layers pass random-5 but fail LOPO with a large drop, the shallow-layer
signal is phrasing-specific → C1 confounded by lexicon. If LOPO stays high, the
signal generalizes across surface forms → C1 is genuinely social-variable-related.

The reference layer is L=16 (M4-supp productive layer). We compare LOPO scores at
the original picked layer ell_V* to LOPO scores at deeper layers.
"""
import json
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
import warnings
warnings.filterwarnings("ignore")


def main():
    train_pack = torch.load("runs/M_main_v1/artifacts/m2/train_activations.pt",
                            map_location="cpu", weights_only=False)
    meta = train_pack["meta"]
    acts = train_pack["acts"]
    layer_ids = train_pack["layer_ids"]

    POS = {"G": "male", "A": "young", "I": "take-frame", "M": "no-meet"}

    phrasing = np.array([m["phrasing_id"] for m in meta])
    uniq = np.unique(phrasing)
    print(f"[c1-lopo] Unique phrasing_ids: {uniq.tolist()} ; counts: "
          f"{[int((phrasing==p).sum()) for p in uniq]}")

    results = {}
    for V in ["G", "A", "I", "M"]:
        pos_v = POS[V]
        y = np.array([1 if m[V] == pos_v else 0 for m in meta])
        results[V] = {}
        for mode_name, mask_fn in [
            ("baseline_only", lambda m: m["is_paired_partner_of"] is None),
            ("all_rows", lambda m: True),
        ]:
            mask = np.array([mask_fn(m) for m in meta])
            y_use = y[mask]
            phrasing_use = phrasing[mask]
            if len(np.unique(y_use)) < 2:
                continue
            per_layer = {}
            for li, ell in enumerate(layer_ids):
                X = acts[mask, li, :].float().numpy()
                logo = LeaveOneGroupOut()
                lopo_scores = cross_val_score(
                    LogisticRegression(max_iter=1000, C=1.0),
                    X, y_use,
                    cv=logo.split(X, y_use, groups=phrasing_use),
                    scoring="accuracy", n_jobs=1)
                k5_scores = cross_val_score(
                    LogisticRegression(max_iter=1000, C=1.0),
                    X, y_use, cv=5, scoring="accuracy", n_jobs=1)
                per_layer[ell] = {
                    "lopo_mean": float(lopo_scores.mean()),
                    "lopo_std": float(lopo_scores.std()),
                    "random5_mean": float(k5_scores.mean()),
                    "random5_std": float(k5_scores.std()),
                }
                print(f"[c1-lopo] V={V} mode={mode_name} ell={ell:>3} "
                      f"lopo={per_layer[ell]['lopo_mean']:.3f}±"
                      f"{per_layer[ell]['lopo_std']:.3f} "
                      f"rand5={per_layer[ell]['random5_mean']:.3f}",
                      flush=True)
            results[V][mode_name] = per_layer

    with open("runs/iteration_round_1/c1_lexical_stress_test.json", "w") as f:
        json.dump(results, f, indent=2)

    ORIG = {"G": 4, "A": 6, "I": 2, "M": 2}
    print()
    print("=" * 80)
    print("C1 LEXICAL STRESS TEST — LOPO vs random 5-fold (baseline_only mode)")
    print("=" * 80)
    print(f"{'V':<3} {'ell':>4} {'random5':>18} {'LOPO':>18} {'drop':>8}")
    for V in ["G", "A", "I", "M"]:
        for ell in [ORIG[V], 4, 8, 12, 16, 20, 24]:
            r = results[V]["baseline_only"].get(ell)
            if r is None:
                continue
            r5 = f"{r['random5_mean']:.3f}±{r['random5_std']:.3f}"
            lopo = f"{r['lopo_mean']:.3f}±{r['lopo_std']:.3f}"
            drop = r["random5_mean"] - r["lopo_mean"]
            marker = " <-- ORIG" if ell == ORIG[V] else (" (L=16)" if ell == 16 else "")
            print(f"{V:<3} {ell:>4} {r5:>18} {lopo:>18} {drop:>+8.3f}{marker}")
        print()


if __name__ == "__main__":
    main()
