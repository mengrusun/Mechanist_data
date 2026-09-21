"""
Step 3: Train linear probes and measure geometry.

For each layer L and each probing position (answer_end, conf_prefix):
    Probe A (accuracy):   sklearn LogisticRegression on hidden -> P(correct)
    Probe C (confidence): sklearn Ridge on hidden -> verbalized confidence
                          plus a binary variant (conf >= 100 vs <100).
Also compute Difference-of-Means direction, which is more robust:
    w_A = mean(hidden|correct=1) - mean(hidden|correct=0)
    w_C = mean(hidden|conf>=100) - mean(hidden|conf<100)

Outputs:
  results/probe_results.json — per-layer metrics and cosine similarities
  results/directions.npz     — probe direction vectors per layer
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, brier_score_loss
from sklearn.model_selection import train_test_split


def unit(v, axis=-1, eps=1e-9):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (n + eps)


def cos(a, b):
    a = a.reshape(-1)
    b = b.reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def train_and_score(H_train, y_train, H_test, y_test, mode, C=1.0, alpha=1.0):
    """Return (probe_direction, metrics)."""
    if mode == "logreg":
        clf = LogisticRegression(C=C, max_iter=2000, n_jobs=1)
        clf.fit(H_train, y_train)
        w = clf.coef_.reshape(-1)  # (D,)
        proba = clf.predict_proba(H_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return w, {
            "auc": float(roc_auc_score(y_test, proba)),
            "acc": float(accuracy_score(y_test, pred)),
            "brier": float(brier_score_loss(y_test, proba)),
        }
    elif mode == "ridge":
        clf = Ridge(alpha=alpha)
        clf.fit(H_train, y_train)
        w = clf.coef_.reshape(-1)
        pred = clf.predict(H_test)
        return w, {
            "r2": float(r2_score(y_test, pred)),
            "pearson": float(np.corrcoef(pred, y_test)[0, 1]) if np.std(pred) > 0 else 0.0,
            "mae": float(np.mean(np.abs(pred - y_test))),
        }
    else:
        raise ValueError(mode)


def diff_of_means(H, mask_pos, mask_neg):
    if mask_pos.sum() == 0 or mask_neg.sum() == 0:
        return np.zeros(H.shape[1])
    return H[mask_pos].mean(0) - H[mask_neg].mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_dirs_npz", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.3)
    ap.add_argument("--position", default="answer_end",
                    choices=["answer_end", "conf_prefix", "both"])
    args = ap.parse_args()

    data = np.load(args.npz)
    correct = data["correct"].astype(int)
    confidence = data["confidence"].astype(np.float32)
    N = correct.shape[0]
    print(f"N={N}, accuracy={correct.mean():.3f}, conf mean={confidence.mean():.2f}")
    print(f"Confidence buckets: <50={np.mean(confidence<50):.3f}, [50,100)={np.mean((confidence>=50)&(confidence<100)):.3f}, ==100={np.mean(confidence==100):.3f}")

    positions = ["answer_end", "conf_prefix"] if args.position == "both" else [args.position]

    # Fixed split for all layers
    idx_train, idx_test = train_test_split(
        np.arange(N), test_size=args.test_frac, random_state=args.seed, stratify=correct
    )

    is_high_conf = (confidence >= 100).astype(int)
    conf_norm = (confidence / 100.0).astype(np.float32)  # in [0,1]

    results = {"per_position": {}}
    directions = {}

    for pos in positions:
        H_all = data[f"hs_{pos}"].astype(np.float32)  # (N, L+1, D)
        L1 = H_all.shape[1]
        D = H_all.shape[2]
        print(f"\n=== position={pos}, L+1={L1}, D={D} ===")

        per_layer = []
        w_A_lr, w_C_lr, w_C_hi_lr = [], [], []
        w_A_dm, w_C_dm = [], []

        for layer in range(L1):
            H = H_all[:, layer, :]
            H_train, H_test = H[idx_train], H[idx_test]

            # ------- Probe A: correctness ---------
            wA, mA = train_and_score(H_train, correct[idx_train], H_test, correct[idx_test], "logreg")

            # ------- Probe C: verbalized confidence (regression) ---------
            wC_reg, mC_reg = train_and_score(
                H_train, conf_norm[idx_train], H_test, conf_norm[idx_test], "ridge"
            )

            # ------- Probe C_hi: binary is-conf-100 ---------
            wC_hi, mC_hi = train_and_score(
                H_train, is_high_conf[idx_train], H_test, is_high_conf[idx_test], "logreg"
            )

            # ------- Difference of means directions ---------
            wA_dm = diff_of_means(H_train, correct[idx_train] == 1, correct[idx_train] == 0)
            wC_dm = diff_of_means(H_train, is_high_conf[idx_train] == 1, is_high_conf[idx_train] == 0)

            # Cosines
            cos_lr_reg = cos(wA, wC_reg)
            cos_lr_hi = cos(wA, wC_hi)
            cos_dm = cos(wA_dm, wC_dm)

            per_layer.append({
                "layer": layer,
                "probeA": mA,
                "probeC_reg": mC_reg,
                "probeC_hi": mC_hi,
                "cos_logreg_reg": cos_lr_reg,
                "cos_logreg_hi": cos_lr_hi,
                "cos_diffmeans": cos_dm,
            })
            w_A_lr.append(wA); w_C_lr.append(wC_reg); w_C_hi_lr.append(wC_hi)
            w_A_dm.append(wA_dm); w_C_dm.append(wC_dm)

            if layer % 4 == 0 or layer == L1 - 1:
                print(f"  L{layer:2d}  A_AUC={mA['auc']:.3f}  C_R2={mC_reg['r2']:+.3f}  "
                      f"Chi_AUC={mC_hi['auc']:.3f}  cos_LR_reg={cos_lr_reg:+.3f}  "
                      f"cos_LR_hi={cos_lr_hi:+.3f}  cos_DM={cos_dm:+.3f}")

        results["per_position"][pos] = per_layer
        directions[f"{pos}_wA_logreg"] = np.stack(w_A_lr)
        directions[f"{pos}_wC_ridge"] = np.stack(w_C_lr)
        directions[f"{pos}_wC_hi_logreg"] = np.stack(w_C_hi_lr)
        directions[f"{pos}_wA_diffmeans"] = np.stack(w_A_dm)
        directions[f"{pos}_wC_diffmeans"] = np.stack(w_C_dm)

    # Additionally, compare (verbalized confidence -> correctness) baseline calibration.
    # ECE with 10 bins:
    def ece(y, p, bins=10):
        edges = np.linspace(0, 1, bins + 1)
        e = 0.0
        for i in range(bins):
            m = (p > edges[i]) & (p <= edges[i + 1])
            if m.sum() == 0: continue
            e += m.mean() * abs(y[m].mean() - p[m].mean())
        return float(e)

    verb_p = confidence / 100.0
    verb_ece = ece(correct.astype(float), verb_p, 10)
    verb_brier = float(np.mean((verb_p - correct) ** 2))
    verb_auc = float(roc_auc_score(correct, verb_p)) if len(set(verb_p)) > 1 else float("nan")

    results["verbalized_calibration"] = {
        "ece": verb_ece, "brier": verb_brier, "auc_vs_correct": verb_auc,
        "n": int(N), "accuracy": float(correct.mean()),
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(args.out_dirs_npz, **directions)
    print("\nSaved:", args.out_json, "and", args.out_dirs_npz)

    # -------- summary printout --------
    for pos in positions:
        pl = results["per_position"][pos]
        aucs = [d["probeA"]["auc"] for d in pl]
        cregs = [d["probeC_reg"]["r2"] for d in pl]
        chis = [d["probeC_hi"]["auc"] for d in pl]
        cos_lrs = [d["cos_logreg_reg"] for d in pl]
        cos_lrh = [d["cos_logreg_hi"] for d in pl]
        cos_dms = [d["cos_diffmeans"] for d in pl]
        best_A_layer = int(np.argmax(aucs))
        best_C_layer = int(np.argmax(chis))
        print(f"\n[{pos}] Best correctness probe: L{best_A_layer} AUC={aucs[best_A_layer]:.3f}")
        print(f"[{pos}] Best conf-hi probe:      L{best_C_layer} AUC={chis[best_C_layer]:.3f}")
        print(f"[{pos}] Mean |cos| across mid layers (10..25): "
              f"LR_reg={np.mean(np.abs(cos_lrs[10:26])):.3f} "
              f"LR_hi={np.mean(np.abs(cos_lrh[10:26])):.3f} "
              f"DM={np.mean(np.abs(cos_dms[10:26])):.3f}")
    print(f"\nVerbalized calibration: ECE={verb_ece:.3f} Brier={verb_brier:.3f} AUC={verb_auc:.3f}")


if __name__ == "__main__":
    main()
