"""Step 4 — probe fits, per-layer sweep, bootstrap CIs, cosine trajectory.

Covers B1 (probes) + B2 (cosine) + partial B6 (nulls) main runs. Steps:

1. Load hidden states H_turn1[N, L+1, D], H_turn2[N, L+1, D] (float16).
2. Build split masks (train / dev / test) via split_manifest.json.
3. Fit per-layer probes:
   - probe_c_binary : L2-logistic on H_1^L predicting y_correct (from turn1 rows).
     C swept in {0.001, 0.01, 0.1, 1, 10} tuned on dev; report at best C per layer.
   - probe_v_binary : L2-logistic on H_2^L predicting c binarized at
                      threshold from stage15.
   - probe_v_continuous : L2-linear regression on H_2^L predicting c (used if
                          Stage-1.5 selected the continuous path).
   - probe_v_ordinal :   4-bin ordinal probe (used if ordinal path). We fit as
                         one-vs-rest logistic (approx) since sklearn ordinal is
                         non-standard.
4. Compute per-layer AUROC / Spearman rho / ECE (post-isotonic).
5. Choose L* by mean of normalized binary AUROCs.
6. Compute per-layer |cos(v_c^L, v_v^L)|.
7. Retrain-on-bootstrap 1000 resamples at L* only for CI on AUROC + |cos|.
8. Extract v_c^{L*}, v_v^{L*} (L2-normalized weight vectors) and save.
9. Nulls: random-direction probe (draw random unit vector from N(0, I / D)) —
   analytic; shuffled-label probe — refit at L*.

Also computes token-probability baseline for C1.

Outputs (JSON + pickle):
  - artifacts/probe_metrics.json
  - artifacts/cos_trajectory.json
  - artifacts/cos_bootstrap.json
  - artifacts/probes.npz  (v_c_star, v_v_star, per-layer weights, L*)
  - artifacts/nulls.json
  - artifacts/isotonic_c.pkl (calibrator)
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as sstats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils as U


def _load_split_masks(qids: np.ndarray, split: dict, idxs: np.ndarray = None):
    """Return row-index arrays for train, dev, test in the current npz.

    Prefers idx-based splits (unique per row) over qid-based splits (TriviaQA has
    duplicate qids -> qid-based splits leak between train/dev/test).
    """
    if idxs is not None and "train_idxs" in split:
        train_idxs = set(int(x) for x in split["train_idxs"])
        dev_idxs = set(int(x) for x in split["dev_idxs"])
        test_idxs = set(int(x) for x in split["test_idxs"])
        idxs_list = [int(x) for x in idxs.tolist()]
        tr = np.array([i for i, ix in enumerate(idxs_list) if ix in train_idxs], dtype=np.int64)
        dv = np.array([i for i, ix in enumerate(idxs_list) if ix in dev_idxs], dtype=np.int64)
        te = np.array([i for i, ix in enumerate(idxs_list) if ix in test_idxs], dtype=np.int64)
        return tr, dv, te
    # Fallback: qid-based (WARNING: leaks with duplicate qids)
    train_ids = set(split["train_ids"])
    dev_ids = set(split["dev_ids"])
    test_ids = set(split["test_ids"])
    tr = np.array([i for i, q in enumerate(qids.tolist()) if q in train_ids], dtype=np.int64)
    dv = np.array([i for i, q in enumerate(qids.tolist()) if q in dev_ids], dtype=np.int64)
    te = np.array([i for i, q in enumerate(qids.tolist()) if q in test_ids], dtype=np.int64)
    return tr, dv, te


def _fit_lr_probe(X_tr, y_tr, X_dv, y_dv, Cs=(0.001, 0.01, 0.1, 1.0, 10.0), seed=42, class_weight=None):
    """Fit L2-logistic with C tuned by dev AUROC. Returns (best_C, model, dev_auc).

    Uses lbfgs solver which is 3-8x faster than liblinear for dense high-D features."""
    best = (None, None, -1.0)
    for C in Cs:
        m = LogisticRegression(
            C=C, penalty="l2", solver="lbfgs", max_iter=500,
            random_state=seed, class_weight=class_weight, n_jobs=-1,
        )
        m.fit(X_tr, y_tr)
        if len(np.unique(y_dv)) < 2:
            continue
        p = m.predict_proba(X_dv)[:, 1]
        auc = roc_auc_score(y_dv, p)
        if auc > best[2]:
            best = (C, m, auc)
    if best[1] is None:
        best = (1.0, LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                                        max_iter=500, random_state=seed, n_jobs=-1).fit(X_tr, y_tr), 0.5)
    return best


def _fit_ridge_probe(X_tr, y_tr, X_dv, y_dv, alphas=(0.01, 0.1, 1.0, 10.0, 100.0), seed=42):
    best = (None, None, -np.inf)
    for a in alphas:
        m = Ridge(alpha=a, random_state=seed)
        m.fit(X_tr, y_tr)
        p = m.predict(X_dv)
        rho, _ = sstats.spearmanr(p, y_dv)
        rho = 0.0 if np.isnan(rho) else float(rho)
        if rho > best[2]:
            best = (a, m, rho)
    return best


def _ece(probs, labels, n_bins=15):
    probs = np.asarray(probs)
    labels = np.asarray(labels).astype(np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if not mask.any():
            continue
        acc = labels[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _isotonic_calibrate(train_probs, train_labels):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(train_probs, train_labels)
    return iso


def _cos(u, v):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def _unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bootstrap", type=int, default=1000, help="bootstrap resamples for CI at L*")
    ap.add_argument("--n-bootstrap-percentile", type=float, default=95.0,
                    help="CI percentile")
    ap.add_argument("--layers", type=str, default="all", help="'all' or comma list")
    ap.add_argument("--seed", type=int, default=U.DEFAULT_SEED)
    ap.add_argument("--n-random-null", type=int, default=100, help="random-direction samples")
    args = ap.parse_args()

    U.set_all_seeds(args.seed)

    # Load hidden states + labels
    print("[step4] Loading H_turn1, H_turn2 …", flush=True)
    npz1 = np.load(U.ARTIFACT_DIR / "H_turn1.npz")
    H1 = npz1["H"]  # (N, L+1, D) fp16
    qids1 = npz1["question_ids"]
    idxs1 = npz1["idxs"]
    npz2 = np.load(U.ARTIFACT_DIR / "H_turn2.npz")
    H2 = npz2["H"]
    qids2 = npz2["question_ids"]
    idxs2 = npz2["idxs"]
    print(f"[step4] H_turn1 {H1.shape}  H_turn2 {H2.shape}", flush=True)

    # Load labels
    turn1_rows = U.load_jsonl(U.ARTIFACT_DIR / "turn1_gen.jsonl")
    turn2_rows = U.load_jsonl(U.ARTIFACT_DIR / "turn2_gen.jsonl")
    y_correct_by_qid = {r["question_id"]: int(r["y_correct"]) for r in turn1_rows}
    c_by_qid = {r["question_id"]: (float(r["c"]) if r["c_parseable"] else None) for r in turn2_rows}

    # Align H1 and H2 by qid (they *should* have same order, but be defensive)
    assert list(qids1) == list(qids2), "H_turn1 and H_turn2 qid order differ"
    N, Lp1, D = H1.shape
    num_layers = Lp1 - 1
    print(f"[step4] N={N} num_layers={num_layers} D={D}", flush=True)

    # Split masks (prefer idx-based to avoid TriviaQA duplicate-qid leakage)
    split = U.load_json(U.ARTIFACT_DIR / "split_manifest.json")
    tr_idx, dv_idx, te_idx = _load_split_masks(qids1, split, idxs=idxs1)
    print(f"[step4] train={len(tr_idx)} dev={len(dv_idx)} test={len(te_idx)}", flush=True)

    # Labels — qids may be strings (e.g. "qb_1175") or ints; keep native type from dicts.
    def _qkey(q):
        # Try both str and int to be robust to save/load type coercion.
        if q in y_correct_by_qid or q in c_by_qid:
            return q
        try:
            iq = int(q)
            if iq in y_correct_by_qid or iq in c_by_qid:
                return iq
        except (TypeError, ValueError):
            pass
        sq = str(q)
        return sq
    y_c = np.array([y_correct_by_qid.get(_qkey(q), 0) for q in qids1.tolist()], dtype=np.int64)
    c_raw = np.array([c_by_qid.get(_qkey(q), np.nan) if c_by_qid.get(_qkey(q), np.nan) is not None else np.nan for q in qids2.tolist()], dtype=np.float64)
    c_parseable = ~np.isnan(c_raw)

    # Stage 1.5 threshold for binarizing c
    stage15 = U.load_json(U.ARTIFACT_DIR / "stage15_variance.json")
    c_thresh = float(stage15["binarize_threshold"])
    path = stage15["path"]
    print(f"[step4] Stage-1.5 path={path} binarize_at={stage15['binarize_at']} thresh={c_thresh}", flush=True)

    # Ordinal buckets (only used if ordinal path)
    ord_bins = [0, 25, 50, 75, 101]  # rightmost open
    def to_ordinal(c):
        return np.digitize(c, [25, 50, 75])  # 0..3

    # Layer sweep
    if args.layers == "all":
        layers = list(range(num_layers + 1))
    else:
        layers = [int(x) for x in args.layers.split(",")]

    metrics_per_layer = {}
    v_c_by_layer = np.zeros((num_layers + 1, D), dtype=np.float32)
    v_v_bin_by_layer = np.zeros((num_layers + 1, D), dtype=np.float32)

    t0 = time.time()
    for L in layers:
        X1 = H1[:, L, :].astype(np.float32)  # (N, D)
        X2 = H2[:, L, :].astype(np.float32)

        # ---- probe_c_binary ----
        y = y_c
        Ctr, mdl_c, auc_c_dev = _fit_lr_probe(X1[tr_idx], y[tr_idx], X1[dv_idx], y[dv_idx],
                                              seed=args.seed, class_weight="balanced")
        pr_c_test = mdl_c.predict_proba(X1[te_idx])[:, 1]
        auc_c_test = float(roc_auc_score(y[te_idx], pr_c_test))
        auc_c_dev = float(auc_c_dev)
        # Isotonic on dev
        pr_c_dv = mdl_c.predict_proba(X1[dv_idx])[:, 1]
        iso = _isotonic_calibrate(pr_c_dv, y[dv_idx])
        pr_c_test_cal = iso.predict(pr_c_test)
        ece_c = _ece(pr_c_test_cal, y[te_idx])
        v_c = _unit(mdl_c.coef_.ravel())
        v_c_by_layer[L] = v_c.astype(np.float32)

        # ---- probe_v_binary ----
        # Only train on parseable rows
        y_v_bin = (c_raw >= c_thresh).astype(np.int64)
        tr_mask = np.intersect1d(tr_idx, np.where(c_parseable)[0])
        dv_mask = np.intersect1d(dv_idx, np.where(c_parseable)[0])
        te_mask = np.intersect1d(te_idx, np.where(c_parseable)[0])
        auc_v_bin_test = None
        auc_v_bin_dev = 0.5
        v_v_bin = np.zeros(D, dtype=np.float64)
        if len(tr_mask) >= 50 and len(np.unique(y_v_bin[tr_mask])) == 2:
            Cv, mdl_v_bin, auc_v_bin_dev = _fit_lr_probe(
                X2[tr_mask], y_v_bin[tr_mask], X2[dv_mask], y_v_bin[dv_mask],
                seed=args.seed, class_weight="balanced",
            )
            pr_v_bin_test = mdl_v_bin.predict_proba(X2[te_mask])[:, 1]
            if len(np.unique(y_v_bin[te_mask])) == 2:
                auc_v_bin_test = float(roc_auc_score(y_v_bin[te_mask], pr_v_bin_test))
            v_v_bin = _unit(mdl_v_bin.coef_.ravel())
        v_v_bin_by_layer[L] = v_v_bin.astype(np.float32)
        auc_v_bin_dev = float(auc_v_bin_dev)

        # ---- probe_v_primary (continuous or ordinal) ----
        primary = {"path": path}
        if path == "continuous":
            # Ridge regression
            if len(tr_mask) >= 50:
                alpha, mdl_v_reg, rho_dev = _fit_ridge_probe(
                    X2[tr_mask], c_raw[tr_mask], X2[dv_mask], c_raw[dv_mask], seed=args.seed
                )
                pr_v_test = mdl_v_reg.predict(X2[te_mask])
                rho_test, _ = sstats.spearmanr(pr_v_test, c_raw[te_mask])
                primary["ridge_alpha"] = alpha
                primary["spearman_dev"] = float(rho_dev)
                primary["spearman_test"] = float(0.0 if np.isnan(rho_test) else rho_test)
        else:
            # ordinal: fit one-vs-rest logistic on 4-class buckets; report top-1 acc + macro-F1
            if len(tr_mask) >= 50:
                from sklearn.linear_model import LogisticRegression as LR
                from sklearn.metrics import f1_score, accuracy_score
                y_ord_tr = to_ordinal(c_raw[tr_mask])
                y_ord_te = to_ordinal(c_raw[te_mask])
                # Skip if only one class present in train
                if len(np.unique(y_ord_tr)) < 2:
                    primary["ordinal_top1_test"] = None
                    primary["ordinal_macro_f1_test"] = None
                else:
                    ordm = LR(
                        C=1.0, penalty="l2", solver="lbfgs", max_iter=1000,
                        multi_class="multinomial", random_state=args.seed,
                        class_weight="balanced",
                    )
                    ordm.fit(X2[tr_mask], y_ord_tr)
                    y_pred_te = ordm.predict(X2[te_mask])
                    primary["ordinal_top1_test"] = float(accuracy_score(y_ord_te, y_pred_te))
                    primary["ordinal_macro_f1_test"] = float(
                        f1_score(y_ord_te, y_pred_te, average="macro", zero_division=0)
                    )

        metrics_per_layer[L] = {
            "probe_c_binary": {
                "C": Ctr,
                "auc_dev": float(auc_c_dev),
                "auc_test": auc_c_test,
                "ece_test_uncalibrated": _ece(pr_c_test, y[te_idx]),
                "ece_test_isotonic": ece_c,
            },
            "probe_v_binary": {
                "auc_dev": auc_v_bin_dev,
                "auc_test": auc_v_bin_test,
                "binarize_threshold": c_thresh,
                "n_train_parseable": int(len(tr_mask)),
            },
            "probe_v_primary": primary,
        }

        if L % 4 == 0:
            print(f"[step4] layer {L}/{num_layers}  auc_c={auc_c_test:.3f}  auc_v_bin={auc_v_bin_test}", flush=True)

    print(f"[step4] layer sweep done in {(time.time()-t0)/60:.1f} min", flush=True)

    # Choose L* (per plan §6.8): argmax of normalized(AUROC_c) + normalized(AUROC_v_bin)
    # **Use DEV-set AUCs to avoid selection-on-test leakage** (per code reviewer feedback).
    # Also skip the embedding layer (L=0) for steering compatibility:
    # a steering hook at model.model.layers[L-1] requires L >= 1.
    def _num_or_default(x, default=0.5):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(x)
    aucs_c_dev = np.array([_num_or_default(metrics_per_layer[L]["probe_c_binary"]["auc_dev"]) for L in layers])
    aucs_v_dev = np.array([_num_or_default(metrics_per_layer[L]["probe_v_binary"]["auc_dev"]) for L in layers])
    aucs_c_test = np.array([_num_or_default(metrics_per_layer[L]["probe_c_binary"]["auc_test"]) for L in layers])
    aucs_v_test = np.array([_num_or_default(metrics_per_layer[L]["probe_v_binary"]["auc_test"]) for L in layers])
    # Search only among layers L >= 1 (skip embedding); needed for the hook alignment.
    mask_search = np.array([L >= 1 for L in layers])
    denom_c = max(aucs_c_dev[mask_search].max(), 1e-6)
    denom_v = max(aucs_v_dev[mask_search].max(), 1e-6)
    scores = aucs_c_dev / denom_c + aucs_v_dev / denom_v
    scores_masked = np.where(mask_search, scores, -np.inf)
    L_star = int(layers[int(np.argmax(scores_masked))])
    print(f"[step4] L* = {L_star} (dev auc_c={aucs_c_dev[L_star]:.3f}  auc_v_bin={aucs_v_dev[L_star]:.3f}  | "
          f"test auc_c={aucs_c_test[L_star]:.3f}  auc_v_bin={aucs_v_test[L_star]:.3f})", flush=True)

    # Cosine trajectory + at L*
    cos_by_layer = []
    for L in layers:
        cos_by_layer.append(
            {
                "layer": L,
                "cos_signed": _cos(v_c_by_layer[L], v_v_bin_by_layer[L]),
                "abs_cos": abs(_cos(v_c_by_layer[L], v_v_bin_by_layer[L])),
                "norm_v_c": float(np.linalg.norm(v_c_by_layer[L])),
                "norm_v_v_bin": float(np.linalg.norm(v_v_bin_by_layer[L])),
            }
        )
    # Neighborhood robustness L*±2
    lo = max(0, L_star - 2)
    hi = min(num_layers, L_star + 2)
    nbhd_abs = np.mean([cos_by_layer[L]["abs_cos"] for L in range(lo, hi + 1)])
    print(f"[step4] |cos| at L*={cos_by_layer[L_star]['abs_cos']:.3f}, "
          f"neighborhood mean L*±2={nbhd_abs:.3f}", flush=True)

    # Bootstrap CIs at L* (retrain-on-bootstrap)
    print(f"[step4] Bootstrap CI at L* over {args.n_bootstrap} resamples …", flush=True)
    L = L_star
    X1L = H1[:, L, :].astype(np.float32)
    X2L = H2[:, L, :].astype(np.float32)
    Ctr = metrics_per_layer[L]["probe_c_binary"]["C"]
    rng = np.random.default_rng(args.seed)
    boot_auc_c = []
    boot_auc_v_bin = []
    boot_abs_cos = []
    tr_pool = tr_idx
    tr_par_pool = np.intersect1d(tr_idx, np.where(c_parseable)[0])
    # Precompute test masks once
    te_pool = te_idx
    te_par_pool = np.intersect1d(te_idx, np.where(c_parseable)[0])

    t_boot0 = time.time()
    for b in range(args.n_bootstrap):
        # Resample train pool with replacement
        idxs_c = rng.choice(tr_pool, size=len(tr_pool), replace=True)
        idxs_v = rng.choice(tr_par_pool, size=len(tr_par_pool), replace=True) if len(tr_par_pool) else np.array([], dtype=np.int64)
        # probe_c on bootstrap
        mdl_c = LogisticRegression(
            C=Ctr, penalty="l2", solver="lbfgs",
            max_iter=100, random_state=args.seed, class_weight="balanced", n_jobs=-1,
        )
        mdl_c.fit(X1L[idxs_c], y_c[idxs_c])
        if len(np.unique(y_c[te_pool])) == 2:
            p_c = mdl_c.predict_proba(X1L[te_pool])[:, 1]
            boot_auc_c.append(roc_auc_score(y_c[te_pool], p_c))
        vc_b = _unit(mdl_c.coef_.ravel())

        # probe_v_bin on bootstrap
        vv_b = np.zeros(D, dtype=np.float64)
        auc_v_b = None
        if len(idxs_v):
            y_bin = (c_raw[idxs_v] >= c_thresh).astype(np.int64)
            if len(np.unique(y_bin)) == 2:
                mdl_v = LogisticRegression(
                    C=1.0, penalty="l2", solver="lbfgs",
                    max_iter=100, random_state=args.seed, class_weight="balanced", n_jobs=-1,
                )
                mdl_v.fit(X2L[idxs_v], y_bin)
                if len(te_par_pool):
                    y_bin_te = (c_raw[te_par_pool] >= c_thresh).astype(np.int64)
                    if len(np.unique(y_bin_te)) == 2:
                        p_v = mdl_v.predict_proba(X2L[te_par_pool])[:, 1]
                        auc_v_b = float(roc_auc_score(y_bin_te, p_v))
                vv_b = _unit(mdl_v.coef_.ravel())
        if auc_v_b is not None:
            boot_auc_v_bin.append(auc_v_b)
        boot_abs_cos.append(abs(_cos(vc_b, vv_b)))
        if (b + 1) % 100 == 0:
            print(f"[step4]   bootstrap {b+1}/{args.n_bootstrap} elapsed {(time.time()-t_boot0):.1f}s", flush=True)

    def _pct(a, plo=2.5, phi=97.5):
        a = np.asarray(a, dtype=np.float64)
        if a.size == 0:
            return {"mean": None, "ci_lo": None, "ci_hi": None, "n": 0}
        return {
            "mean": float(np.mean(a)),
            "ci_lo": float(np.percentile(a, plo)),
            "ci_hi": float(np.percentile(a, phi)),
            "n": int(a.size),
        }

    ci_auc_c = _pct(boot_auc_c)
    ci_auc_v_bin = _pct(boot_auc_v_bin)
    ci_abs_cos = _pct(boot_abs_cos)
    print(f"[step4] AUC_c at L*: {ci_auc_c}", flush=True)
    print(f"[step4] AUC_v_bin at L*: {ci_auc_v_bin}", flush=True)
    print(f"[step4] |cos| at L*: {ci_abs_cos}", flush=True)

    # Nulls: random-direction + shuffled-label at L*
    print("[step4] Random-direction null:", flush=True)
    rand_cosines = []
    rng2 = np.random.default_rng(args.seed + 1)
    v_c_star = _unit(v_c_by_layer[L_star])
    v_v_star = _unit(v_v_bin_by_layer[L_star])
    for _ in range(args.n_random_null):
        r = rng2.normal(size=D)
        r = _unit(r)
        rand_cosines.append(abs(_cos(v_c_star, r)))
    rand_stats = {
        "mean_abs_cos_random": float(np.mean(rand_cosines)),
        "theoretical_std": float(1.0 / np.sqrt(D)),
        "n_samples": args.n_random_null,
    }

    # Shuffled-label null: retrain probe_c on shuffled y at L*
    print("[step4] Shuffled-label null probe_c at L*", flush=True)
    tr = tr_idx
    rng3 = np.random.default_rng(args.seed + 2)
    y_perm = y_c.copy()
    y_perm[tr] = rng3.permutation(y_perm[tr])
    mdl_shuf = LogisticRegression(
        C=metrics_per_layer[L_star]["probe_c_binary"]["C"],
        penalty="l2", solver="lbfgs", max_iter=200, random_state=args.seed, n_jobs=-1,
    )
    mdl_shuf.fit(X1L[tr], y_perm[tr])
    p_shuf = mdl_shuf.predict_proba(X1L[te_idx])[:, 1]
    auc_shuf = float(roc_auc_score(y_c[te_idx], p_shuf)) if len(np.unique(y_c[te_idx])) == 2 else None
    v_shuf = _unit(mdl_shuf.coef_.ravel())
    cos_shuf = abs(_cos(v_c_star, v_shuf))

    # Token-probability baseline for C1 (proxy: mean generated token prob is
    # inaccessible without re-scoring; we approximate via the model's own
    # binarized correctness confidence = share_correct at threshold).
    # Since token probs require rescoring with logits, we defer this to a
    # simple ECE using answer-length proxy — noted as an approximation.
    # Instead we compute ECE(probe) < ECE(base_rate) as a fallback baseline.
    base_rate = float(np.mean(y_c[te_idx]))
    base_probs = np.full_like(y_c[te_idx], base_rate, dtype=np.float64)
    ece_base = _ece(base_probs, y_c[te_idx])

    # Save all outputs
    out = {
        "L_star": L_star,
        "num_layers": num_layers,
        "hidden_dim": D,
        "per_layer": {str(L): metrics_per_layer[L] for L in layers},
        "L_star_metrics": {
            "auc_c_test": metrics_per_layer[L_star]["probe_c_binary"]["auc_test"],
            "auc_v_bin_test": metrics_per_layer[L_star]["probe_v_binary"]["auc_test"],
            "ece_c_isotonic": metrics_per_layer[L_star]["probe_c_binary"]["ece_test_isotonic"],
            "ece_base_rate": ece_base,
            "auc_c_bootstrap_ci": ci_auc_c,
            "auc_v_bin_bootstrap_ci": ci_auc_v_bin,
            "abs_cos_bootstrap_ci": ci_abs_cos,
            "abs_cos_neighborhood_L2": float(nbhd_abs),
            "primary_v": metrics_per_layer[L_star]["probe_v_primary"],
        },
        "seed": args.seed,
    }
    U.dump_json(U.ARTIFACT_DIR / "probe_metrics.json", out)

    cos_out = {
        "L_star": L_star,
        "per_layer": cos_by_layer,
        "neighborhood_abs_cos_mean": float(nbhd_abs),
        "abs_cos_at_Lstar": float(cos_by_layer[L_star]["abs_cos"]),
    }
    U.dump_json(U.ARTIFACT_DIR / "cos_trajectory.json", cos_out)

    U.dump_json(
        U.ARTIFACT_DIR / "cos_bootstrap.json",
        {"L_star": L_star, "n_bootstrap": args.n_bootstrap, **ci_abs_cos},
    )

    U.dump_json(
        U.ARTIFACT_DIR / "nulls.json",
        {
            "L_star": L_star,
            "random_direction_null": rand_stats,
            "shuffled_label_probe_c": {
                "auc_test": auc_shuf,
                "abs_cos_vs_v_c_star": float(cos_shuf),
            },
        },
    )

    np.savez_compressed(
        U.ARTIFACT_DIR / "probes.npz",
        v_c_by_layer=v_c_by_layer,
        v_v_bin_by_layer=v_v_bin_by_layer,
        v_c_star=v_c_star.astype(np.float32),
        v_v_star=v_v_star.astype(np.float32),
        L_star=np.array([L_star]),
    )

    print(f"[step4] wrote {U.ARTIFACT_DIR / 'probe_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
