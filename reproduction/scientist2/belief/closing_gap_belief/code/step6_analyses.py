"""Step 6 — Downstream analyses:

- B4 (dissociation-when-disagree): bin (probe_c_calibrated_output, verbalized_c),
  compute per-cell accuracy, and McNemar test on (low probe, high verbal) vs.
  (low probe, low verbal).
- B6 (nulls + paraphrase): re-fit probe_v_binary on H_2^L* under P1 and P2
  paraphrases (extracted in step 2), report ΔSpearman/AUROC/|cos| vs P0.
- B7 (single-pass): fit probe_c_single, probe_v_single on H_single^L* and
  compute cos(v_c_single, v_v_single).
- Nulls already computed in step4 (random-direction + shuffled-label).

Inputs: probes.npz, probe_metrics.json, cos_trajectory.json,
        H_turn1.npz, H_turn2.npz, H_turn2_p1.npz, H_turn2_p2.npz, H_single.npz,
        turn1_gen.jsonl, turn2_gen.jsonl, turn2_p1_gen.jsonl, turn2_p2_gen.jsonl,
        single_gen.jsonl, split_manifest.json, stage15_variance.json.

Outputs:
  - artifacts/dissociation.json  (B4)
  - artifacts/paraphrase.json    (B6)
  - artifacts/single_pass.json   (B7)
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sstats
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils as U


def _mcnemar(b: int, c: int):
    """Basic McNemar (chi-square, continuity corrected). b = only-in-A correct,
    c = only-in-B correct."""
    if b + c == 0:
        return {"stat": 0.0, "p": 1.0, "b": b, "c": c}
    stat = ((abs(b - c) - 1) ** 2) / (b + c)
    p = 1.0 - sstats.chi2.cdf(stat, df=1)
    return {"stat": float(stat), "p": float(p), "b": int(b), "c": int(c)}


def _unit(v):
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def _cos(u, v):
    u = _unit(u); v = _unit(v)
    return float(np.dot(u, v))


def _load_split_masks(qids: np.ndarray, split: dict, idxs: np.ndarray = None):
    """Prefer idx-based splits (unique per row) over qid-based splits (TriviaQA has
    duplicate qids -> qid-based splits leak between train/dev/test)."""
    if idxs is not None and "train_idxs" in split:
        train_idxs = set(int(x) for x in split["train_idxs"])
        dev_idxs = set(int(x) for x in split["dev_idxs"])
        test_idxs = set(int(x) for x in split["test_idxs"])
        idxs_list = [int(x) for x in idxs.tolist()]
        tr = np.array([i for i, ix in enumerate(idxs_list) if ix in train_idxs], dtype=np.int64)
        dv = np.array([i for i, ix in enumerate(idxs_list) if ix in dev_idxs], dtype=np.int64)
        te = np.array([i for i, ix in enumerate(idxs_list) if ix in test_idxs], dtype=np.int64)
        return tr, dv, te
    train_ids = set(split["train_ids"])
    dev_ids = set(split["dev_ids"])
    test_ids = set(split["test_ids"])
    tr = np.array([i for i, q in enumerate(qids.tolist()) if q in train_ids], dtype=np.int64)
    dv = np.array([i for i, q in enumerate(qids.tolist()) if q in dev_ids], dtype=np.int64)
    te = np.array([i for i, q in enumerate(qids.tolist()) if q in test_ids], dtype=np.int64)
    return tr, dv, te


def _qkey_lookup(q, *dicts):
    """Return q in a form that matches one of the dicts (str or int)."""
    for d in dicts:
        if q in d:
            return q
    try:
        iq = int(q)
        for d in dicts:
            if iq in d:
                return iq
    except (TypeError, ValueError):
        pass
    return str(q)


def main():
    probes = np.load(U.ARTIFACT_DIR / "probes.npz")
    L_star = int(probes["L_star"][0])
    v_c_star = probes["v_c_star"]
    v_v_star = probes["v_v_star"]

    metrics = U.load_json(U.ARTIFACT_DIR / "probe_metrics.json")
    split = U.load_json(U.ARTIFACT_DIR / "split_manifest.json")
    stage15 = U.load_json(U.ARTIFACT_DIR / "stage15_variance.json")
    c_thresh = float(stage15["binarize_threshold"])

    # --- B4: dissociation-when-disagree ---
    print("[step6] B4: dissociation-when-disagree", flush=True)
    turn1_rows = U.load_jsonl(U.ARTIFACT_DIR / "turn1_gen.jsonl")
    turn2_rows = U.load_jsonl(U.ARTIFACT_DIR / "turn2_gen.jsonl")
    y_by_qid = {r["question_id"]: int(r["y_correct"]) for r in turn1_rows}
    c_by_qid = {r["question_id"]: (float(r["c"]) if r["c_parseable"] else None) for r in turn2_rows}

    # Compute probe_c calibrated output on the 2k held-out test slice
    npz1 = np.load(U.ARTIFACT_DIR / "H_turn1.npz")
    H1 = npz1["H"]
    qids = npz1["question_ids"].tolist()
    idxs_arr = npz1["idxs"]
    tr, dv, te = _load_split_masks(np.array(qids), split, idxs=idxs_arr)
    X1L = H1[:, L_star, :].astype(np.float32)
    y_c = np.array([y_by_qid.get(_qkey_lookup(q, y_by_qid), 0) for q in qids], dtype=np.int64)
    # Refit probe_c on train + dev (like final) with best C
    best_C = metrics["per_layer"][str(L_star)]["probe_c_binary"]["C"]
    mdl_c = LogisticRegression(
        C=best_C, penalty="l2", solver="liblinear", max_iter=1000,
        random_state=42, class_weight="balanced",
    )
    mdl_c.fit(X1L[tr], y_c[tr])
    p_dev = mdl_c.predict_proba(X1L[dv])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_dev, y_c[dv])
    p_test_raw = mdl_c.predict_proba(X1L[te])[:, 1]
    p_test_cal = iso.predict(p_test_raw)

    # Binning: probe_c_calibrated (low: <0.5) × c (high: >=50) contingency
    test_qids = [qids[i] for i in te]
    def _cv(q):
        k = _qkey_lookup(q, c_by_qid)
        v = c_by_qid.get(k, np.nan)
        return np.nan if v is None else v
    c_test = np.array([_cv(q) for q in test_qids])
    parseable = ~np.isnan(c_test)
    # Only keep parseable
    mask = parseable
    p_low = p_test_cal[mask] < 0.5
    c_high = c_test[mask] >= 50.0
    y_test = y_c[te][mask]

    cells = {}
    for p_bin in [False, True]:
        for c_bin in [False, True]:
            sel = (p_low == p_bin) & (c_high == c_bin)
            n = int(sel.sum())
            key = f"probe_{'low' if p_bin else 'high'}_verbal_{'high' if c_bin else 'low'}"
            acc = float(y_test[sel].mean()) if n > 0 else None
            cells[key] = {"n": n, "accuracy": acc}
    # McNemar-like test on the two low-probe cells (paired by index):
    # (probe_low & verbal_high) vs (probe_low & verbal_low) — but these are
    # disjoint samples, so use a proportions z-test instead.
    low_hi = cells["probe_low_verbal_high"]
    low_lo = cells["probe_low_verbal_low"]
    if low_hi["n"] > 0 and low_lo["n"] > 0:
        # 2-prop z-test
        p1 = low_hi["accuracy"]; n1 = low_hi["n"]
        p2 = low_lo["accuracy"]; n2 = low_lo["n"]
        pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = float(np.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2)))
        z = (p1 - p2) / (se + 1e-12)
        # one-sided p-value: alternative is p1 < p2
        p_val = float(sstats.norm.cdf(z))
        stat_test = {
            "test": "two_proportion_z",
            "p1_low_verbal_high": p1,
            "p2_low_verbal_low": p2,
            "n1": n1,
            "n2": n2,
            "z": float(z),
            "p_one_sided": p_val,
            "significant_at_0.05": p_val < 0.05,
        }
    else:
        stat_test = {"error": "one of the cells is empty"}
    U.dump_json(U.ARTIFACT_DIR / "dissociation.json", {
        "L_star": L_star,
        "cells": cells,
        "test": stat_test,
        "success_criterion": "Accuracy(low probe, high verbal) < Accuracy(low probe, low verbal), p<0.05",
        "passes": bool(isinstance(stat_test, dict) and stat_test.get("significant_at_0.05")
                       and stat_test.get("p1_low_verbal_high", 1.0) < stat_test.get("p2_low_verbal_low", 0.0)),
    })

    # --- B6: paraphrase P1, P2 ---
    print("[step6] B6: paraphrase P1, P2", flush=True)
    para_out = {"P0_v_v_star_ref": True}
    for pkey, mode in [("P1", "turn2_p1"), ("P2", "turn2_p2")]:
        try:
            npz = np.load(U.ARTIFACT_DIR / f"H_{mode}.npz")
        except FileNotFoundError:
            para_out[pkey] = {"status": "no_hidden_states"}
            continue
        Hp = npz["H"]
        qids_p = npz["question_ids"].tolist()
        rows_p = U.load_jsonl(U.ARTIFACT_DIR / f"{mode}_gen.jsonl")
        c_p_by_qid = {r["question_id"]: (float(r["c"]) if r["c_parseable"] else None) for r in rows_p}
        # These are the dev-500 slice; parseable rows only
        def _cvp(q):
            k = _qkey_lookup(q, c_p_by_qid)
            v = c_p_by_qid.get(k, np.nan)
            return np.nan if v is None else v
        c_p = np.array([_cvp(q) for q in qids_p], dtype=np.float64)
        mask_par = ~np.isnan(c_p)
        Xp = Hp[:, L_star, :].astype(np.float32)
        # Also get P0 c on these SAME qids
        def _cvp0(q):
            k = _qkey_lookup(q, c_by_qid)
            v = c_by_qid.get(k, np.nan)
            return np.nan if v is None else v
        c_p0 = np.array([_cvp0(q) for q in qids_p], dtype=np.float64)
        mask_both = mask_par & (~np.isnan(c_p0))
        # ΔSpearman: paired sample of P0 vs Pk c-values
        if int(mask_both.sum()) >= 20:
            rho_p, _ = sstats.spearmanr(c_p[mask_both], c_p0[mask_both])
        else:
            rho_p = np.nan
        # Refit probe_v_binary on Hp at L*, using c_p binarized at same threshold
        y_v_p = (c_p >= c_thresh).astype(np.int64)
        auc_p = None
        cos_p_vs_v_c = None
        cos_p_vs_v_v_p0 = None
        if int(mask_par.sum()) >= 50 and len(np.unique(y_v_p[mask_par])) == 2:
            # 80/20 split within this dev-500 slice
            n = int(mask_par.sum())
            idx = np.where(mask_par)[0]
            rng = np.random.default_rng(42)
            rng.shuffle(idx)
            n_tr = int(0.7 * n); n_dv = int(0.15 * n)
            tr_p = idx[:n_tr]; dv_p = idx[n_tr:n_tr + n_dv]; te_p = idx[n_tr + n_dv:]
            mdl_v_p = LogisticRegression(
                C=1.0, penalty="l2", solver="liblinear",
                max_iter=1000, random_state=42, class_weight="balanced",
            )
            mdl_v_p.fit(Xp[tr_p], y_v_p[tr_p])
            if len(np.unique(y_v_p[te_p])) == 2:
                p_te = mdl_v_p.predict_proba(Xp[te_p])[:, 1]
                auc_p = float(roc_auc_score(y_v_p[te_p], p_te))
            v_v_p = _unit(mdl_v_p.coef_.ravel())
            cos_p_vs_v_c = abs(_cos(v_c_star, v_v_p))
            cos_p_vs_v_v_p0 = abs(_cos(v_v_star, v_v_p))
        # Read v_v_p0 auc for delta
        auc_v_p0 = metrics["L_star_metrics"]["auc_v_bin_test"]
        para_out[pkey] = {
            "n_parseable": int(mask_par.sum()),
            "spearman_c_vs_P0": float(0.0 if np.isnan(rho_p) else rho_p),
            "auc_v_binarized": auc_p,
            "abs_cos_vs_v_c_star": cos_p_vs_v_c,
            "abs_cos_vs_v_v_p0": cos_p_vs_v_v_p0,
            "delta_auc": (auc_p - auc_v_p0) if (auc_p is not None and auc_v_p0 is not None) else None,
        }
    U.dump_json(U.ARTIFACT_DIR / "paraphrase.json", para_out)

    # --- B7: single-pass variant ---
    print("[step6] B7: single-pass |cos|", flush=True)
    try:
        npzS = np.load(U.ARTIFACT_DIR / "H_single.npz")
    except FileNotFoundError:
        U.dump_json(U.ARTIFACT_DIR / "single_pass.json", {"status": "no_hidden_states"})
        return
    HS = npzS["H"]
    qids_s = npzS["question_ids"].tolist()
    idxs_s = npzS["idxs"]
    single_rows = U.load_jsonl(U.ARTIFACT_DIR / "single_gen.jsonl")
    y_s_by_qid = {r["question_id"]: int(r["y_correct"]) for r in single_rows}
    c_s_by_qid = {r["question_id"]: (float(r["c"]) if r["c_parseable"] else None) for r in single_rows}

    y_s = np.array([y_s_by_qid.get(_qkey_lookup(q, y_s_by_qid), 0) for q in qids_s], dtype=np.int64)
    def _cvs(q):
        k = _qkey_lookup(q, c_s_by_qid)
        v = c_s_by_qid.get(k, np.nan)
        return np.nan if v is None else v
    c_s = np.array([_cvs(q) for q in qids_s], dtype=np.float64)
    parseable_s = ~np.isnan(c_s)

    # Splits using idx membership (avoid duplicate-qid leakage)
    tr_s, dv_s, te_s = _load_split_masks(np.array(qids_s), split, idxs=idxs_s)
    XSL = HS[:, L_star, :].astype(np.float32)

    # probe_c_single
    mdl_c_s = LogisticRegression(
        C=metrics["per_layer"][str(L_star)]["probe_c_binary"]["C"],
        penalty="l2", solver="liblinear", max_iter=1000, random_state=42, class_weight="balanced",
    )
    mdl_c_s.fit(XSL[tr_s], y_s[tr_s])
    if len(np.unique(y_s[te_s])) == 2:
        auc_c_s = float(roc_auc_score(y_s[te_s], mdl_c_s.predict_proba(XSL[te_s])[:, 1]))
    else:
        auc_c_s = None
    v_c_s = _unit(mdl_c_s.coef_.ravel())

    # probe_v_single (binary)
    tr_mask = np.intersect1d(tr_s, np.where(parseable_s)[0])
    te_mask = np.intersect1d(te_s, np.where(parseable_s)[0])
    y_v_s = (c_s >= c_thresh).astype(np.int64)
    auc_v_s = None
    v_v_s = np.zeros_like(v_c_s)
    if len(tr_mask) >= 50 and len(np.unique(y_v_s[tr_mask])) == 2:
        mdl_v_s = LogisticRegression(
            C=1.0, penalty="l2", solver="liblinear", max_iter=1000, random_state=42, class_weight="balanced",
        )
        mdl_v_s.fit(XSL[tr_mask], y_v_s[tr_mask])
        if len(np.unique(y_v_s[te_mask])) == 2:
            auc_v_s = float(roc_auc_score(y_v_s[te_mask], mdl_v_s.predict_proba(XSL[te_mask])[:, 1]))
        v_v_s = _unit(mdl_v_s.coef_.ravel())

    abs_cos_single = abs(_cos(v_c_s, v_v_s))
    abs_cos_two_pass = float(metrics["L_star_metrics"]["abs_cos_bootstrap_ci"]["mean"])

    single_out = {
        "L_star": L_star,
        "n_total": len(qids_s),
        "n_parseable": int(parseable_s.sum()),
        "auc_c_single_test": auc_c_s,
        "auc_v_binarized_single_test": auc_v_s,
        "abs_cos_single_pass": abs_cos_single,
        "abs_cos_two_pass_ref": abs_cos_two_pass,
        "cross_condition_abs_cos_v_c_single_vs_v_c_star": abs(_cos(v_c_s, v_c_star)),
        "cross_condition_abs_cos_v_v_single_vs_v_v_star": abs(_cos(v_v_s, v_v_star)),
    }
    # Pre-registered interpretation
    interp = None
    if abs_cos_two_pass <= 0.3 and abs_cos_single <= 0.3:
        interp = "both_pass_dissociation_strengthened"
    elif abs_cos_two_pass <= 0.3 and abs_cos_single > 0.5:
        interp = "two_pass_only_no_single_shared_conclusion"
    elif abs_cos_single > 0.3 and abs_cos_two_pass > 0.3:
        interp = "c3a_not_supported"
    else:
        interp = "intermediate_reported_honestly"
    single_out["interpretation"] = interp

    U.dump_json(U.ARTIFACT_DIR / "single_pass.json", single_out)
    print(f"[step6] wrote single_pass.json  interp={interp}", flush=True)


if __name__ == "__main__":
    main()
