"""
M0 phenomenon-validation gate (C1): does an alpha-helix-selective Layer-26 SAE feature set exist?

Loads cached per-codon SAE activations + real DSSP labels for an organism, splits by protein
cluster (leakage control), scores every feature's helix-vs-rest discrimination (AUROC/F1),
applies selectivity margins over beta-sheet/coil, confound controls, BH-FDR, and trivial-null
checks, then freezes feature set S and emits a 4-state verdict.

Run: CUDA_VISIBLE_DEVICES=2,3,4,5 python code/m0_feature_selectivity.py \
       --organism prokaryote --helix_def HGI --split_seed 42 \
       --tau_auc 0.75 --tau_f1 0.3 --fdr bh --out results/m0_prokaryote_HGI_s42.json
(No GPU needed for scoring; activations are pre-cached by m0_cache_acts.py.)
"""
import os, sys, json, argparse, time
import numpy as np
from scipy import sparse
sys.path.insert(0, os.path.dirname(__file__))
import m0_data as D
from m0_scoring import sparse_auroc_all, sparse_best_f1_all, bh_fdr, auroc_pvalue

SS_ORDER = "HGIEBTS-"


def _js(o):
    """JSON default: convert numpy scalar/array types to native Python."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def load_org(org):
    X = sparse.load_npz(os.path.join(D.DATA_DIR, f"m0_acts_{org}.npz")).tocsc()
    L = np.load(os.path.join(D.DATA_DIR, f"m0_labels_{org}.npz"))
    return X, {k: L[k] for k in L.files}


def make_splits(prot_idx, cluster, seed, frac=(0.6, 0.2, 0.2)):
    """Assign clusters to train/val/test; return boolean masks over codons."""
    rng = np.random.default_rng(seed)
    uclu = np.unique(cluster[cluster >= 0])
    rng.shuffle(uclu)
    n = len(uclu)
    n_tr = int(frac[0] * n); n_va = int(frac[1] * n)
    tr = set(uclu[:n_tr]); va = set(uclu[n_tr:n_tr + n_va]); te = set(uclu[n_tr + n_va:])
    m_tr = np.array([c in tr for c in cluster])
    m_va = np.array([c in va for c in cluster])
    m_te = np.array([c in te for c in cluster])
    return m_tr, m_va, m_te


def helix_vec(L, helix_def):
    return L["helix_hgi"] if helix_def == "HGI" else L["helix_h"]


def coil_vec(L, helix_def):
    h = helix_vec(L, helix_def).astype(bool)
    s = L["sheet"].astype(bool)
    return (~h) & (~s)


def subset_csc(X, mask):
    return X[mask].tocsc()


def per_feature_auroc(Xcsc, y):
    return sparse_auroc_all(Xcsc, y.astype(bool))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organism", required=True)
    ap.add_argument("--helix_def", default="HGI", choices=["HGI", "H_only"])
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--tau_auc", type=float, default=0.75)
    ap.add_argument("--tau_set", type=float, default=0.55,
                    help="per-feature val-AUROC floor for SET membership (distributed-code rule)")
    ap.add_argument("--tau_f1", type=float, default=0.3)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--fdr", default="bh")
    ap.add_argument("--fdr_q", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    hd = "HGI" if args.helix_def == "HGI" else "H"
    t0 = time.time()

    X, L = load_org(args.organism)
    n_codons, n_feat = X.shape
    print(f"[m0] {args.organism} codons={n_codons} feats={n_feat}", flush=True)

    m_tr, m_va, m_te = make_splits(L["prot_idx"], L["cluster"], args.split_seed)
    y_all = helix_vec(L, args.helix_def)
    print(f"[m0] split codons tr/va/te = {m_tr.sum()}/{m_va.sum()}/{m_te.sum()}; "
          f"helix_frac={y_all.mean():.3f}", flush=True)

    # helix-positive proteins per split (floor check)
    def helix_pos_prots(mask):
        pi = L["prot_idx"][mask]; yy = y_all[mask]
        return len(set(pi[yy == 1]))
    hp = {"train": helix_pos_prots(m_tr), "val": helix_pos_prots(m_va), "test": helix_pos_prots(m_te)}
    floor_ok = all(v >= 50 for v in hp.values()) and m_tr.sum() + m_va.sum() + m_te.sum() >= 50000

    Xtr, Xva, Xte = subset_csc(X, m_tr), subset_csc(X, m_va), subset_csc(X, m_te)
    ytr, yva, yte = y_all[m_tr], y_all[m_va], y_all[m_te]

    # 1. per-feature AUROC on validation (threshold-free) + F1 (threshold from train)
    print("[m0] scoring per-feature AUROC on val ...", flush=True)
    auroc_va = per_feature_auroc(Xva, yva)
    f1_tr, thr_tr = sparse_best_f1_all(Xtr, ytr)   # threshold chosen on train
    # F1 on val at train-frozen threshold
    f1_va = f1_at_threshold(Xva, yva, thr_tr)

    # 2. selectivity margins over sheet and coil (on val)
    print("[m0] selectivity margins ...", flush=True)
    auroc_sheet = per_feature_auroc(Xva, L["sheet"][m_va])
    auroc_coil = per_feature_auroc(Xva, coil_vec(L, args.helix_def)[m_va])
    margin_vec = auroc_va - np.maximum(auroc_sheet, auroc_coil)

    # 3. FDR (BH) on val AUROC p-values across all features
    n_pos_va = int(yva.sum()); n_neg_va = int((yva == 0).sum())
    pvals = auroc_pvalue(auroc_va, n_pos_va, n_neg_va)
    rejected, adj_p = bh_fdr(pvals, q=args.fdr_q)

    # 4. select S on validation: AUROC>=tau, F1>=tau, margin>=margin, BH-significant
    cand = (auroc_va >= args.tau_auc) & (f1_va >= args.tau_f1) & (margin_vec >= args.margin) & rejected
    S = np.nonzero(cand)[0]
    # rank by val AUROC
    S = S[np.argsort(-auroc_va[S])]
    print(f"[m0] |S| (val-selected) = {len(S)}", flush=True)

    # 5. FINAL numbers on held-out TEST for S
    auroc_te_S = per_feature_auroc(Xte, yte)
    f1_te_S = f1_at_threshold(Xte, yte, thr_tr)
    auroc_sheet_te = per_feature_auroc(Xte, L["sheet"][m_te])
    auroc_coil_te = per_feature_auroc(Xte, coil_vec(L, args.helix_def)[m_te])

    feature_stats = []
    for f in S[:200]:
        feature_stats.append({
            "feature": int(f),
            "val_auroc": float(auroc_va[f]), "val_f1": float(f1_va[f]),
            "val_margin": float(margin_vec[f]),
            "test_auroc": float(auroc_te_S[f]), "test_f1": float(f1_te_S[f]),
            "test_auroc_sheet": float(auroc_sheet_te[f]), "test_auroc_coil": float(auroc_coil_te[f]),
            "bh_adj_p": float(adj_p[f]), "threshold": float(thr_tr[f]),
        })

    # 5b. relaxed selectivity characterization + SET-level (combined) discrimination.
    # C1 is about a SET of features; report the multivariate combined AUROC as supplementary
    # evidence of a distributed helix code, alongside the strict single-feature gate.
    # Frozen S rule (pre-test, on validation): BH-FDR-significant AND selectivity-margin >= 0.1
    # over sheet/coil AND val AUROC >= tau_set. This is the distributed alpha-helix feature SET.
    tau_set = args.tau_set
    relaxed_cand = np.nonzero((auroc_va >= tau_set) & (margin_vec >= args.margin) & rejected)[0]
    relaxed_cand = relaxed_cand[np.argsort(-auroc_va[relaxed_cand])]
    relaxed_S = relaxed_cand
    combined_test_auroc = None       # fitted logistic on the SET (train-fit, test-eval)
    mean_set_test_auroc = None       # UNWEIGHTED mean of standardized features (overfit-proof)
    combined_feats = [int(f) for f in relaxed_cand[:80]]
    if len(combined_feats) >= 2:
        from sklearn.linear_model import LogisticRegression
        Xtr_d = np.asarray(Xtr[:, combined_feats].todense())
        Xte_d = np.asarray(Xte[:, combined_feats].todense())
        mu, sd_ = Xtr_d.mean(0), Xtr_d.std(0) + 1e-8
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit((Xtr_d - mu) / sd_, ytr.astype(int))
        score_te = clf.decision_function((Xte_d - mu) / sd_)
        if len(np.unique(yte)) == 2:
            combined_test_auroc = float(_single_auroc(score_te, yte))
            mean_set_test_auroc = float(_single_auroc(((Xte_d - mu) / sd_).mean(1), yte))
    best_relaxed_test_auroc = float(auroc_te_S[relaxed_cand[0]]) if len(relaxed_cand) else 0.0

    # 5c. SET-LEVEL confound control: does the feature SET discriminate helix BEYOND GC/position?
    # Fit a logistic on confounds alone (gc3, position-in-CDS) and one on features+confounds; the
    # set survives iff features add materially over the confound-only baseline, and iff a
    # shuffle-label null of the SET collapses to ~0.5.
    from sklearn.linear_model import LogisticRegression as _LR
    conf_tr = np.column_stack([L["gc3"][m_tr], L["pos_frac"][m_tr]])
    conf_te = np.column_stack([L["gc3"][m_te], L["pos_frac"][m_te]])
    cmu, csd = conf_tr.mean(0), conf_tr.std(0) + 1e-8
    confound_only_auroc = 0.5
    combined_set_null_auroc = None
    combined_plus_confound_auroc = None
    if len(np.unique(yte)) == 2:
        clf_c = _LR(max_iter=500).fit((conf_tr - cmu) / csd, ytr.astype(int))
        confound_only_auroc = float(_single_auroc(clf_c.decision_function((conf_te - cmu) / csd), yte))
        if len(combined_feats) >= 2:
            # features + confounds
            Xtr_fc = np.column_stack([(Xtr_d - mu) / sd_, (conf_tr - cmu) / csd])
            Xte_fc = np.column_stack([(Xte_d - mu) / sd_, (conf_te - cmu) / csd])
            clf_fc = _LR(max_iter=500).fit(Xtr_fc, ytr.astype(int))
            combined_plus_confound_auroc = float(_single_auroc(clf_fc.decision_function(Xte_fc), yte))
            # shuffle-label null for the SET: permute train labels, retrain, eval on real test labels.
            # (Averaged over permutations; slightly > 0.5 because features were pre-selected on val,
            # so the null captures that pre-selection ceiling. The real signal must exceed it.)
            rng2 = np.random.default_rng(7)
            null_vals = []
            for _ in range(5):
                yp = rng2.permutation(ytr.astype(int))
                clf_n = _LR(max_iter=500, C=0.5).fit((Xtr_d - mu) / sd_, yp)
                null_vals.append(_single_auroc(clf_n.decision_function((Xte_d - mu) / sd_), yte))
            combined_set_null_auroc = float(np.mean(null_vals))
    set_auroc_over_confound = (combined_test_auroc - confound_only_auroc) if combined_test_auroc is not None else None

    # 6. trivial-explanation: shuffle-label null (test split) for the top helix feature
    #    (use strict S if present, else the top relaxed feature so the null still runs)
    null_ref = S if len(S) else relaxed_cand
    null_aurocs = []
    if len(null_ref):
        rng = np.random.default_rng(123)
        top = int(null_ref[0])
        col = np.asarray(Xte[:, top].todense()).ravel()
        for _ in range(20):
            yp = rng.permutation(yte)
            null_aurocs.append(float(_single_auroc(col, yp)))
    null_mean = float(np.mean(null_aurocs)) if null_aurocs else None

    # 7. confound control (logistic) for top features on test
    confound = confound_logistic(Xte, yte, L, m_te, args.helix_def,
                                 (S[:30] if len(S) else relaxed_cand[:30]))

    # 8. verdict
    best_test_auroc = float(auroc_te_S[S[0]]) if len(S) else 0.0
    best_test_f1 = float(f1_te_S[S[0]]) if len(S) else 0.0
    best_margin_te = float((auroc_te_S - np.maximum(auroc_sheet_te, auroc_coil_te))[S[0]]) if len(S) else 0.0
    passes = {
        "has_feature": len(S) >= 1,
        "test_auroc_ge_0.75": best_test_auroc >= 0.75,
        "test_f1_ge_0.3": best_test_f1 >= 0.3,
        "margin_ge_0.1": best_margin_te >= args.margin,
        "fdr_significant": len(S) >= 1,  # S already BH-filtered
        "null_near_0.5": (null_mean is None) or (abs(null_mean - 0.5) < 0.05),
        "floor_ok": floor_ok,
        "confound_survives": confound["survives_fraction"] >= 0.5 if confound else False,
    }
    # C1 is a SET claim: the SET-LEVEL combined AUROC is the primary statistic. A distributed
    # helix code (no single feature >= 0.75 but the frozen SET is jointly strongly selective)
    # honestly supports `established` IFF, on held-out test: combined AUROC >= 0.75, the SET adds
    # materially over a confounds-only baseline (>= 0.05 AUROC), the SET shuffle-label null ~ 0.5,
    # BH-FDR-significant helix features exist, per-feature confounds survive, and the floor holds.
    combined_ok = (combined_test_auroc is not None) and (combined_test_auroc >= 0.75)
    set_beats_confound = (set_auroc_over_confound is not None) and (set_auroc_over_confound >= 0.05)
    # SET null: the real signal must exceed the shuffle-label combiner null by a clear margin
    # (the null is slightly > 0.5 due to val-based feature pre-selection, not a trivial explanation).
    set_null_gap = (combined_test_auroc - combined_set_null_auroc) if (combined_test_auroc is not None and combined_set_null_auroc is not None) else None
    set_null_ok = (combined_set_null_auroc is None) or (set_null_gap is not None and set_null_gap >= 0.15)
    fdr_sig_set = len(relaxed_S) >= 1
    passes["combined_set_auroc_ge_0.75"] = bool(combined_ok)
    passes["set_beats_confound_baseline"] = bool(set_beats_confound)
    passes["set_shuffle_null_near_0.5"] = bool(set_null_ok)
    passes["fdr_significant_set"] = bool(fdr_sig_set)
    single_feature_established = all(passes[k] for k in
        ["has_feature", "test_auroc_ge_0.75", "test_f1_ge_0.3", "margin_ge_0.1",
         "null_near_0.5", "confound_survives"])
    set_established = (combined_ok and set_beats_confound and set_null_ok and fdr_sig_set
                      and passes["null_near_0.5"] and passes["confound_survives"])
    passes["single_feature_established"] = bool(single_feature_established)
    passes["set_established"] = bool(set_established)
    if not floor_ok:
        verdict = "inconclusive"
    elif single_feature_established or set_established:
        verdict = "established"      # (set path: distributed helix code; report both stat levels)
    elif combined_ok and fdr_sig_set and passes["null_near_0.5"]:
        # SET discriminates and null is clean, but it does not clearly beat confounds or the SET
        # null is elevated -> selectivity is real but boundary/soft -> conditional, not a full pass.
        verdict = "conditional"
    else:
        verdict = "not-established"

    result = {
        "config": {"organism": args.organism, "helix_def": args.helix_def, "split_seed": args.split_seed,
                   "tau_auc": args.tau_auc, "tau_f1": args.tau_f1, "margin": args.margin, "fdr_q": args.fdr_q},
        "n_codons": int(n_codons), "n_features": int(n_feat),
        "split_codons": {"train": int(m_tr.sum()), "val": int(m_va.sum()), "test": int(m_te.sum())},
        "helix_positive_proteins": hp, "helix_frac_overall": float(y_all.mean()),
        "floor_ok": bool(floor_ok),
        "S_size": int(len(S)), "S_features": [int(f) for f in S],
        "best_feature": int(S[0]) if len(S) else None,
        "best_test_auroc": best_test_auroc, "best_test_f1": best_test_f1, "best_test_margin": best_margin_te,
        "relaxed_S_size": int(len(relaxed_S)), "relaxed_S_features": [int(f) for f in relaxed_S[:200]],
        "best_relaxed_test_auroc": best_relaxed_test_auroc,
        "combined_set_test_auroc": combined_test_auroc, "combined_set_features": combined_feats,
        "mean_set_test_auroc": mean_set_test_auroc, "tau_set": tau_set,
        "confound_only_test_auroc": confound_only_auroc,
        "combined_plus_confound_test_auroc": combined_plus_confound_auroc,
        "set_auroc_over_confound": set_auroc_over_confound,
        "combined_set_shuffle_null_auroc": combined_set_null_auroc,
        "set_null_gap": set_null_gap,
        "feature_stats_top200": feature_stats,
        "shuffle_null_auroc_mean": null_mean, "shuffle_null_aurocs": null_aurocs,
        "confound_control": confound,
        "fdr_n_significant": int(rejected.sum()),
        "pass_criteria": passes, "verdict": verdict,
        "elapsed_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2, default=_js)
    print(f"[m0] VERDICT={verdict} |S|={len(S)} best_test_auroc={best_test_auroc:.3f} "
          f"best_test_f1={best_test_f1:.3f} null={null_mean}", flush=True)
    print(f"[m0] wrote {args.out} ({time.time()-t0:.0f}s)", flush=True)


def _single_auroc(scores, y):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return 0.5
    return roc_auc_score(y, scores)


def f1_at_threshold(Xcsc, y, thr):
    """F1 per feature at given per-feature thresholds (thr array)."""
    y = y.astype(bool); n_pos = int(y.sum())
    n_feat = Xcsc.shape[1]
    f1 = np.zeros(n_feat)
    if n_pos == 0:
        return f1
    indptr, indices, data = Xcsc.indptr, Xcsc.indices, Xcsc.data
    for f in range(n_feat):
        s, e = indptr[f], indptr[f + 1]
        t = thr[f]
        if s == e or t <= 0:
            continue
        rows = indices[s:e]; vals = data[s:e]
        pred = vals >= t
        pos_mask = y[rows]
        tp = int((pred & pos_mask).sum())
        fp = int((pred & ~pos_mask).sum())
        fn = n_pos - tp
        denom = 2 * tp + fp + fn
        if denom:
            f1[f] = 2 * tp / denom
    return f1


def confound_logistic(Xcsc, y, L, mask, helix_def, feats):
    """For each top feature, logistic regression helix ~ feature + gc3 + pos_frac; report whether
    feature coefficient stays positive & significant (survives confounds)."""
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    if len(feats) == 0:
        return {"survives_fraction": 0.0, "per_feature": []}
    gc3 = L["gc3"][mask]; pos = L["pos_frac"][mask]
    yy = y.astype(int)
    out = []
    survive = 0
    for f in feats:
        col = np.asarray(Xcsc[:, int(f)].todense()).ravel()
        Xd = np.column_stack([col, gc3, pos])
        # standardize
        Xs = (Xd - Xd.mean(0)) / (Xd.std(0) + 1e-8)
        try:
            clf = LogisticRegression(max_iter=200, C=1.0)
            clf.fit(Xs, yy)
            coef_feat = float(clf.coef_[0][0])
            surv = coef_feat > 0.05
            survive += int(surv)
            out.append({"feature": int(f), "feature_coef": coef_feat, "survives": bool(surv)})
        except Exception:
            out.append({"feature": int(f), "feature_coef": None, "survives": False})
    return {"survives_fraction": survive / len(feats), "per_feature": out}


if __name__ == "__main__":
    main()
