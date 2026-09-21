"""
Fast per-feature discrimination scoring for M0.

Given a sparse codon x feature activation matrix (CSC) and a binary label vector,
compute per-feature AUROC (threshold-free) and best-threshold F1 efficiently, exploiting
that each feature column is sparse (mostly zeros).
"""
import numpy as np
from scipy import sparse


def sparse_auroc_all(X_csc, y):
    """AUROC of every feature (column) as a predictor of y (binary), vectorized over sparsity.

    For each feature the score at a codon is its activation (>=0); absent -> 0.
    AUROC = P(score_pos > score_neg) + 0.5 P(tie). We compute it exactly accounting for
    the mass of zeros. Returns array (n_features,).
    """
    y = y.astype(bool)
    n_pos = int(y.sum()); n_neg = int((~y).sum())
    n_features = X_csc.shape[1]
    auroc = np.full(n_features, 0.5, dtype=np.float64)
    if n_pos == 0 or n_neg == 0:
        return auroc
    indptr, indices, data = X_csc.indptr, X_csc.indices, X_csc.data
    for f in range(n_features):
        s, e = indptr[f], indptr[f + 1]
        if s == e:
            continue  # all zeros -> AUROC 0.5
        rows = indices[s:e]
        vals = data[s:e]
        # nonzero values are all > 0 (ReLU activations). zeros are the rest.
        pos_mask = y[rows]
        vp = vals[pos_mask]      # positive nonzero values
        vn = vals[~pos_mask]     # negative nonzero values
        n_pos_nz = vp.size; n_neg_nz = vn.size
        n_pos_z = n_pos - n_pos_nz     # positives at value 0
        n_neg_z = n_neg - n_neg_nz     # negatives at value 0
        # count concordant pairs (pos value > neg value) + 0.5 ties
        # 1) both nonzero: compare vp vs vn via sorting
        conc = 0.0
        if n_pos_nz and n_neg_nz:
            vn_sorted = np.sort(vn)
            # for each vp, number of vn strictly less, and equal (ties)
            less = np.searchsorted(vn_sorted, vp, side="left")
            lesseq = np.searchsorted(vn_sorted, vp, side="right")
            ties = lesseq - less
            conc += less.sum() + 0.5 * ties.sum()
        # 2) pos nonzero (>0) vs neg zero: pos always wins
        conc += n_pos_nz * n_neg_z
        # 3) pos zero vs neg nonzero(>0): neg wins -> contributes 0
        # 4) pos zero vs neg zero: tie -> 0.5
        conc += 0.5 * n_pos_z * n_neg_z
        auroc[f] = conc / (n_pos * n_neg)
    return auroc


def sparse_best_f1_all(X_csc, y, n_thresh=32):
    """Best-threshold F1 per feature. Threshold candidates from the feature's own nonzero
    activation quantiles. Returns (f1 array, threshold array)."""
    y = y.astype(bool)
    n_pos = int(y.sum())
    n_features = X_csc.shape[1]
    f1 = np.zeros(n_features, dtype=np.float64)
    thr = np.zeros(n_features, dtype=np.float64)
    if n_pos == 0:
        return f1, thr
    indptr, indices, data = X_csc.indptr, X_csc.indices, X_csc.data
    for f in range(n_features):
        s, e = indptr[f], indptr[f + 1]
        if s == e:
            continue
        rows = indices[s:e]; vals = data[s:e]
        pos_mask = y[rows]
        # candidate thresholds: quantiles of nonzero values
        qs = np.quantile(vals, np.linspace(0.05, 0.95, n_thresh)) if vals.size >= 4 else np.unique(vals)
        best = 0.0; bthr = 0.0
        vp = vals[pos_mask]; vn = vals[~pos_mask]
        for t in np.unique(qs):
            tp = int((vp >= t).sum())
            fp = int((vn >= t).sum())
            fn = n_pos - tp
            denom = (2 * tp + fp + fn)
            if denom == 0:
                continue
            fval = 2 * tp / denom
            if fval > best:
                best = fval; bthr = t
        f1[f] = best; thr[f] = bthr
    return f1, thr


def bh_fdr(pvals, q=0.05):
    """Benjamini-Hochberg. Return boolean mask of rejected (significant) and adjusted p-values."""
    p = np.asarray(pvals)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj_full = np.empty(n); adj_full[order] = np.clip(adj, 0, 1)
    rejected = adj_full < q
    return rejected, adj_full


def auroc_pvalue(auroc, n_pos, n_neg):
    """Normal-approx p-value for AUROC != 0.5 (Mann-Whitney U)."""
    from scipy import stats
    mu = 0.5
    sigma = np.sqrt((n_pos + n_neg + 1) / (12.0 * n_pos * n_neg))
    z = (auroc - mu) / sigma
    # one-sided (feature marks helix -> auroc > 0.5)
    p = stats.norm.sf(z)
    return p
