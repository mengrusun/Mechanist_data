"""RFM concept-vector extraction + per-block linear probe screen.

Implements:
- LinearProbe: sklearn-style one-vs-rest ridge classification, fit per block.
- extract_rfm: kernel-ridge + AGOP alternation; returns unit-norm top eigenvector.
- caa_mean_diff: baseline mean-difference direction (for cosine-sim audit).

All numerical work is in torch/numpy on CPU (RFM at n<=400 with d=4096 is cheap).
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


# -----------------------------------------------------------------------------
# Per-block linear probe (used for both Location screen and C5 monitoring)
# -----------------------------------------------------------------------------

def fit_linear_probe(X: np.ndarray, y: np.ndarray, ridge: float = 1e-2) -> tuple[np.ndarray, float]:
    """L2-regularized least-squares logistic-like classifier with an intercept.

    Returns (w, b) where sign(X @ w + b) predicts y ∈ {-1,+1}.
    Uses closed-form ridge on augmented [X, 1] with labels ±1.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, d = X.shape
    # Augment with a bias column
    Xa = np.hstack([X, np.ones((n, 1))])
    reg = ridge * np.eye(d + 1, dtype=np.float64)
    reg[-1, -1] = 0.0  # don't regularise the intercept
    A = Xa.T @ Xa + reg
    b = Xa.T @ y
    w_full = np.linalg.solve(A, b)
    return w_full[:d], float(w_full[-1])


def probe_accuracy(w, X: np.ndarray, y: np.ndarray) -> float:
    """`w` may be either a (d,) vector or (w_vec, bias)."""
    if isinstance(w, tuple):
        wv, bb = w
    else:
        wv, bb = w, 0.0
    preds = np.sign(X @ wv + bb)
    preds[preds == 0] = 1.0
    return float((preds == y).mean())


def probe_auroc(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """AUROC of raw score X @ w against y ∈ {0,1} or {-1,+1}. Only requires ordering."""
    scores = X @ w
    y = np.asarray(y).reshape(-1)
    # Normalise labels to {0,1}
    if set(np.unique(y).tolist()) == {-1.0, 1.0}:
        y = (y > 0).astype(np.int64)
    else:
        y = y.astype(np.int64)
    order = np.argsort(-scores)
    y_sorted = y[order]
    npos = int(y_sorted.sum())
    nneg = len(y_sorted) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    # Rank-sum formula: AUROC = (sum of positive ranks - npos*(npos+1)/2) / (npos * nneg)
    ranks = np.arange(1, len(y_sorted) + 1)
    # Convert descending sort to ranks where smallest score = rank 1
    order_asc = np.argsort(scores)
    ranks_asc = np.empty_like(order_asc, dtype=np.float64)
    ranks_asc[order_asc] = np.arange(1, len(scores) + 1)
    pos_rank_sum = ranks_asc[y == 1].sum()
    return float((pos_rank_sum - npos * (npos + 1) / 2) / (npos * nneg))


# -----------------------------------------------------------------------------
# RFM (Recursive Feature Machines) — kernel + AGOP reweighting
# -----------------------------------------------------------------------------

def _rbf_kernel(X: np.ndarray, Z: np.ndarray, MX: np.ndarray, MZ: np.ndarray | None,
                bandwidth: float, cross_MX_Z: np.ndarray | None = None) -> np.ndarray:
    """Kernel k(x, z) = exp(-((x-z)^T M (x-z)) / bandwidth).

    Uses (x-z)^T M (x-z) = x^T M x + z^T M z - 2 x^T M z, so we never build the
    (n, m, d) `diff` tensor. Caller passes precomputed M X^T etc.

    - MX: shape (n,) — diagonal x^T M x
    - MZ: shape (m,) — diagonal z^T M z (or None if same as MX)
    - cross_MX_Z: (n, m) = X @ M @ Z^T (precomputed to avoid recomputing)
    """
    if MZ is None:
        MZ = MX
    quad = MX[:, None] + MZ[None, :] - 2.0 * cross_MX_Z  # (n, m)
    # Clip tiny negatives from FP round-off
    quad = np.maximum(quad, 0.0)
    return np.exp(-quad / bandwidth)


def _kernel_all(X: np.ndarray, M: np.ndarray, bandwidth: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (K, MX_diag) for the symmetric case (Z=X)."""
    # X @ M @ X.T — this is (n, n) — cheap.
    MXt = M @ X.T  # (d, n)
    cross = X @ MXt  # (n, n)
    MX_diag = np.diag(cross).copy()
    K = _rbf_kernel(X, X, MX_diag, MX_diag, bandwidth, cross_MX_Z=cross)
    return K, MX_diag


def _compute_agop(X: np.ndarray, alphas: np.ndarray, M: np.ndarray, bandwidth: float) -> np.ndarray:
    """Compute AGOP for the RBF-kernel predictor f(x) = sum_i alpha_i k(x, x_i).

    ∇f(x) = sum_i alpha_i * (-2/bandwidth) * M (x - x_i) * k(x, x_i)

    Memory-efficient reformulation (no (n, n, d) intermediate):
      grads[j] = (-2/bw) * M @ (sum_i alpha_i K[j,i] * (x_j - x_i))
              = (-2/bw) * M @ (x_j * (c row sum) - X.T @ c_row)
      where c[j,i] = alpha_i * K[j,i]

    Returns d x d matrix.
    """
    n, d = X.shape
    K, MX_diag = _kernel_all(X, M, bandwidth)
    # c[j,i] = alpha_i * K[j,i]
    c = K * alphas[None, :]  # (n, n)
    c_rowsum = c.sum(axis=1, keepdims=True)  # (n, 1)
    # inner[j] = x_j * c_rowsum[j] - X.T @ c[j]   (per-row)
    # Vectorised: inner = X * c_rowsum - c @ X   → (n, d)
    inner = X * c_rowsum - c @ X  # (n, d)
    # grads[j] = (-2/bw) * M @ inner[j]  →  grads = (-2/bw) * inner @ M  (since M symmetric)
    grads = (-2.0 / bandwidth) * (inner @ M)  # (n, d)
    agop = grads.T @ grads / float(n)  # (d, d)
    return agop


def extract_rfm(
    X: np.ndarray,
    y: np.ndarray,
    n_iters: int = 5,
    ridge: float = 1e-2,
    bandwidth: float | None = None,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Run RFM alternation and return the top AGOP eigenvector as concept direction.

    X: (n, d) activations
    y: (n,) labels ∈ {-1,+1}

    Returns:
      dict with:
        v_c: (d,) unit-norm top eigenvector of final AGOP
        eigvals: top-5 eigenvalues of final AGOP
        top_ratio: eigvals[0] / mean(eigvals[1:])
        cos_history: per-iter cosine sim between successive top eigenvectors
        agop_final: (d, d) final AGOP
        alpha_final: (n,) kernel weights at last iter
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, d = X.shape
    if bandwidth is None:
        # Median pairwise squared distance as bandwidth heuristic.
        idx = rng.choice(n, size=min(n, 200), replace=False)
        pd = X[idx][:, None, :] - X[idx][None, :, :]
        sq = (pd * pd).sum(-1)
        bandwidth = float(np.median(sq[sq > 0])) if (sq > 0).any() else 1.0
        bandwidth = max(bandwidth, 1e-3)

    # Initialise M = I / d (equivalent to isotropic RBF).
    # For d=4096 we can't store d x d densely in float64 (128 MB) but that's fine at n<=400.
    M = np.eye(d, dtype=np.float64) / float(d)

    prev_v = None
    cos_history = []
    top_ratios = []
    eigvals = np.zeros(5)
    for it in range(n_iters):
        # Fit kernel ridge: (K + λ I) α = y
        K, _ = _kernel_all(X, M, bandwidth)
        A = K + ridge * np.eye(n)
        alpha = np.linalg.solve(A, y)
        # Compute AGOP (d x d — heavy but tractable at d=4096, ~128 MB fp64)
        agop = _compute_agop(X, alpha, M, bandwidth)
        agop = (agop + agop.T) * 0.5  # symmetrise
        # Top eigenvector: for d=4096, eigh is ~20-30s. Use scipy.sparse.linalg.eigsh k=5
        try:
            from scipy.sparse.linalg import eigsh
            eigvals_top, eigvecs_top = eigsh(agop, k=min(5, d-1), which="LA")
            # Ascending — flip
            order = np.argsort(-eigvals_top)
            eigvals_top = eigvals_top[order]
            eigvecs_top = eigvecs_top[:, order]
            v_top = eigvecs_top[:, 0]
            eigvals = eigvals_top
        except Exception:
            eigvals_full, eigvecs_full = np.linalg.eigh(agop)
            eigvals_full = eigvals_full[::-1]; eigvecs_full = eigvecs_full[:, ::-1]
            v_top = eigvecs_full[:, 0]; eigvals = eigvals_full[:5]

        # Normalise
        v_top = v_top / (np.linalg.norm(v_top) + 1e-12)

        top_ratio = float(eigvals[0] / max(eigvals[1:5].mean(), 1e-12))
        top_ratios.append(top_ratio)
        if prev_v is not None:
            cos = float(abs(prev_v @ v_top))
            cos_history.append(cos)
        prev_v = v_top

        # Update M for next iteration (renormalise scale)
        M = agop / max(np.trace(agop) / d, 1e-12)

        if verbose:
            print(f"  RFM iter {it+1}/{n_iters}: top_eigval_ratio={top_ratio:.3f}  "
                  f"cos_prev={cos_history[-1] if cos_history else float('nan'):.4f}")

    # Align sign so that positive-labeled mean has positive projection
    proj = X @ v_top
    if (proj * y).mean() < 0:
        v_top = -v_top

    return {
        "v_c": v_top.astype(np.float32),
        "eigvals_top5": eigvals[:5].astype(np.float32),
        "top_ratio": top_ratios[-1],
        "cos_history": cos_history,
        "bandwidth": bandwidth,
        "alpha_final": alpha.astype(np.float32),
    }


def caa_mean_diff(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Contrastive Activation Addition baseline: mean(X | y=+1) - mean(X | y=-1)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    pos = X[y > 0].mean(axis=0)
    neg = X[y <= 0].mean(axis=0)
    v = pos - neg
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.astype(np.float32)


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def save_vector(path: Path, v: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, v.astype(np.float32))
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def load_vector(path: Path) -> Tuple[np.ndarray, dict]:
    v = np.load(path)
    with open(path.with_suffix(".json")) as f:
        meta = json.load(f)
    return v, meta
