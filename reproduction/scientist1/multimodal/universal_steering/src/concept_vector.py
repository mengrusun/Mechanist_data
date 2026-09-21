"""Compute per-block concept vectors from activations.

We implement two methods:

1. Difference-of-means (DiffMean): per-block, v_l = mean(pos) - mean(neg).
   A well-known lightweight direction that captures the linear concept axis.

2. RFM (Recursive Feature Machine, following Radhakrishnan+2024): for each
   block we fit a Laplacian kernel regressor with a Mahalanobis metric M,
   then update M by the Average Gradient Outer Product (AGOP). We iterate a
   few times and take the top eigenvector of M as the concept direction.

The RFM implementation follows the standard AGOP formulation but with tiny
scale (only ~100 train points, 4096-d features), so it stays cheap.
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def diff_mean(acts, labels):
    """acts: (N, L, D)  labels: (N,) in {0,1}."""
    pos = acts[labels == 1]
    neg = acts[labels == 0]
    v = pos.mean(0) - neg.mean(0)  # (L, D)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    return v


def laplace_kernel(X1, X2, M, bandwidth):
    """Laplace-Mahalanobis kernel exp(-||x-y||_M / L)."""
    # Compute pairwise Mahalanobis distances via M^{1/2}.
    # For efficiency we assume M = U @ U^T low-rank not enforced; use full form.
    diff = X1[:, None, :] - X2[None, :, :]           # (n1, n2, D)
    # ||v||_M = sqrt(v^T M v)
    Mv = diff @ M                                    # (n1, n2, D)
    dist_sq = (Mv * diff).sum(-1).clamp(min=0)
    dist = torch.sqrt(dist_sq + 1e-12)
    return torch.exp(-dist / bandwidth)


def rfm_direction(X, y, iters=3, bandwidth=10.0, reg=1e-3):
    """RFM to get a concept direction for a single block.

    X: (N, D) float tensor
    y: (N,) float tensor in {0,1}
    Returns: (D,) direction vector (top eigvec of learned M).
    """
    device = X.device
    N, D = X.shape
    M = torch.eye(D, device=device, dtype=X.dtype)

    for _ in range(iters):
        K = laplace_kernel(X, X, M, bandwidth)             # (N, N)
        # solve (K + reg*I) a = y
        a = torch.linalg.solve(
            K + reg * torch.eye(N, device=device, dtype=X.dtype), y
        )                                                   # (N,)

        # AGOP: E_x [ grad_x f(x) grad_x f(x)^T ]
        # For Laplace kernel, grad_x K(x, x_i) = -K(x,x_i)/(||x-x_i||_M) * M (x - x_i)
        # We compute grad on the training points themselves.
        # grad f(x_j) = sum_i a_i * grad_x K(x_j, x_i)
        # -> shape (D,)
        # Build in a vectorized way per training point.
        diff = X[:, None, :] - X[None, :, :]                # (N,N,D)
        Mv = diff @ M                                       # (N,N,D)
        dist_sq = (Mv * diff).sum(-1).clamp(min=1e-12)
        dist = torch.sqrt(dist_sq)                          # (N,N)
        coef = -K / dist / bandwidth                        # (N,N)
        # grad of K wrt x_j w.r.t. anchor i: coef[j,i] * Mv[j,i]
        weighted = (a[None, :, None] * coef[:, :, None] * Mv)  # (N,N,D)
        grads = weighted.sum(dim=1)                         # (N, D)

        # AGOP = (1/N) * grads^T grads
        M_new = grads.T @ grads / N
        # normalize magnitude
        M_new = M_new / (M_new.norm() + 1e-12) * D
        M = M_new

    # Concept direction = top eigenvector of M
    # Use symmetric eig on CPU for stability.
    Mcpu = M.detach().cpu().double()
    evals, evecs = torch.linalg.eigh((Mcpu + Mcpu.T) / 2)
    v = evecs[:, -1]                                        # top eigenvector
    # Fix sign: dot with (mean pos - mean neg) so we point "positive"
    pos = X[y > 0.5].mean(0).cpu().double()
    neg = X[y < 0.5].mean(0).cpu().double()
    if (pos - neg) @ v < 0:
        v = -v
    v = v / (v.norm() + 1e-12)
    return v.to(X.dtype).to(device)


def compute_concept(acts, labels, method="diffmean", rfm_iters=3):
    """acts: (N, L, D), labels: (N,) long. Return (L, D) unit vectors."""
    if method == "diffmean":
        return diff_mean(acts, labels)

    if method == "rfm":
        L = acts.shape[1]
        y = labels.float().to(acts.device)
        vs = []
        for l in range(L):
            X = acts[:, l, :].contiguous()
            # Center to help kernel
            v = rfm_direction(X, y, iters=rfm_iters)
            vs.append(v)
        v = torch.stack(vs, dim=0)
        # unit normalize (already done)
        return v

    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="diffmean", choices=["diffmean", "rfm"])
    ap.add_argument("--rfm_iters", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    data = torch.load(args.acts, weights_only=False)
    acts = data["activations"].to(args.device)
    labels = data["labels"].to(args.device)
    print(f"Loaded {acts.shape}, labels distribution: {labels.bincount().tolist()}")

    v = compute_concept(acts, labels, args.method, args.rfm_iters)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"vectors": v.cpu(), "method": args.method, "model": data["model"]},
        args.out,
    )
    print(f"Saved concept vectors {v.shape} to {args.out}")


if __name__ == "__main__":
    main()
