"""Compute 'pure' variable directions by residualising each direction wrt
the other three (Gram-Schmidt in the linear span of the raw directions).

Report cosine similarities before/after: pure directions should have ~0
cosine to other pure directions."""

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT/"results"/"directions_llama"
VARS = ["gender", "age", "instruction", "meeting"]


def gram_schmidt_purify(D_mat):
    """D_mat: (V, H) — each row is a direction. Return (V, H) where each row is
    orthogonalised against the *other* rows via projection: subtract the
    component of d_v that lies in span(d_others). We do this per row using OLS
    projection onto the remaining rows and subtracting."""
    V, H = D_mat.shape
    P = np.zeros_like(D_mat)
    for v in range(V):
        others = np.delete(D_mat, v, axis=0)  # (V-1, H)
        # solve for coeffs a s.t. || d_v - a^T others ||^2 min in R^H
        # Note: with V-1 rows and H columns (H>>V-1), this is well-posed.
        # a = (O O^T)^{-1} O d_v
        A = others @ others.T  # (V-1, V-1)
        b = others @ D_mat[v]  # (V-1,)
        a = np.linalg.solve(A + 1e-6*np.eye(A.shape[0]), b)
        P[v] = D_mat[v] - a @ others
    return P


def cos(u, v):
    return (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)


def cos_matrix(D_mat):
    V = D_mat.shape[0]
    M = np.zeros((V, V))
    for i in range(V):
        for j in range(V):
            M[i,j] = cos(D_mat[i], D_mat[j])
    return M


def main():
    # load per-layer directions
    dirs = {v: np.load(D/f"dir_{v}.npy") for v in VARS}   # each (L+1, H)
    L = dirs[VARS[0]].shape[0]
    H = dirs[VARS[0]].shape[1]

    pure = {v: np.zeros_like(dirs[v]) for v in VARS}
    for li in range(L):
        Dmat = np.stack([dirs[v][li] for v in VARS])  # (4, H)
        Pmat = gram_schmidt_purify(Dmat)
        for i, v in enumerate(VARS):
            pure[v][li] = Pmat[i]

    for v in VARS:
        np.save(D/f"pure_{v}.npy", pure[v])

    # log summary at a mid-late layer, e.g. layer 16
    for li in [8, 16, 24, 30]:
        Dmat = np.stack([dirs[v][li] for v in VARS])
        Pmat = np.stack([pure[v][li] for v in VARS])
        print(f"\n=== layer {li} ===")
        print(" raw  cos:")
        C = cos_matrix(Dmat)
        for i, v in enumerate(VARS):
            print("  ", v, {VARS[j]: round(C[i,j],3) for j in range(4)})
        print(" pure cos:")
        C = cos_matrix(Pmat)
        for i, v in enumerate(VARS):
            print("  ", v, {VARS[j]: round(C[i,j],3) for j in range(4)})
        # per-var norm shrinkage
        print(" norm ratio pure/raw:")
        for i, v in enumerate(VARS):
            r = np.linalg.norm(Pmat[i]) / (np.linalg.norm(Dmat[i])+1e-9)
            print(f"   {v}: {r:.3f}")

    print("\nSaved pure_*.npy to", D)


if __name__ == "__main__":
    main()
