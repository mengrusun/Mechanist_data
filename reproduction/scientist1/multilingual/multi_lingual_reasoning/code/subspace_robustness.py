"""Check whether the language subspace can be estimated from a small probe set (Claim 1).

We resample probe sets of increasing size N in {2,4,8,16,32,64} and, for each layer,
compare the estimated subspace against a "ground truth" subspace estimated from all 64
probes. Similarity measured by principal angle (subspace alignment).

Output: results/subspace_robustness.json
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import LANGS

CACHE = "cache/hidden_states.npz"
K = 10
SIZES = [2, 4, 8, 16, 32]
LAYERS_SAMPLE = [4, 8, 12, 16, 20, 24, 28, 32]
N_REPEATS = 5


def compute_basis(H_by_lang, indices, layer, k=K):
    """H_by_lang[lang]: (Nfull, L+1, D). Returns (D, k) basis at layer for given probe indices."""
    mus = []
    for lg in LANGS:
        h = H_by_lang[lg][indices, layer]      # (n, D)
        mus.append(h.astype(np.float32).mean(axis=0))
    M = np.stack(mus, axis=0)                    # (n_lang, D)
    c = M.mean(axis=0)
    U, S, Vt = np.linalg.svd(M - c[None], full_matrices=False)
    return Vt[:k].T, c                            # (D, k), (D,)


def principal_angle_similarity(P1, P2):
    """Return the mean cos(principal angle) — 1.0 means perfect alignment."""
    Q1, _ = np.linalg.qr(P1)
    Q2, _ = np.linalg.qr(P2)
    U, S, Vt = np.linalg.svd(Q1.T @ Q2, full_matrices=False)
    return float(S.mean())


def main():
    npz = np.load(CACHE)
    H = {lg: npz[f"h_{lg}"] for lg in LANGS}
    Nfull = H["en"].shape[0]
    print(f"Nfull={Nfull}")

    # Reference subspace = uses ALL probes
    ref_P = {}
    for L in LAYERS_SAMPLE:
        P, _ = compute_basis(H, np.arange(Nfull), L)
        ref_P[L] = P

    rng = np.random.default_rng(0)
    results = {}
    for L in LAYERS_SAMPLE:
        results[str(L)] = {}
        for n in SIZES:
            sims = []
            for _ in range(N_REPEATS):
                idx = rng.choice(Nfull, size=n, replace=False)
                Pn, _ = compute_basis(H, idx, L, k=min(K, len(LANGS) - 1))
                Pref = ref_P[L][:, :Pn.shape[1]]
                sims.append(principal_angle_similarity(Pn, Pref))
            results[str(L)][str(n)] = {"mean_sim": float(np.mean(sims)),
                                        "std_sim": float(np.std(sims))}
            print(f"L={L} n={n} mean cos(angle) = {np.mean(sims):.3f} ± {np.std(sims):.3f}")
    os.makedirs("results", exist_ok=True)
    with open("results/subspace_robustness.json", "w") as f:
        json.dump({"reference_n": Nfull, "sizes": SIZES,
                   "n_repeats": N_REPEATS, "results": results}, f, indent=2)
    print("wrote results/subspace_robustness.json")


if __name__ == "__main__":
    main()
