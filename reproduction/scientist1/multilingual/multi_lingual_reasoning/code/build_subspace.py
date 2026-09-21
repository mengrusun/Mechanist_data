"""Identify the language-specific subspace at each layer and evaluate Claim 1.

Approach: for each layer L,
  - Compute per-language mean hidden state (over the same set of parallel probes).
  - Center by the global mean of language means, then run PCA on the (n_lang, d) matrix.
  - Top-k principal components form the language subspace basis P_L (d, k).
  - Language identity decodability: nearest-mean classifier on centered probes.
  - Reasoning content check: after projecting out the subspace, verify that the
    'question-identity' (i.e. which of the N parallel problems it is) is still
    linearly separable across languages, which is the language-agnostic content.

Outputs:
  cache/subspace.npz: bases per layer (d, k) plus global mean c_L (d,)
  results/subspace_stats.json: per-layer diagnostics
"""
import os, sys, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import LANGS

CACHE = "cache/hidden_states.npz"
OUT_BASES = "cache/subspace.npz"
OUT_STATS = "results/subspace_stats.json"

K_LANG = 10  # dimension of language subspace (<= n_lang-1 = 10)


def load_hs():
    npz = np.load(CACHE)
    return {lg: npz[f"h_{lg}"].astype(np.float32) for lg in LANGS}


def per_layer_stats(H_by_lang):
    """H_by_lang[lang] -> (N, L+1, D). Returns per-layer diagnostics + bases."""
    N = H_by_lang["en"].shape[0]
    L1 = H_by_lang["en"].shape[1]        # layers + 1 (incl. embed)
    D  = H_by_lang["en"].shape[2]

    langs = LANGS
    n_lang = len(langs)

    bases = np.zeros((L1, D, K_LANG), dtype=np.float32)
    centres = np.zeros((L1, D), dtype=np.float32)
    stats = []
    for layer in range(L1):
        # Stack all probes per language into a big matrix
        X = np.stack([H_by_lang[lg][:, layer] for lg in langs], axis=0)  # (n_lang, N, D)
        mu_lang = X.mean(axis=1)                                          # (n_lang, D)
        c = mu_lang.mean(axis=0)                                          # (D,) global mean

        # PCA on centred language means
        M = mu_lang - c
        # SVD
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        # Language subspace = top-K principal directions (rows of Vt).
        k = min(K_LANG, S.shape[0])
        P = Vt[:k].T                                                      # (D, k)
        var_expl = float(S[:k].sum() / max(S.sum(), 1e-8))

        # Decodability: nearest-lang-mean classifier on centred probes
        Xc = X.reshape(n_lang * N, D) - c[None]                            # (n_lang*N, D)
        y  = np.repeat(np.arange(n_lang), N)                               # (n_lang*N,)
        # In full space
        dists = np.linalg.norm(Xc[:, None, :] - (mu_lang - c)[None, :, :], axis=-1)
        pred_full = dists.argmin(axis=1)
        acc_full = float((pred_full == y).mean())

        # After removing the language subspace (h -> h - P P^T h)
        Xr = Xc - Xc @ P @ P.T
        mu_r = mu_lang - c - (mu_lang - c) @ P @ P.T
        dists_r = np.linalg.norm(Xr[:, None, :] - mu_r[None, :, :], axis=-1)
        pred_res = dists_r.argmin(axis=1)
        acc_res = float((pred_res == y).mean())

        # Norm ratio: what fraction of centred probe norm lies in P?
        proj = Xc @ P                                                     # (n, k)
        frac_norm = float((proj ** 2).sum() / max((Xc ** 2).sum(), 1e-8))

        # "Reasoning" separability proxy: for each language, we have N parallel probes,
        # i.e. probe i in every language is the same question. Compute the mean of probe i
        # ACROSS languages -> (N, D). This is the language-agnostic 'content' direction.
        # We check if it survives after subspace removal and if it collapses in P.
        mu_probe = X.mean(axis=0)                                         # (N, D)
        Mp = mu_probe - c                                                 # (N, D)
        # Total variance of Mp
        v_probe_total = float((Mp ** 2).sum())
        v_probe_in_P  = float(((Mp @ P) ** 2).sum())
        v_probe_out_P = v_probe_total - v_probe_in_P

        stats.append({
            "layer": layer,
            "svals": [float(x) for x in S[:k].tolist()],
            "top_k_var_explained_of_lang_means": var_expl,
            "lang_id_acc_full": acc_full,
            "lang_id_acc_after_removal": acc_res,
            "frac_probe_norm_in_lang_subspace": frac_norm,
            "content_var_in_P_over_total": v_probe_in_P / max(v_probe_total, 1e-8),
            "content_var_out_P_over_total": v_probe_out_P / max(v_probe_total, 1e-8),
        })

        bases[layer]  = P
        centres[layer] = c

    return bases, centres, stats


def main():
    H = load_hs()
    print("H shapes:", {k: v.shape for k, v in list(H.items())[:1]})
    bases, centres, stats = per_layer_stats(H)
    np.savez_compressed(OUT_BASES, bases=bases.astype(np.float16), centres=centres.astype(np.float16))
    os.makedirs(os.path.dirname(OUT_STATS), exist_ok=True)
    with open(OUT_STATS, "w") as f:
        json.dump(stats, f, indent=2)
    print("wrote", OUT_BASES, OUT_STATS)
    # print a compact summary
    for s in stats:
        if s["layer"] % 4 == 0 or s["layer"] == len(stats) - 1:
            print(f"layer {s['layer']:2d} | "
                  f"lang-acc full/after {s['lang_id_acc_full']:.2f}/"
                  f"{s['lang_id_acc_after_removal']:.2f} | "
                  f"varExpl(topK)={s['top_k_var_explained_of_lang_means']:.2f} | "
                  f"probeNorm in P={s['frac_probe_norm_in_lang_subspace']:.3f} | "
                  f"content in P={s['content_var_in_P_over_total']:.3f}")


if __name__ == "__main__":
    main()
