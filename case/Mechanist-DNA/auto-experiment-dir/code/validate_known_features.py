"""
Decisive SAE-validity + site/normalization check: does my pipeline reproduce the Evo2 paper's
named Layer-26 features?  alpha-helix = f/28741, beta-sheet = f/22326.

For each (site, normalize) config, cache per-codon feature activations on a small labeled set and
report: AUROC of f/28741 for helix(HGI), AUROC of f/22326 for sheet(E), and the top-ranked
helix/sheet features. The config that makes the paper's features top discriminators is the
correct residual site + SAE input normalization, and proves the SAE is valid on Evo2-7B L26.
"""
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, "code")
import m0_data as D
from evo2_sae import load_evo2, BatchTopKSAE, tokenize_seq, SAE_DICT
from sklearn.metrics import roc_auc_score

ALPHA_FEAT = 28741
BETA_FEAT = 22326
CONFIGS = [
    ("blocks.26", "unit_sqrtd"),
    ("blocks.26.post_norm", "none"),
    ("blocks.26.pre_norm", "none"),
    ("blocks.26.mlp.l3", "none"),
    ("blocks.26", "none"),
]


def load_labeled(org, n):
    path = os.path.join(D.DATA_DIR, f"m0_dataset_{org}.jsonl")
    recs = []
    for line in open(path):
        try:
            recs.append(json.loads(line))
        except Exception:
            pass
        if len(recs) >= n:
            break
    return recs


def cache_config(model, sae, site, recs):
    """Return per-codon activation matrix (dense np), helix labels, sheet labels."""
    feats = []; helix = []; sheet = []
    for rec in recs:
        n_cod = rec["n_codons"]; nt = rec["cds"][: n_cod * 3]
        if len(nt) < 30 or len(nt) > 8000:
            continue
        tok = tokenize_seq(model, nt, "cuda:0")
        with torch.no_grad():
            out = model(tok, return_embeddings=True, layer_names=[site])
        resid = out[1][site].float()[0]
        if resid.shape[0] != len(nt):
            continue
        z = sae.encode(resid)  # (L,32768)
        Lc = resid.shape[0] // 3
        z = z[:Lc*3].reshape(Lc, 3, SAE_DICT).mean(1).cpu().numpy()
        for c in range(min(Lc, n_cod)):
            ss = rec["codon_ss"][c]
            if ss is None:
                continue
            feats.append(z[c]); helix.append(1 if ss in D.HELIX_HGI else 0)
            sheet.append(1 if ss in D.SHEET else 0)
    return np.array(feats), np.array(helix), np.array(sheet)


def main():
    org = sys.argv[1] if len(sys.argv) > 1 else "prokaryote"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    recs = load_labeled(org, n)
    print(f"labeled proteins: {len(recs)}", flush=True)
    model = load_evo2("cuda:0")
    report = {}
    for site, norm in CONFIGS:
        sae = BatchTopKSAE(device="cuda:0", normalize=norm)
        X, yh, ys = cache_config(model, sae, site, recs)
        if X.shape[0] < 100 or yh.sum() < 5 or ys.sum() < 5:
            print(f"[{site},{norm}] too few labeled codons ({X.shape})", flush=True)
            continue
        # AUROC per feature (dense, feasible for a small codon set)
        def col_auroc(y):
            aur = np.full(SAE_DICT, 0.5)
            active = np.nonzero(X.any(0))[0]
            for f in active:
                col = X[:, f]
                if len(np.unique(y)) == 2:
                    aur[f] = roc_auc_score(y, col)
            return aur
        aur_h = col_auroc(yh); aur_s = col_auroc(ys)
        top_h = np.argsort(-aur_h)[:8]
        top_s = np.argsort(-aur_s)[:8]
        report[f"{site}|{norm}"] = {
            "n_codons": int(X.shape[0]), "helix_frac": float(yh.mean()), "sheet_frac": float(ys.mean()),
            "f28741_helix_auroc": float(aur_h[ALPHA_FEAT]), "f28741_helix_rank": int((aur_h > aur_h[ALPHA_FEAT]).sum()),
            "f22326_sheet_auroc": float(aur_s[BETA_FEAT]), "f22326_sheet_rank": int((aur_s > aur_s[BETA_FEAT]).sum()),
            "f28741_active": bool(X[:, ALPHA_FEAT].any()), "f22326_active": bool(X[:, BETA_FEAT].any()),
            "top_helix_feats": [int(f) for f in top_h], "top_helix_auroc": [round(float(aur_h[f]),3) for f in top_h],
            "top_sheet_feats": [int(f) for f in top_s], "top_sheet_auroc": [round(float(aur_s[f]),3) for f in top_s],
        }
        r = report[f"{site}|{norm}"]
        print(f"\n[{site} | {norm}] codons={r['n_codons']} helix={r['helix_frac']:.2f} sheet={r['sheet_frac']:.2f}", flush=True)
        print(f"  f/28741 (alpha) helix AUROC={r['f28741_helix_auroc']:.3f} rank={r['f28741_helix_rank']} active={r['f28741_active']}", flush=True)
        print(f"  f/22326 (beta)  sheet AUROC={r['f22326_sheet_auroc']:.3f} rank={r['f22326_sheet_rank']} active={r['f22326_active']}", flush=True)
        print(f"  top helix feats: {r['top_helix_feats']} auroc {r['top_helix_auroc']}", flush=True)
        print(f"  top sheet feats: {r['top_sheet_feats']} auroc {r['top_sheet_auroc']}", flush=True)
    json.dump(report, open("results/validate_known_features.json", "w"), indent=2)
    print("\nwrote results/validate_known_features.json", flush=True)


if __name__ == "__main__":
    main()
