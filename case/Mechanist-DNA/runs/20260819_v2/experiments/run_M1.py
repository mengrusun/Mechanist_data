"""
M1: Locate candidate alpha-helical-propensity directions (Location / screen).

Three extractors on the contrastive high-alpha vs low-alpha CDS windows:
  A. SAE feature (Goodfire L26 BatchTopK) -- rank 32768 features by class separation.
  B. Contrastive mean-difference vector at spaced blocks {14,18,20,22,24,26,28}.
  C. Linear probe (logistic) weight direction at the same blocks.

Directions are FIT on the train split and RANKED by AUROC on the disjoint held-out split.
Outputs: results/M1_candidate_directions.json ; assets/directions.npz (vectors for M2).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import evo2lib as E

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
ASSETS = os.path.abspath(os.path.join(HERE, "..", "assets"))
BLOCKS = [14, 18, 20, 22, 24, 26, 28]
SAE_BLOCK = 26

def auroc(y, s):
    try: return float(roc_auc_score(y, s))
    except Exception: return float('nan')

@torch.no_grad()
def main():
    t0 = time.time()
    E.set_seed(42)
    windows = json.load(open(os.path.join(DATA, "contrastive.json")))
    windows = [w for w in windows if w["label"] in ("high_alpha", "low_alpha")]
    m = E.load_evo2()
    print(f"[M1] evo2 loaded; {len(windows)} windows", flush=True)

    have_sae = os.path.exists(os.path.join(ASSETS, ".sae_dl_done"))
    sae = None
    if have_sae:
        try:
            sae = E.SAEClamp()
            print(f"[M1] SAE loaded: n_feat={sae.n_feat} d={sae.d_model} keys={sae.raw_keys}", flush=True)
        except Exception as ex:
            print(f"[M1] SAE load failed: {ex}", flush=True); sae = None; have_sae = False

    names = [E.block_name(b) for b in BLOCKS]
    pooled = {b: [] for b in BLOCKS}
    sae_feat = []  # mean feature activation per window at block 26
    labels = []; splits = []
    for i, w in enumerate(windows):
        ids = E.tokenize(m, w["dna"])
        _, emb = m.forward(ids, return_embeddings=True, layer_names=names)
        for b in BLOCKS:
            pooled[b].append(emb[E.block_name(b)][0].float().mean(0).cpu().numpy())
        if sae is not None:
            h26 = emb[E.block_name(SAE_BLOCK)][0].float()  # [L,d]
            fa = sae.feature_activations(h26).mean(0)       # [n_feat] mean over tokens
            sae_feat.append(fa.cpu().numpy())
        labels.append(1 if w["label"] == "high_alpha" else 0)
        splits.append(w["split"])
        if (i+1) % 100 == 0:
            print(f"[M1] activations {i+1}/{len(windows)}", flush=True)
    labels = np.array(labels); splits = np.array(splits)
    tr = splits == "train"; ho = splits == "heldout"
    print(f"[M1] train={tr.sum()} heldout={ho.sum()} pos_frac={labels.mean():.2f}", flush=True)

    candidates = []
    directions = {}  # name -> vector (d_model,)
    scales = {}      # block -> mean act norm

    for b in BLOCKS:
        X = np.stack(pooled[b])
        scales[b] = float(np.linalg.norm(X, axis=1).mean())
        Xtr, ytr = X[tr], labels[tr]; Xho, yho = X[ho], labels[ho]
        # B: contrastive mean-difference
        v = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
        proj_ho = Xho @ v
        a_b = auroc(yho, proj_ho)
        directions[f"contrastive_b{b}"] = v.astype(np.float32)
        candidates.append({"family": "contrastive_vector", "block": b,
                           "auroc_heldout": round(a_b, 4), "norm": float(np.linalg.norm(v)),
                           "scale_meanactnorm": round(scales[b], 3),
                           "key": f"contrastive_b{b}"})
        # C: logistic probe
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, ytr)
        w_dir = clf.coef_[0]
        proj_probe = Xho @ w_dir
        a_p = auroc(yho, proj_probe)
        # scale probe direction to contrastive norm for comparable steering
        w_unit = w_dir / (np.linalg.norm(w_dir) + 1e-8)
        w_scaled = (w_unit * np.linalg.norm(v)).astype(np.float32)
        directions[f"probe_b{b}"] = w_scaled
        candidates.append({"family": "linear_probe", "block": b,
                           "auroc_heldout": round(a_p, 4), "norm": float(np.linalg.norm(w_scaled)),
                           "scale_meanactnorm": round(scales[b], 3),
                           "key": f"probe_b{b}"})

    # A: SAE feature ranking (block 26)
    sae_result = None
    if sae is not None and len(sae_feat) > 0:
        F = np.stack(sae_feat)  # [N, n_feat]
        Ftr, Fho = F[tr], F[ho]
        ytr, yho = labels[tr], labels[ho]
        # rank by train mean-gap (high - low), normalized
        gap = Ftr[ytr == 1].mean(0) - Ftr[ytr == 0].mean(0)
        order = np.argsort(-np.abs(gap))
        top = []
        for idx in order[:10]:
            a = auroc(yho, Fho[:, idx])
            top.append({"feature": int(idx), "train_gap": round(float(gap[idx]), 4),
                        "auroc_heldout": round(a, 4),
                        "high_mean_act": round(float(Ftr[ytr==1][:, idx].mean()), 4)})
        # best by heldout auroc among top-gap
        best = max(top, key=lambda d: (d["auroc_heldout"] if d["auroc_heldout"]==d["auroc_heldout"] else 0))
        # direction = decoder column for that feature
        feat_dir = sae.W[best["feature"]].float().cpu().numpy().astype(np.float32)
        directions[f"sae_feat{best['feature']}_b26"] = feat_dir
        sae_result = {"top_features": top, "best_feature": best}
        candidates.append({"family": "sae_feature", "block": 26,
                           "feature": best["feature"], "auroc_heldout": best["auroc_heldout"],
                           "clamp_value_ref": best["high_mean_act"],
                           "key": f"sae_feat{best['feature']}_b26",
                           "scale_meanactnorm": round(scales[26], 3)})
        print(f"[M1] SAE best feature {best['feature']} auroc={best['auroc_heldout']}", flush=True)

    # rank all
    candidates_sorted = sorted(candidates, key=lambda c: (c["auroc_heldout"] if c["auroc_heldout"]==c["auroc_heldout"] else 0), reverse=True)
    np.savez(os.path.join(ASSETS, "directions.npz"),
             **{k: v for k, v in directions.items()},
             scales=np.array([scales[b] for b in BLOCKS]), blocks=np.array(BLOCKS))

    result = {
        "milestone": "M1", "claim": "C1", "role": "location_screen",
        "blocks_screened": BLOCKS, "sae_available": bool(sae is not None),
        "n_train": int(tr.sum()), "n_heldout": int(ho.sum()),
        "ranked_candidates": candidates_sorted,
        "top3": candidates_sorted[:3],
        "sae_detail": sae_result,
        "scales_meanactnorm": {str(b): round(scales[b], 3) for b in BLOCKS},
        "gpu_hours": round((time.time()-t0)/3600, 3),
    }
    json.dump(result, open(os.path.join(RES, "M1_candidate_directions.json"), "w"), indent=2)
    print("[M1] TOP3", json.dumps(candidates_sorted[:3]), flush=True)
    print("M1_DONE", flush=True)

if __name__ == "__main__":
    main()
