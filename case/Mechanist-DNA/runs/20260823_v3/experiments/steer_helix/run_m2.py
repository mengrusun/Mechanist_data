"""M2: localize the alpha-helix representation (Location, correlational screen).

- Per-block linear (ridge) probes predicting the ORF alpha-helix fraction from mean-pooled
  residual activations; rank candidate blocks by held-out R^2 (train/val inside DEV only).
- Screen the Goodfire Evo-2 Layer-26 SAE features for helix selectivity; shortlist a feature SET.

Outputs top1/top2 steering sites + a SAE feature shortlist for M3. DEV split only (anti-circularity).
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import common as C

def ridge_r2(X, y, l2=10.0, val_frac=0.3, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y)); cut = int(len(y) * (1 - val_frac))
    tr, va = idx[:cut], idx[cut:]
    Xtr, Xva = X[tr], X[va]
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xva = (Xva - mu) / sd
    A = Xtr.T @ Xtr + l2 * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ (y[tr] - y[tr].mean()))
    pred = Xva @ w + y[tr].mean()
    ss_res = ((y[va] - pred) ** 2).sum(); ss_tot = ((y[va] - y[va].mean()) ** 2).sum() + 1e-9
    return 1 - ss_res / ss_tot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1_out", default=os.path.join(C.PROJECT, "results/m1"))
    ap.add_argument("--out", default=os.path.join(C.PROJECT, "results/m2"))
    ap.add_argument("--candidate_blocks", default="12,14,16,18,20,22,24,26,28,30")
    ap.add_argument("--sae_block", type=int, default=26)
    ap.add_argument("--n_sae_feats", type=int, default=16)
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args()
    if args.gpu: os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out, exist_ok=True)

    recs = json.load(open(os.path.join(args.m1_out, "baseline_records.json")))
    dev = [r for r in recs if r["split"] == "DEV" and r["fold_ok"] and r["protein"]]
    seqs = [r["dna"] for r in dev]
    y = np.array([r["helix_frac"] for r in dev], dtype=np.float32)
    blocks = [int(b) for b in args.candidate_blocks.split(",")]

    evo = C.Evo2Wrapper("evo2_7b")
    acts = evo.block_activations(seqs, blocks)   # {block: [n, 4096]}

    ranking = []
    for b in blocks:
        r2 = float(ridge_r2(acts[b].astype(np.float32), y))
        ranking.append(dict(block=b, r2=r2))
    ranking.sort(key=lambda d: d["r2"], reverse=True)
    top = [d["block"] for d in ranking[:2]]

    # ---- SAE feature screen at block 26 ----
    sae_shortlist = []
    try:
        sae = C.TopKSAE()
        import torch
        A = torch.tensor(acts[args.sae_block].astype(np.float32)).cuda()
        F = sae.encode(A).cpu().numpy()            # [n, n_features]
        # correlate each feature activation with helix fraction
        fy = (y - y.mean())
        num = (F - F.mean(0)) * fy[:, None]
        corr = num.sum(0) / (F.std(0) * fy.std() * len(y) + 1e-9)
        order = np.argsort(-np.abs(corr))[:args.n_sae_feats]
        sae_shortlist = [dict(feature=int(i), corr=float(corr[i])) for i in order]
    except Exception as e:
        sae_shortlist = [{"error": str(e)[:120]}]

    out = dict(probe_ranking=ranking, top_sites=top,
               sae_block=args.sae_block, sae_shortlist=sae_shortlist,
               n_dev=len(dev))
    C.save_json(out, os.path.join(args.out, "localization.json"))
    print("[M2] probe ranking:", ranking)
    print("[M2] top steering sites:", top)
    print("[M2] SAE shortlist (top-5):", sae_shortlist[:5])

if __name__ == "__main__":
    main()
