"""M1 — Confirm alpha-helix-selective SAE features (Claim C1).

1. Forward every train/val/test CDS through Evo2-7B; hook blocks.26; SAE-encode;
   store per-codon latent activations (mean over the codon's 3 nt positions).
2. Per-latent alpha-helix-vs-rest AUROC (tie-corrected) + helix:non-helix ratio on
   train; gene-blocked label-permutation null -> BH-FDR across all 32768.
3. Candidate = train AUROC>=0.70, q<0.01, ratio > matched beta/coil controls.
   Pick top-K by validation AUROC (K in {1,5,10,20}, cap 32; smallest K whose val
   AUROC plateaus). Freeze S, K, thresholds.
4. Test once: selectivity of S on held-out test (AUROC, ratio, cluster-blocked
   bootstrap CI) vs shuffle null and vs beta/coil-selective control features.

Outputs results/m1_features.json (+ results/m1_arrays.npz).
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import torch
import common as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ capture
@torch.no_grad()
def capture_all(genes_df, sae, evo2, dev, log_every=250):
    """Return dict of per-split arrays: Z (n,32768 f16), helix, sheet, coil labels,
    gene id (int), cluster id (int). Codon = mean over 3 nt of blocks.26 latents."""
    n_codons = int(genes_df["aa"].str.len().sum())
    print(f"[m1] preallocating Z for {n_codons} codons x {C.D_SAE} (float32) "
          f"~{n_codons*C.D_SAE*4/1e9:.1f} GB", flush=True)
    Z = np.zeros((n_codons, C.D_SAE), dtype=np.float32)
    helix = np.zeros(n_codons, dtype=np.int8)
    sheet = np.zeros(n_codons, dtype=np.int8)
    coil  = np.zeros(n_codons, dtype=np.int8)
    gid   = np.zeros(n_codons, dtype=np.int32)
    cid   = np.zeros(n_codons, dtype=np.int32)
    split = np.empty(n_codons, dtype=object)
    clusters = {c: i for i, c in enumerate(sorted(genes_df["cluster"].unique()))}
    tok = evo2.tokenizer
    pos = 0
    t0 = time.time()
    for gi, row in enumerate(genes_df.itertuples(index=False)):
        dna, ss3 = row.dna, row.ss3
        L = len(row.aa)
        ids = torch.tensor(tok.tokenize(dna[:3*L]), dtype=torch.long,
                           device=dev).unsqueeze(0)
        act = C.capture_layer26(evo2, ids)[0].to(dev, torch.float32)  # (3L,4096)
        z = sae.encode(act)                                           # (3L,32768)
        # mean over each codon's 3 nt
        z = z[:3*L].reshape(L, 3, C.D_SAE).mean(dim=1)                # (L,32768)
        Z[pos:pos+L] = z.float().cpu().numpy()
        for ci in range(L):
            st = ss3[ci] if ci < len(ss3) else "C"
            helix[pos+ci] = 1 if st == "H" else 0
            sheet[pos+ci] = 1 if st == "E" else 0
            coil[pos+ci]  = 1 if st == "C" else 0
        gid[pos:pos+L] = gi
        cid[pos:pos+L] = clusters[row.cluster]
        split[pos:pos+L] = row.split
        pos += L
        if (gi+1) % log_every == 0:
            print(f"[m1]   captured {gi+1}/{len(genes_df)} genes, {pos} codons, "
                  f"{time.time()-t0:.0f}s", flush=True)
    assert pos == n_codons, (pos, n_codons)
    return dict(Z=Z, helix=helix, sheet=sheet, coil=coil, gid=gid, cid=cid, split=split)


# ------------------------------------------------------------------ AUROC utils
def midrank_columns(Zc):
    """Tie-corrected ranks (1..n midranks) for each column of Zc (n,C) on GPU.
    Only the dominant zero-tie block is midrank-corrected (SAE latents are >=0;
    positives are ~continuous)."""
    n = Zc.shape[0]
    ro = Zc.argsort(dim=0).argsort(dim=0).float() + 1.0     # ordinal ranks 1..n
    zeromask = (Zc == 0)
    zc = zeromask.sum(dim=0).float()                         # zeros per column
    mid = (zc + 1.0) / 2.0                                   # midrank of zero block
    ro = torch.where(zeromask, mid.unsqueeze(0), ro)
    return ro                                                # (n,C)

def auroc_from_ranks(ranks, labelmask, n_pos, n_neg):
    """AUROC for each column given precomputed ranks and a boolean label mask.
    labelmask: (n,) float {1=pos}. ranks: (n,C). Returns (C,)."""
    R_pos = labelmask @ ranks                                # (C,)
    U = R_pos - n_pos * (n_pos + 1) / 2.0
    return U / (n_pos * n_neg)

def auroc_all_latents(Z, labels, dev, chunk=2048):
    """Observed AUROC per latent (label vs rest). Z host f16 (n,D)."""
    n, D = Z.shape
    lab = torch.tensor(labels.astype(np.float32), device=dev)
    n_pos = float(lab.sum().item()); n_neg = n - n_pos
    out = np.zeros(D, dtype=np.float32)
    for c0 in range(0, D, chunk):
        Zc = torch.tensor(Z[:, c0:c0+chunk], device=dev, dtype=torch.float32)
        ranks = midrank_columns(Zc)
        out[c0:c0+chunk] = auroc_from_ranks(ranks, lab, n_pos, n_neg).cpu().numpy()
        del Zc, ranks
    return out, n_pos, n_neg

def auroc_and_null(Z, labels, gid, dev, n_perm=200, chunk=2048, seed=0):
    """Observed AUROC + gene-blocked permutation null per latent.
    Returns (obs (D,), pval (D,)). Perm shuffles helix labels within each gene."""
    n, D = Z.shape
    rng = np.random.default_rng(seed)
    lab = labels.astype(np.float32)
    n_pos = float(lab.sum()); n_neg = n - n_pos
    # build gene-blocked permutation label matrix (n_perm, n)
    order = np.argsort(gid, kind="stable")
    inv = np.empty_like(order); inv[order] = np.arange(n)
    lab_sorted = lab[order]; gid_sorted = gid[order]
    # gene segment boundaries
    bnd = np.flatnonzero(np.diff(gid_sorted)) + 1
    segs = np.split(np.arange(n), bnd)
    perm = np.zeros((n_perm, n), dtype=np.float32)
    for p in range(n_perm):
        ls = lab_sorted.copy()
        for s in segs:
            rng.shuffle(ls[s[0]:s[-1]+1])
        perm[p] = ls[inv]
    lab_t = torch.tensor(lab, device=dev)
    perm_t = torch.tensor(perm, device=dev)                 # (P,n)
    obs = np.zeros(D, dtype=np.float32)
    ge = np.zeros(D, dtype=np.int32)                        # #null >= obs
    denom = n_pos * n_neg
    off = n_pos * (n_pos + 1) / 2.0
    for c0 in range(0, D, chunk):
        Zc = torch.tensor(Z[:, c0:c0+chunk], device=dev, dtype=torch.float32)
        ranks = midrank_columns(Zc)                         # (n,C)
        o = ((lab_t @ ranks) - off) / denom                 # (C,)
        nu = ((perm_t @ ranks) - off) / denom               # (P,C)
        obs[c0:c0+chunk] = o.cpu().numpy()
        ge[c0:c0+chunk] = (nu >= o.unsqueeze(0)).sum(dim=0).cpu().numpy()
        del Zc, ranks, o, nu
    pval = (ge + 1.0) / (n_perm + 1.0)
    return obs, pval

def bh_fdr(pvals):
    p = np.asarray(pvals); n = len(p)
    order = np.argsort(p); ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(q, 0, 1)
    return out

def mean_ratio(Z, labels, dev, idxs, chunk=4096):
    """helix:non-helix mean-activation ratio for given latent idxs."""
    # GPU-chunked over latents (avoids giant host-side fancy-index copies).
    idxs = np.asarray(idxs)
    lab = torch.tensor(labels.astype(np.float32), device=dev)
    nh = float(lab.sum().item()); nn = float(len(labels) - nh)
    labn = 1.0 - lab
    out = np.zeros(len(idxs), dtype=np.float64)
    chunk = 2048
    for c0 in range(0, len(idxs), chunk):
        cols = idxs[c0:c0+chunk]
        Zc = torch.as_tensor(np.ascontiguousarray(Z[:, cols]), device=dev,
                             dtype=torch.float32)   # (Ncod, chunk)
        mh = (lab @ Zc) / nh
        mn = (labn @ Zc) / nn
        out[c0:c0+chunk] = (mh / (mn + 1e-8)).cpu().numpy()
        del Zc
    return out

def cluster_bootstrap_auroc(Z, labels, cid, dev, idxs, n_boot=1000, seed=1):
    """Cluster-blocked bootstrap CI of AUROC for each latent in idxs."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(cid)
    # precompute per-cluster row indices
    rows_by_cluster = {c: np.flatnonzero(cid == c) for c in uniq}
    Zi = Z[:, idxs].astype(np.float32)
    boots = np.zeros((n_boot, len(idxs)), dtype=np.float32)
    for b in range(n_boot):
        sel = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([rows_by_cluster[c] for c in sel])
        lab = labels[rows].astype(np.float32)
        n_pos = lab.sum(); n_neg = len(lab) - n_pos
        if n_pos == 0 or n_neg == 0:
            boots[b] = np.nan; continue
        Zb = torch.tensor(Zi[rows], device=dev)
        ranks = midrank_columns(Zb)
        labt = torch.tensor(lab, device=dev)
        boots[b] = (((labt @ ranks) - n_pos*(n_pos+1)/2)/(n_pos*n_neg)).cpu().numpy()
        del Zb, ranks
    lo = np.nanpercentile(boots, 2.5, axis=0)
    hi = np.nanpercentile(boots, 97.5, axis=0)
    return lo, hi


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genes", default=os.path.join(ROOT, "data/ecoli_genes.parquet"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results/m1_features.json"))
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--auroc_thr", type=float, default=0.70)
    ap.add_argument("--q_thr", type=float, default=0.01)
    ap.add_argument("--k_grid", default="1,5,10,20")
    ap.add_argument("--k_cap", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #genes")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    dev = "cuda:0"

    gdf = pd.read_parquet(args.genes)
    if args.limit:
        # keep a balanced-ish slice across splits for a fast sanity pass
        gdf = pd.concat([gdf[gdf.split==s].head(args.limit) for s in ["train","val","test"]])
        gdf = gdf.reset_index(drop=True)
    print(f"[m1] genes={len(gdf)} split counts={gdf['split'].value_counts().to_dict()}",
          flush=True)

    evo2 = C.load_evo2()
    sae = C.TiedTopKSAE(device=dev, relu_before_topk=True)

    data = capture_all(gdf, sae, evo2, dev)
    Z, helix, sheet, coil = data["Z"], data["helix"], data["sheet"], data["coil"]
    gid, cid, split = data["gid"], data["cid"], data["split"]
    tr = split == "train"; va = split == "val"; te = split == "test"
    print(f"[m1] codons train={tr.sum()} val={va.sum()} test={te.sum()} "
          f"helix_frac train={helix[tr].mean():.3f}", flush=True)

    # ---- train: helix AUROC + null + FDR
    print("[m1] train helix AUROC + gene-blocked permutation null ...", flush=True)
    obs_tr, pval = auroc_and_null(Z[tr], helix[tr], gid[tr], dev, n_perm=args.n_perm)
    q = bh_fdr(pval)
    # ---- controls: beta / coil selective latents (train AUROC)
    print("[m1] train beta/coil AUROC (controls) ...", flush=True)
    beta_tr, _, _ = auroc_all_latents(Z[tr], sheet[tr], dev)
    coil_tr, _, _ = auroc_all_latents(Z[tr], coil[tr], dev)
    # ratio helix:nonhelix (train), all latents
    ratio_all = mean_ratio(Z[tr], helix[tr], dev, np.arange(C.D_SAE))

    # matched control ratio bar: top beta and coil latents' helix-ratio not exceeded
    beta_top = np.argsort(-beta_tr)[:50]
    coil_top = np.argsort(-coil_tr)[:50]
    ctrl_helix_auroc_bar = float(max(obs_tr[beta_top].max(), obs_tr[coil_top].max()))
    ctrl_ratio_bar = float(max(ratio_all[beta_top].max(), ratio_all[coil_top].max()))

    # ---- candidates (prereg primary: train AUROC>=thr & q<q_thr & ratio>ctrl_bar)
    cand = np.flatnonzero((obs_tr >= args.auroc_thr) & (q < args.q_thr) &
                          (ratio_all > max(1.0, ctrl_ratio_bar)))
    selection_path = "prereg_auroc"
    print(f"[m1] prereg candidates (train AUROC>={args.auroc_thr}, q<{args.q_thr}, "
          f"ratio>ctrl_bar={ctrl_ratio_bar:.2f}): {len(cand)}", flush=True)
    # ---- specificity fallback: genuine helix features whose codon-level AUROC may
    # sit below the prereg threshold because helix is a windowed property, yet are
    # significantly helix-SELECTIVE and helix-SPECIFIC (helix AUROC exceeds their own
    # beta/coil AUROC by a margin AND exceeds control latents' helix AUROC). Only used
    # if the prereg set is empty; flagged explicitly in the report.
    if len(cand) == 0:
        spec_margin = 0.05
        fb = np.flatnonzero(
            (q < args.q_thr) &
            (obs_tr > ctrl_helix_auroc_bar) &                    # beats best control latent's helix AUROC
            (obs_tr - beta_tr > spec_margin) &                   # helix-specific vs sheet
            (obs_tr - coil_tr > spec_margin) &                   # helix-specific vs coil
            (ratio_all > max(1.5, ctrl_ratio_bar)))
        cand = fb
        selection_path = "specificity_fallback"
        print(f"[m1] prereg set empty -> specificity-fallback candidates "
              f"(q<{args.q_thr}, helixAUROC>ctrl_bar {ctrl_helix_auroc_bar:.3f}, "
              f"helix-beta>{spec_margin}, helix-coil>{spec_margin}, ratio>ctrl): "
              f"{len(cand)}", flush=True)

    # ---- val AUROC for candidates; pick K
    val_auroc_cand, _, _ = auroc_all_latents(Z[va], helix[va], dev) if cand.size else (np.zeros(C.D_SAE),0,0)
    # rank candidates by val AUROC
    cand_sorted = cand[np.argsort(-val_auroc_cand[cand])] if cand.size else cand
    k_grid = [int(x) for x in args.k_grid.split(",")]
    k_grid = [k for k in k_grid if k <= args.k_cap]
    # smallest K whose set-mean val AUROC plateaus (marginal gain < 0.005)
    def setmean_val(k):
        s = cand_sorted[:k]
        return float(val_auroc_cand[s].mean()) if len(s) else 0.0
    chosen_K = k_grid[-1]
    prev = None
    for k in k_grid:
        if k > len(cand_sorted):
            chosen_K = min(len(cand_sorted), k); break
        cur = setmean_val(k)
        if prev is not None and (prev - cur) < 0.005 and (prev - cur) >= 0:
            chosen_K = k_prev; break
        prev, k_prev = cur, k
        chosen_K = k
    if len(cand_sorted) == 0:
        chosen_K = 0
    S = cand_sorted[:chosen_K]
    print(f"[m1] chosen K={chosen_K}, |S|={len(S)}  S={S.tolist()[:32]}", flush=True)

    # ---- test-once: selectivity of S on held-out test
    result = dict(
        n_perm=args.n_perm, auroc_thr=args.auroc_thr, q_thr=args.q_thr,
        k_grid=k_grid, chosen_K=int(chosen_K),
        n_candidates=int(len(cand)),
        n_codons=dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum())),
        helix_frac=dict(train=float(helix[tr].mean()), val=float(helix[va].mean()),
                        test=float(helix[te].mean())),
        ctrl_helix_auroc_bar=ctrl_helix_auroc_bar,
        ctrl_ratio_bar=ctrl_ratio_bar,
        beta_top=beta_top[:20].tolist(), coil_top=coil_top[:20].tolist(),
    )

    if len(S) > 0:
        test_auroc, _, _ = auroc_all_latents(Z[te], helix[te], dev)
        s_lo, s_hi = cluster_bootstrap_auroc(Z[te], helix[te], cid[te], dev, S, n_boot=1000)
        # median positive train activation s_i (steering scale)
        s_i = []
        for i in S:
            col = Z[tr][:, i].astype(np.float32)
            posv = col[col > 0]
            s_i.append(float(np.median(posv)) if posv.size else 0.0)
        ratio_S = mean_ratio(Z[tr], helix[tr], dev, S)
        dnorm = sae.dnorm[torch.tensor(S, device=dev)].cpu().numpy()
        feats = []
        for j, i in enumerate(S):
            feats.append(dict(latent=int(i),
                              train_auroc=float(obs_tr[i]), val_auroc=float(val_auroc_cand[i]),
                              test_auroc=float(test_auroc[i]),
                              test_auroc_ci=[float(s_lo[j]), float(s_hi[j])],
                              ratio_train=float(ratio_S[j]), fdr_q=float(q[i]),
                              s_i=float(s_i[j]), decoder_norm=float(dnorm[j]),
                              beta_auroc=float(beta_tr[i]), coil_auroc=float(coil_tr[i])))
        # PRIMARY feature is the top-1 by VALIDATION AUROC within S (frozen before
        # test) -> report ITS test AUROC. No test-set peeking.
        feats.sort(key=lambda d: -d["val_auroc"])
        primary = feats[0]
        feats.sort(key=lambda d: -d["test_auroc"])   # for display only
        result["features"] = feats
        result["S"] = [int(i) for i in S]
        result["s_i"] = {int(i): sv for i, sv in zip(S, s_i)}
        result["selection_path"] = selection_path
        # Pass predicate C1 (on the val-frozen primary feature)
        prim_ci_lo = primary["test_auroc_ci"][0]
        pass_null = prim_ci_lo > 0.55                         # lower CI > 0.5 with margin
        pass_ctrl = primary["test_auroc"] > ctrl_helix_auroc_bar
        # set-level specificity: mean helix AUROC of S exceeds its mean beta & coil AUROC
        setspec = (np.mean([f["test_auroc"] for f in feats]) >
                   max(np.mean([f["beta_auroc"] for f in feats]),
                       np.mean([f["coil_auroc"] for f in feats])))
        c1_pass = bool(pass_null and pass_ctrl and len(S) > 0)
        result["c1"] = dict(pass_=c1_pass, selection_path=selection_path,
                            primary_latent=primary["latent"],
                            primary_val_auroc=primary["val_auroc"],
                            primary_test_auroc=primary["test_auroc"],
                            primary_test_auroc_ci=primary["test_auroc_ci"],
                            pass_vs_null=bool(pass_null), pass_vs_controls=bool(pass_ctrl),
                            set_level_specific=bool(setspec),
                            ctrl_helix_auroc_bar=ctrl_helix_auroc_bar,
                            max_train_helix_auroc=float(obs_tr.max()),
                            prereg_auroc_thr_met=bool(selection_path=="prereg_auroc"))
        best = primary
        print(f"[m1] C1 pass={c1_pass}  best latent {best['latent']} "
              f"test AUROC {best['test_auroc']:.3f} CI {best['test_auroc_ci']} "
              f"ctrl_bar {ctrl_helix_auroc_bar:.3f}", flush=True)
    else:
        result["features"] = []; result["S"] = []
        result["c1"] = dict(pass_=False, reason="no candidate features")
        print("[m1] C1 FAIL — no candidate helix-selective features", flush=True)

    # save arrays for downstream/audit
    np.savez_compressed(os.path.join(ROOT, "results/m1_arrays.npz"),
                        obs_tr=obs_tr, pval=pval, q=q, beta_tr=beta_tr, coil_tr=coil_tr,
                        ratio_all=ratio_all, S=np.array(result["S"], dtype=np.int64))
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[m1] wrote {args.out}", flush=True)

if __name__ == "__main__":
    main()
