"""M1 analyze: merge capture shards, compute per-latent helix AUROC + gene-blocked
permutation null (FDR), beta/coil controls, select feature set S, test-once with
cluster-blocked bootstrap CI, freeze S + s_i. Writes results/m1_features.json.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import common as C
from m1_feature_select import (midrank_columns, auroc_from_ranks, auroc_all_latents,
                               auroc_and_null, bh_fdr, mean_ratio, cluster_bootstrap_auroc)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nshard", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(ROOT, "results/m1_features.json"))
    ap.add_argument("--n_perm", type=int, default=200)
    ap.add_argument("--auroc_thr", type=float, default=0.70)
    ap.add_argument("--q_thr", type=float, default=0.01)
    ap.add_argument("--k_grid", default="1,5,10,20")
    ap.add_argument("--k_cap", type=int, default=32)
    args = ap.parse_args()
    dev = "cuda:0"
    t0 = time.time()

    # two-pass: get per-shard sizes, preallocate, fill slices (avoids 154GB concat copy)
    sizes = []
    for k in range(args.nshard):
        with np.load(os.path.join(ROOT, f"results/m1_shard{k}.npz"), allow_pickle=True) as d:
            sizes.append(d["helix"].shape[0])
    N = int(sum(sizes)); off = np.cumsum([0]+sizes)
    Z = np.empty((N, C.D_SAE), dtype=np.float32)
    helix = np.empty(N, np.int8); sheet = np.empty(N, np.int8); coil = np.empty(N, np.int8)
    gid = np.empty(N, np.int32); cluster = np.empty(N, object); split = np.empty(N, object)
    for k in range(args.nshard):
        a, b = off[k], off[k+1]
        with np.load(os.path.join(ROOT, f"results/m1_shard{k}.npz"), allow_pickle=True) as d:
            Z[a:b] = d["Z"]; helix[a:b] = d["helix"]; sheet[a:b] = d["sheet"]
            coil[a:b] = d["coil"]; gid[a:b] = d["gene_id"]
            cluster[a:b] = d["cluster"]; split[a:b] = d["split"]
        print(f"[m1a] loaded shard {k}: {b-a} codons ({time.time()-t0:.0f}s)", flush=True)
    clu_map = {c: i for i, c in enumerate(sorted(set(cluster.tolist())))}
    cid = np.array([clu_map[c] for c in cluster], dtype=np.int32)
    print(f"[m1a] merged Z {Z.shape} in {time.time()-t0:.0f}s", flush=True)

    tr = split == "train"; va = split == "val"; te = split == "test"
    print(f"[m1a] codons train={tr.sum()} val={va.sum()} test={te.sum()} "
          f"helix_frac train={helix[tr].mean():.3f}", flush=True)

    CH = 384   # latent chunk: keeps midrank argsort tensors small on 80GB GPU
    Ztr = np.ascontiguousarray(Z[tr]); htr = helix[tr]; gidtr = gid[tr]
    print("[m1a] train helix AUROC + gene-blocked permutation null ...", flush=True)
    obs_tr, pval = auroc_and_null(Ztr, htr, gidtr, dev, n_perm=args.n_perm, chunk=CH)
    q = bh_fdr(pval)
    print(f"[m1a] max train helix AUROC={obs_tr.max():.3f}  #>=0.70={int((obs_tr>=0.70).sum())} "
          f"#q<0.01={int((q<0.01).sum())} ({time.time()-t0:.0f}s)", flush=True)
    beta_tr, _, _ = auroc_all_latents(Ztr, sheet[tr], dev, chunk=CH)
    coil_tr, _, _ = auroc_all_latents(Ztr, coil[tr], dev, chunk=CH)
    ratio_all = mean_ratio(Ztr, htr, dev, np.arange(C.D_SAE))
    beta_top = np.argsort(-beta_tr)[:50]; coil_top = np.argsort(-coil_tr)[:50]
    ctrl_helix_auroc_bar = float(max(obs_tr[beta_top].max(), obs_tr[coil_top].max()))
    ctrl_ratio_bar = float(max(ratio_all[beta_top].max(), ratio_all[coil_top].max()))

    cand = np.flatnonzero((obs_tr >= args.auroc_thr) & (q < args.q_thr) &
                          (ratio_all > max(1.0, ctrl_ratio_bar)))
    selection_path = "prereg_auroc"
    print(f"[m1a] prereg candidates: {len(cand)} (ctrl_ratio_bar={ctrl_ratio_bar:.2f})", flush=True)
    if len(cand) == 0:
        spec_margin = 0.05
        cand = np.flatnonzero((q < args.q_thr) & (obs_tr > ctrl_helix_auroc_bar) &
                              (obs_tr - beta_tr > spec_margin) & (obs_tr - coil_tr > spec_margin) &
                              (ratio_all > max(1.5, ctrl_ratio_bar)))
        selection_path = "specificity_fallback"
        print(f"[m1a] specificity-fallback candidates: {len(cand)} "
              f"(ctrl_helix_bar={ctrl_helix_auroc_bar:.3f})", flush=True)

    val_auroc, _, _ = auroc_all_latents(np.ascontiguousarray(Z[va]), helix[va], dev, chunk=CH)
    cand_sorted = cand[np.argsort(-val_auroc[cand])] if cand.size else cand
    k_grid = [int(x) for x in args.k_grid.split(",") if int(x) <= args.k_cap]
    def setmean_val(k):
        s = cand_sorted[:k]; return float(val_auroc[s].mean()) if len(s) else 0.0
    chosen_K = 0; prev = None; k_prev = 0
    for k in k_grid:
        if k > len(cand_sorted):
            chosen_K = min(len(cand_sorted), k); break
        cur = setmean_val(k)
        if prev is not None and 0 <= (prev - cur) < 0.005:
            chosen_K = k_prev; break
        prev, k_prev = cur, k; chosen_K = k
    if len(cand_sorted) == 0:
        chosen_K = 0
    S = cand_sorted[:chosen_K]
    print(f"[m1a] selection_path={selection_path} chosen_K={chosen_K} |S|={len(S)} "
          f"S={S.tolist()[:32]}", flush=True)

    result = dict(n_perm=args.n_perm, auroc_thr=args.auroc_thr, q_thr=args.q_thr,
                  k_grid=k_grid, chosen_K=int(chosen_K), n_candidates=int(len(cand)),
                  selection_path=selection_path,
                  n_codons=dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum())),
                  helix_frac=dict(train=float(helix[tr].mean()), val=float(helix[va].mean()),
                                  test=float(helix[te].mean())),
                  ctrl_helix_auroc_bar=ctrl_helix_auroc_bar, ctrl_ratio_bar=ctrl_ratio_bar,
                  beta_top=beta_top[:20].tolist(), coil_top=coil_top[:20].tolist(),
                  max_train_helix_auroc=float(obs_tr.max()))

    if len(S) > 0:
        Zte = np.ascontiguousarray(Z[te])
        test_auroc, _, _ = auroc_all_latents(Zte, helix[te], dev, chunk=CH)
        s_lo, s_hi = cluster_bootstrap_auroc(Zte, helix[te], cid[te], dev, S, n_boot=1000)
        s_i = []
        for i in S:
            col = Ztr[:, i]; posv = col[col > 0]
            s_i.append(float(np.median(posv)) if posv.size else 0.0)
        ratio_S = mean_ratio(Ztr, htr, dev, S)
        dnorm = C.TiedTopKSAE(device=dev).dnorm[torch.tensor(S, device=dev)].cpu().numpy()
        feats = []
        for j, i in enumerate(S):
            feats.append(dict(latent=int(i), train_auroc=float(obs_tr[i]),
                              val_auroc=float(val_auroc[i]), test_auroc=float(test_auroc[i]),
                              test_auroc_ci=[float(s_lo[j]), float(s_hi[j])],
                              ratio_train=float(ratio_S[j]), fdr_q=float(q[i]),
                              s_i=float(s_i[j]), decoder_norm=float(dnorm[j]),
                              beta_auroc=float(beta_tr[i]), coil_auroc=float(coil_tr[i])))
        prim = sorted(feats, key=lambda d: -d["val_auroc"])[0]
        feats.sort(key=lambda d: -d["test_auroc"])
        result["features"] = feats
        result["S"] = [int(i) for i in S]
        result["s_i"] = {int(i): sv for i, sv in zip(S, s_i)}
        pass_null = prim["test_auroc_ci"][0] > 0.55
        pass_ctrl = prim["test_auroc"] > ctrl_helix_auroc_bar
        setspec = (np.mean([f["test_auroc"] for f in feats]) >
                   max(np.mean([f["beta_auroc"] for f in feats]),
                       np.mean([f["coil_auroc"] for f in feats])))
        result["c1"] = dict(pass_=bool(pass_null and pass_ctrl and len(S) > 0),
                            selection_path=selection_path, primary_latent=prim["latent"],
                            primary_val_auroc=prim["val_auroc"], primary_test_auroc=prim["test_auroc"],
                            primary_test_auroc_ci=prim["test_auroc_ci"],
                            pass_vs_null=bool(pass_null), pass_vs_controls=bool(pass_ctrl),
                            set_level_specific=bool(setspec),
                            ctrl_helix_auroc_bar=ctrl_helix_auroc_bar,
                            max_train_helix_auroc=float(obs_tr.max()),
                            prereg_auroc_thr_met=bool(selection_path == "prereg_auroc"))
        print(f"[m1a] C1 pass={result['c1']['pass_']} primary latent {prim['latent']} "
              f"test AUROC {prim['test_auroc']:.3f} CI {prim['test_auroc_ci']} "
              f"ctrl_bar {ctrl_helix_auroc_bar:.3f}", flush=True)
    else:
        result["features"] = []; result["S"] = []
        result["c1"] = dict(pass_=False, reason="no candidate helix-selective features",
                            max_train_helix_auroc=float(obs_tr.max()),
                            ctrl_helix_auroc_bar=ctrl_helix_auroc_bar)
        print("[m1a] C1 FAIL — no candidates", flush=True)

    np.savez_compressed(os.path.join(ROOT, "results/m1_arrays.npz"),
                        obs_tr=obs_tr, pval=pval, q=q, beta_tr=beta_tr, coil_tr=coil_tr,
                        ratio_all=ratio_all, S=np.array(result["S"], dtype=np.int64))
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[m1a] wrote {args.out} in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
