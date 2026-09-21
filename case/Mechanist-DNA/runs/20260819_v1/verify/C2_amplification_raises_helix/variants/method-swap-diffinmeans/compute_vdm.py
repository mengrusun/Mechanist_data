"""VERIFY C2 — METHOD swap: build the DIFF-IN-MEANS steering direction.

Instead of the SAE decoder direction v_S = Sum_{i in S} s_i * dhat_i, extract the
steering direction from a simple mass-mean (diff-in-means) estimator on the layer-26
residual: v_dm = mean(residual over HELIX codons) - mean(residual over NON-HELIX codons),
estimated on TRAIN-split genes only (no test leakage). Norm-matched to ||v_S||.

Same hook site (blocks.26), same additive residual intervention downstream — only the
direction-EXTRACTION method changes. Saves vdm.npz (v_dm, v_S, cosine, norms).
"""
import os, sys, json, time, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../../.."))
sys.path.insert(0, os.path.join(ROOT, "code"))
import common as C


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_genes", type=int, default=600, help="train genes for the mean estimate")
    ap.add_argument("--features", default=os.path.join(ROOT, "results/m1_features.json"))
    args = ap.parse_args()
    dev = "cuda:0"
    import pandas as pd
    gdf = pd.read_parquet(os.path.join(ROOT, "data/ecoli_genes.parquet"))
    tr = gdf[gdf.split == "train"].reset_index(drop=True)
    if args.n_genes and len(tr) > args.n_genes:
        tr = tr.iloc[:args.n_genes].reset_index(drop=True)
    print(f"[vdm] estimating diff-in-means on {len(tr)} train genes", flush=True)

    t0 = time.time()
    evo2 = C.load_evo2()
    tok = evo2.tokenizer

    sum_h = torch.zeros(C.D_MODEL, dtype=torch.float64, device=dev)
    sum_n = torch.zeros(C.D_MODEL, dtype=torch.float64, device=dev)
    cnt_h = 0; cnt_n = 0
    for gi, row in enumerate(tr.itertuples(index=False)):
        L = len(row.aa)
        ss = row.ss3[:L]
        ids = torch.tensor(tok.tokenize(row.dna[:3 * L]), dtype=torch.long,
                           device=dev).unsqueeze(0)
        act = C.capture_layer26(evo2, ids)[0].to(dev, torch.float32)   # (3L, 4096)
        if act.shape[0] < 3 * L:      # fail closed on frame/token misalignment
            print(f"[vdm] skip gene {gi}: act len {act.shape[0]} < 3L={3*L}", flush=True)
            continue
        # per-codon residual = mean over the codon's 3 nt (same convention as m1_capture)
        cod = act[:3 * L].reshape(L, 3, C.D_MODEL).mean(dim=1)          # (L, 4096)
        h = torch.tensor([1.0 if (c < len(ss) and ss[c] == "H") else 0.0 for c in range(L)],
                         device=dev)
        mh = h.sum().item()
        sum_h += (cod * h.unsqueeze(1)).sum(dim=0).double()
        sum_n += (cod * (1.0 - h).unsqueeze(1)).sum(dim=0).double()
        cnt_h += int(mh); cnt_n += int(L - mh)
        if (gi + 1) % 100 == 0:
            print(f"[vdm]  {gi+1}/{len(tr)} genes, helix_codons={cnt_h} nonhelix={cnt_n} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    mean_h = (sum_h / max(1, cnt_h)).float()
    mean_n = (sum_n / max(1, cnt_n)).float()
    v_dm = (mean_h - mean_n)                     # (4096,)
    raw_norm = float(v_dm.norm().item())

    # reference SAE steer vector v_S, and its norm, for norm-matching + cosine
    sae = C.TiedTopKSAE(device=dev, relu_before_topk=True)
    feat = json.load(open(args.features))
    S = np.array(feat["S"], dtype=np.int64)
    s_i = np.array([feat["s_i"][str(i)] for i in S], dtype=np.float32)
    v_S = sae.steer_vector(torch.tensor(S, device=dev), torch.tensor(s_i, device=dev)).float()
    normS = float(v_S.norm().item())

    v_dm_matched = v_dm * (normS / (raw_norm + 1e-8))     # norm-matched to ||v_S||
    cos = float(torch.nn.functional.cosine_similarity(v_dm, v_S, dim=0).item())

    outp = os.path.join(HERE, "vdm.npz")
    np.savez(outp,
             v_dm=v_dm_matched.cpu().numpy(), v_dm_raw=v_dm.cpu().numpy(),
             v_S=v_S.cpu().numpy(), norm_S=normS, raw_norm_vdm=raw_norm,
             cosine_vdm_vS=cos, n_helix_codons=cnt_h, n_nonhelix_codons=cnt_n,
             n_genes=len(tr))
    print(f"[vdm] ||v_dm_raw||={raw_norm:.3f} matched to ||v_S||={normS:.3f}; "
          f"cosine(v_dm, v_S)={cos:.3f}; helix_codons={cnt_h} nonhelix={cnt_n}", flush=True)
    print(f"[vdm] saved {outp} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
