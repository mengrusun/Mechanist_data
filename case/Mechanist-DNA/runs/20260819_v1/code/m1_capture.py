"""M1 capture (one GPU shard): forward the gene shard through Evo2-7B, hook
blocks.26, SAE-encode, store per-codon latents (mean over 3 nt) + SS labels.
Saves results/m1_shard{k}.npz. Vectorized labels; float32 latents.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
import torch
import common as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshard", type=int, default=4)
    ap.add_argument("--genes", default=os.path.join(ROOT, "data/ecoli_genes.parquet"))
    args = ap.parse_args()
    dev = "cuda:0"
    gdf = pd.read_parquet(args.genes).reset_index(drop=True)
    sub = gdf.iloc[args.shard::args.nshard].reset_index(drop=True)   # strided shard
    n_codons = int(sub["aa"].str.len().sum())
    print(f"[cap{args.shard}] {len(sub)} genes, {n_codons} codons "
          f"(~{n_codons*C.D_SAE*4/1e9:.1f}GB)", flush=True)

    evo2 = C.load_evo2()
    sae = C.TiedTopKSAE(device=dev, relu_before_topk=True)

    Z = np.zeros((n_codons, C.D_SAE), dtype=np.float32)
    helix = np.zeros(n_codons, dtype=np.int8)
    sheet = np.zeros(n_codons, dtype=np.int8)
    coil  = np.zeros(n_codons, dtype=np.int8)
    gene_id = np.zeros(n_codons, dtype=np.int32)
    cluster = np.empty(n_codons, dtype=object)
    split = np.empty(n_codons, dtype=object)
    tok = evo2.tokenizer
    pos = 0; t0 = time.time()
    for gi, row in enumerate(sub.itertuples(index=False)):
        L = len(row.aa)
        ss = np.frombuffer(row.ss3[:L].encode("ascii"), dtype="S1")
        if len(ss) < L:
            ss = np.concatenate([ss, np.array([b"C"]*(L-len(ss)), dtype="S1")])
        ids = torch.tensor(tok.tokenize(row.dna[:3*L]), dtype=torch.long,
                           device=dev).unsqueeze(0)
        act = C.capture_layer26(evo2, ids)[0].to(dev, torch.float32)
        z = sae.encode(act)[:3*L].reshape(L, 3, C.D_SAE).mean(dim=1)
        Z[pos:pos+L] = z.cpu().numpy()
        helix[pos:pos+L] = (ss == b"H")
        sheet[pos:pos+L] = (ss == b"E")
        coil[pos:pos+L]  = (ss == b"C")
        # globally-unique gene id = original stride index
        gene_id[pos:pos+L] = args.shard + gi*args.nshard
        cluster[pos:pos+L] = row.cluster
        split[pos:pos+L] = row.split
        pos += L
        if (gi+1) % 250 == 0:
            print(f"[cap{args.shard}] {gi+1}/{len(sub)} genes {pos} codons "
                  f"{time.time()-t0:.0f}s", flush=True)
    assert pos == n_codons
    outp = os.path.join(ROOT, f"results/m1_shard{args.shard}.npz")
    np.savez(outp, Z=Z, helix=helix, sheet=sheet, coil=coil,
             gene_id=gene_id, cluster=cluster.astype(str), split=split.astype(str))
    print(f"[cap{args.shard}] saved {outp} in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
