"""
Cache Layer-26 SAE per-codon feature activations for one organism's M0 dataset (GPU).

For each natural CDS: Evo2-7B forward -> extract chosen Layer-26 residual site per nucleotide
-> SAE-encode (top-k=64) per nucleotide -> mean-pool over each codon's 3 nucleotides
-> per-codon 32768-dim sparse feature vector.

Outputs (data/):
  m0_acts_<org>.npz         scipy CSR (n_codons x 32768) mean-pooled activations
  m0_labels_<org>.npz       arrays: helix_hgi, helix_h, sheet, ss_char(int), prot_idx, cluster,
                            pos_frac (position in CDS), gc3 (codon GC), codon_id (0..63)
  m0_index_<org>.json       prot_idx -> {acc, pdb, chain, n_codons_labeled}
Run: CUDA_VISIBLE_DEVICES=2 python code/m0_cache_acts.py --organism prokaryote
"""
import os, sys, json, argparse, time
import numpy as np
from scipy import sparse
import torch
sys.path.insert(0, os.path.dirname(__file__))
import m0_data as D
from evo2_sae import load_evo2, BatchTopKSAE, tokenize_seq, HIDDEN, SAE_DICT

BASES = "ACGT"
CODON2ID = {a+b+c: i for i, (a, b, c) in enumerate(
    [(x, y, z) for x in BASES for y in BASES for z in BASES])}


SAE_NORM = "none"

def get_site():
    # Validated by known-feature reproduction: blocks.26.post_norm (RMSNorm output feeding the MLP)
    # with raw (none) SAE input normalization makes the Evo2 paper's named Layer-26 features the
    # TOP discriminators of their class: alpha-helix f/28741 = rank-0 helix (AUROC 0.663),
    # beta-sheet f/22326 = rank-0 sheet. This identifies post_norm as the SAE's native training site.
    return "blocks.26.post_norm"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organism", required=True)
    ap.add_argument("--max_nt", type=int, default=8000, help="skip CDS longer than this (context)")
    ap.add_argument("--batch_dtype", default="float16")
    args = ap.parse_args()
    site = get_site()
    print(f"[cache] organism={args.organism} site={site}", flush=True)

    ds_path = os.path.join(D.DATA_DIR, f"m0_dataset_{args.organism}.jsonl")
    records = [json.loads(l) for l in open(ds_path)]
    clu_path = os.path.join(D.DATA_DIR, f"m0_clusters_{args.organism}.json")
    if os.path.exists(clu_path):
        clusters = json.load(open(clu_path))
    else:
        # fallback (sanity / clusters not yet written): each protein its own cluster
        clusters = {r["acc"]: i for i, r in enumerate(records)}
        print("[cache] WARNING: clusters file missing; using per-protein clusters (sanity mode)", flush=True)
    print(f"[cache] {len(records)} proteins", flush=True)

    model = load_evo2("cuda:0")
    sae = BatchTopKSAE(device="cuda:0", dtype=torch.float32)

    rows_idx, cols_idx, vals = [], [], []
    helix_hgi, helix_h, sheet, ss_int = [], [], [], []
    prot_idx, cluster_arr, pos_frac, gc3, codon_id = [], [], [], [], []
    index = {}
    codon_counter = 0
    t0 = time.time()
    SS_ORDER = "HGIEBTS-"  # map ss char to int
    ss_map = {c: i for i, c in enumerate(SS_ORDER)}

    for pi, rec in enumerate(records):
        cds = rec["cds"]; codon_ss = rec["codon_ss"]; n_cod = rec["n_codons"]
        nt = cds[: n_cod * 3]
        if len(nt) > args.max_nt:
            # cache still (truncation would misalign codons) -> skip overly long
            pass
        if len(nt) < 30 or len(nt) > args.max_nt:
            continue
        tok = tokenize_seq(model, nt, "cuda:0")
        with torch.no_grad():
            out = model(tok, return_embeddings=True, layer_names=[site])
        resid = out[1][site].float()[0]  # (L_nt, 4096)
        if resid.shape[0] != len(nt):
            continue
        z = sae.encode(resid)  # (L_nt, 32768) sparse-dense
        # mean-pool per codon over 3 nt
        Lc = resid.shape[0] // 3
        z = z[: Lc * 3].reshape(Lc, 3, SAE_DICT).mean(1)  # (Lc, 32768)
        z = z.cpu().numpy()
        labeled_here = 0
        for c in range(min(Lc, n_cod)):
            ss = codon_ss[c]
            if ss is None:
                continue
            nzc = np.nonzero(z[c])[0]
            if nzc.size == 0:
                continue
            rows_idx.append(np.full(nzc.size, codon_counter, dtype=np.int64))
            cols_idx.append(nzc.astype(np.int64))
            vals.append(z[c, nzc].astype(np.float32))
            helix_hgi.append(1 if ss in D.HELIX_HGI else 0)
            helix_h.append(1 if ss in D.HELIX_H else 0)
            sheet.append(1 if ss in D.SHEET else 0)
            ss_int.append(ss_map.get(ss, 7))
            prot_idx.append(pi)
            cluster_arr.append(clusters.get(rec["acc"], -1))
            pos_frac.append(c / max(n_cod - 1, 1))
            cod = nt[c*3:c*3+3].upper()
            gc3.append((cod.count("G") + cod.count("C")) / 3.0)
            codon_id.append(CODON2ID.get(cod, -1))
            codon_counter += 1
            labeled_here += 1
        index[pi] = {"acc": rec["acc"], "pdb": rec["pdb"], "chain": rec["chain"],
                     "labeled": labeled_here}
        if (pi + 1) % 100 == 0:
            print(f"  {pi+1}/{len(records)} proteins; codons={codon_counter} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    print(f"[cache] total labeled codons: {codon_counter}", flush=True)
    rows_idx = np.concatenate(rows_idx); cols_idx = np.concatenate(cols_idx)
    vals = np.concatenate(vals)
    X = sparse.csr_matrix((vals, (rows_idx, cols_idx)), shape=(codon_counter, SAE_DICT))
    sparse.save_npz(os.path.join(D.DATA_DIR, f"m0_acts_{args.organism}.npz"), X)
    np.savez(os.path.join(D.DATA_DIR, f"m0_labels_{args.organism}.npz"),
             helix_hgi=np.array(helix_hgi, np.int8), helix_h=np.array(helix_h, np.int8),
             sheet=np.array(sheet, np.int8), ss_int=np.array(ss_int, np.int8),
             prot_idx=np.array(prot_idx, np.int32), cluster=np.array(cluster_arr, np.int32),
             pos_frac=np.array(pos_frac, np.float32), gc3=np.array(gc3, np.float32),
             codon_id=np.array(codon_id, np.int16))
    json.dump(index, open(os.path.join(D.DATA_DIR, f"m0_index_{args.organism}.json"), "w"))
    print(f"[cache] saved. X shape={X.shape} nnz={X.nnz} site={site}", flush=True)
    print(f"CACHE_DONE organism={args.organism} codons={codon_counter}", flush=True)


if __name__ == "__main__":
    main()
