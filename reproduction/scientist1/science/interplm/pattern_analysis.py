"""For a set of SAE features, gather top-activating protein contexts and
report AA composition / consensus motif around the peak residue.

Two use cases:
  1. Well-aligned features (high Swiss-Prot F1) → sanity check that the
     motif matches the concept name.
  2. Unaligned features (low Swiss-Prot F1) → look for coherent AA patterns
     that suggest an unlabeled biological concept.
"""
from __future__ import annotations
import argparse
import pickle
import os
import numpy as np
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from sae_module import load_sae


AA = "ACDEFGHIKLMNPQRSTVWY"


def entropy(counter, total):
    p = np.array([counter.get(a, 0) / total for a in AA])
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def consensus(positions_seqs):
    """positions_seqs: list of aligned windows (same length, may include '-').
    Returns per-position consensus letter and frequency."""
    if not positions_seqs:
        return ""
    L = len(positions_seqs[0])
    out = []
    for i in range(L):
        col = [s[i] for s in positions_seqs if i < len(s) and s[i] != "-"]
        if not col:
            out.append("_"); continue
        c = Counter(col).most_common(1)[0]
        letter, count = c
        freq = count / len(col)
        out.append(letter if freq >= 0.5 else letter.lower() if freq >= 0.3 else "x")
    return "".join(out)


@torch.no_grad()
def gather(args):
    device = args.device
    tok = AutoTokenizer.from_pretrained(args.esm_dir)
    model = AutoModel.from_pretrained(args.esm_dir, torch_dtype=torch.float32).to(device).eval()
    sae = load_sae(args.sae_dir, normalized=True, device=device)

    with open(args.corpus, "rb") as f:
        corpus = pickle.load(f)
    entries, seqs = corpus["entries"], corpus["seqs"]
    if args.max_proteins and args.max_proteins < len(seqs):
        seqs = seqs[:args.max_proteins]
        entries = entries[:args.max_proteins]

    feat_ids = args.features
    K = args.topk
    win = args.window
    # keep top-K (val, prot_idx, pos, seq_window) per feature
    top = {fid: [] for fid in feat_ids}

    for i in tqdm(range(len(seqs))):
        s = seqs[i]
        enc = tok(s, return_tensors='pt', truncation=True, max_length=1024).to(device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[args.layer][0, 1:-1]
        z = sae.encode(h)                             # (L, D_sae)
        for fid in feat_ids:
            zf = z[:, fid]
            top_vals, top_pos = torch.topk(zf, k=min(2, zf.shape[0]))
            for v, p in zip(top_vals.tolist(), top_pos.tolist()):
                if v <= 0: break
                a = max(0, p - win); b = min(len(s), p + win + 1)
                pad_l = win - (p - a)
                pad_r = win - (b - p - 1)
                window = "-" * pad_l + s[a:b] + "-" * pad_r  # fixed length 2*win+1
                item = (v, i, p, window, entries[i])
                arr = top[fid]
                if len(arr) < K:
                    arr.append(item)
                    arr.sort(reverse=True)
                elif v > arr[-1][0]:
                    arr[-1] = item
                    arr.sort(reverse=True)

    results = []
    for fid in feat_ids:
        arr = top[fid]
        wins = [item[3] for item in arr]
        pool = Counter("".join(w for w in wins))
        cons = consensus(wins)
        center = Counter(w[win] if len(w) > win else "-" for w in wins)
        rec = {
            "feature": fid,
            "top_activation": arr[0][0] if arr else None,
            "n_contexts": len(arr),
            "consensus": cons,
            "center_aa": dict(center.most_common(5)),
            "examples": [(round(v, 3), ac, p, w) for v, i, p, w, ac in arr[:8]],
        }
        results.append(rec)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(results, f)
    for r in results[:10]:
        print(f"\nfeat {r['feature']:>5d}  top_act={r['top_activation']}")
        print(f"  consensus:  {r['consensus']}")
        print(f"  center aa:  {r['center_aa']}")
        for v, ac, p, w in r["examples"]:
            print(f"    {v:.2f}  {ac}@{p}  {w}")
    print(f"\nwrote {args.out}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm_dir", default="/data/zhenqian/models/esm2_t33_650M_UR50D")
    ap.add_argument("--sae_dir", required=True)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--features", type=int, nargs="+", required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--max_proteins", type=int, default=3000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


if __name__ == "__main__":
    gather(parse_args())
