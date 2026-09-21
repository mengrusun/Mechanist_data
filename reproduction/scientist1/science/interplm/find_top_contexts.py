"""For each of a chosen set of SAE features, gather the top-K protein contexts
where the feature activates most strongly. A "context" is a (protein_id, residue,
window_seq) triple, with a small local window around the peak residue.

Streams over the corpus; keeps a heap per feature.
"""
from __future__ import annotations
import argparse
import heapq
import pickle
import os
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from sae_module import load_sae


@torch.no_grad()
def run(args):
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

    feature_ids = args.features
    K = args.topk
    win = args.window

    # heap of (act, prot_idx, residue, window_seq) per feature
    heaps: dict[int, list] = {fid: [] for fid in feature_ids}

    for i in tqdm(range(len(seqs))):
        s = seqs[i]
        enc = tok(s, return_tensors='pt', truncation=True, max_length=1024).to(device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[args.layer][0, 1:-1]  # (L, D)
        z = sae.encode(h)                            # (L, D_sae)
        for fid in feature_ids:
            z_f = z[:, fid]
            top_vals, top_pos = torch.topk(z_f, k=min(3, z_f.shape[0]))
            for val, pos in zip(top_vals.tolist(), top_pos.tolist()):
                if val <= 0:
                    break
                a = max(0, pos - win)
                b = min(len(s), pos + win + 1)
                window_seq = s[a:b]
                mark_seq = s[a:pos] + "[" + s[pos] + "]" + s[pos+1:b]
                item = (val, i, pos, mark_seq, entries[i])
                h_ = heaps[fid]
                if len(h_) < K:
                    heapq.heappush(h_, item)
                elif val > h_[0][0]:
                    heapq.heapreplace(h_, item)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    result = {fid: sorted(heaps[fid], reverse=True) for fid in feature_ids}
    with open(args.out, "wb") as f:
        pickle.dump(result, f)
    for fid in feature_ids[:5]:
        print(f"\n== feature {fid} ==")
        for val, pi, pos, mark_seq, ac in sorted(heaps[fid], reverse=True)[:5]:
            print(f"  act={val:.3f}  {ac}@{pos}  {mark_seq}")
    print(f"wrote {args.out}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm_dir", default="/data/zhenqian/models/esm2_t33_650M_UR50D")
    ap.add_argument("--sae_dir", required=True)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--features", type=int, nargs="+", required=True)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--window", type=int, default=12)
    ap.add_argument("--max_proteins", type=int, default=3000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
