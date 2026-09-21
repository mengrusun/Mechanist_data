"""Extract ESM-2-650M residual-stream activations at a target layer for each
Swiss-Prot protein in a parsed corpus, then push them through the SAE.

For each protein we write:
  * per-residue raw activations           (T, D_esm)     - int8 quantized to save disk (optional)
  * per-residue SAE features (post-ReLU)  (T, D_sae)     - float16, sparse-ish

Because storing dense per-residue tensors for 10k proteins is huge, this script
instead consumes the corpus in a streaming fashion and produces the aggregate
per-feature vs per-concept confusion counts we care about, without persisting
the raw tensors.

Confusion accumulator (per layer, one for SAE and one for raw neurons):
  For each candidate feature/neuron f and each concept c we track over M
  thresholds: TP[f,c,t], FP[f,c,t], FN[f,c,t]. Because that is O(F * C * T)
  and F=10240, C~500, T=20, it's ~100M entries per layer - fine as int32.

Positives per concept c across all residues is tracked once (P[c]).
"""
from __future__ import annotations
import argparse
import os
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from sae_module import load_sae


def batched(seqs, batch_size):
    for i in range(0, len(seqs), batch_size):
        yield i, seqs[i:i + batch_size]


@torch.no_grad()
def run(args):
    device = args.device
    tok = AutoTokenizer.from_pretrained(args.esm_dir)
    print("loading ESM-2 ...")
    model = AutoModel.from_pretrained(args.esm_dir, torch_dtype=torch.float32).to(device).eval()
    D_esm = model.config.hidden_size
    print(f"loaded ESM-2, hidden_size={D_esm}, num_layers={model.config.num_hidden_layers}")

    print("loading SAE ...")
    sae = load_sae(args.sae_dir, normalized=True, device=device)
    D_sae = sae.d_sae
    print(f"loaded SAE, d_in={sae.d_in} d_sae={D_sae}")

    with open(args.corpus, "rb") as f:
        corpus = pickle.load(f)
    entries, seqs, anns, concepts = corpus["entries"], corpus["seqs"], corpus["anns"], corpus["concepts"]
    C = len(concepts)
    print(f"corpus: {len(seqs)} proteins, {C} concepts")

    # subsample proteins for speed if requested
    if args.max_proteins and args.max_proteins < len(seqs):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(seqs), size=args.max_proteins, replace=False)
        idx.sort()
        entries = [entries[i] for i in idx]
        seqs = [seqs[i] for i in idx]
        anns = [anns[i] for i in idx]
        print(f"subsampled to {len(seqs)} proteins")

    # thresholds: log-spaced multiples of a global activation scale
    # SAE features are non-negative; neurons can be signed - we take abs
    T = args.n_thresholds
    # per-feature thresholds are chosen as quantiles of nonzero activations;
    # here we use a simple percentile-based scheme after gathering a sample.

    # ---- first pass: gather max activations to fix per-feature thresholds ----
    print("=== pass 1: fitting per-feature thresholds from a sample ===")
    sample_ids = list(range(min(args.sample_size, len(seqs))))
    max_sae = torch.zeros(D_sae, device=device)
    max_neu = torch.zeros(D_esm, device=device)
    n_res = 0
    for i in tqdm(sample_ids):
        s = seqs[i]
        enc = tok(s, return_tensors='pt', truncation=True, max_length=args.max_len + 2).to(device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[args.layer][0, 1:-1]  # drop CLS/EOS => (T, D)
        max_sae = torch.maximum(max_sae, sae.encode(h).max(dim=0).values)
        max_neu = torch.maximum(max_neu, h.abs().max(dim=0).values)
        n_res += h.shape[0]
    print(f"residues sampled: {n_res}")

    # per-feature threshold grid: fractions of max
    frac = torch.linspace(0.05, 0.95, T, device=device)  # (T,)
    thr_sae = max_sae[:, None] * frac[None, :]           # (D_sae, T)
    thr_neu = max_neu[:, None] * frac[None, :]           # (D_esm, T)

    # ---- allocate confusion accumulators ----
    print("=== allocating accumulators ===")
    # For memory: TP has shape (F, C, T). SAE: 10240*500*20*4B ~ 400 MB int32.
    # Neurons: 1280*500*20*4B ~ 50 MB.
    def _alloc(F):
        return {
            "TP": np.zeros((F, C, T), dtype=np.int32),
            "PredPos": np.zeros((F, T), dtype=np.int64),  # sum over residues of indicator
        }
    acc_sae = _alloc(D_sae)
    acc_neu = _alloc(D_esm)
    label_pos = np.zeros(C, dtype=np.int64)  # count of positive-labeled residues per concept
    total_residues = 0

    # ---- second pass: compute per-residue predictions and update accumulators ----
    print("=== pass 2: running over corpus ===")
    for i in tqdm(range(len(seqs))):
        s = seqs[i]
        L = len(s)
        enc = tok(s, return_tensors='pt', truncation=True, max_length=args.max_len + 2).to(device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[args.layer][0, 1:-1]  # (L, D_esm)  (L clipped if truncated)
        L_eff = h.shape[0]

        # build per-residue label matrix Y (L_eff, C) sparse -> dense via segments
        Y = np.zeros((L_eff, C), dtype=bool)
        for (a, b, c) in anns[i]:
            if a >= L_eff:
                continue
            Y[a:min(b + 1, L_eff), c] = True
        label_pos += Y.sum(axis=0)
        total_residues += L_eff

        # SAE features
        z = sae.encode(h)                                # (L, D_sae)
        # predictions at all T thresholds: (L, D_sae, T)
        # avoid materializing full 3D tensor for D_sae; do in feature chunks
        Y_t = torch.from_numpy(Y).to(device)             # (L, C)  bool

        # Chunk over features
        chunk = args.feat_chunk
        for f0 in range(0, D_sae, chunk):
            f1 = min(f0 + chunk, D_sae)
            z_c = z[:, f0:f1]                            # (L, cf)
            t_c = thr_sae[f0:f1]                         # (cf, T)
            # pred: (L, cf, T)
            pred = z_c.unsqueeze(-1) > t_c.unsqueeze(0)  # bool
            # for each threshold: PredPos[f,t] += sum_L pred[L,f,t]
            pp = pred.sum(dim=0)                         # (cf, T)  int64
            acc_sae["PredPos"][f0:f1] += pp.cpu().numpy().astype(np.int64)
            # TP[f,c,t] += sum_L pred[L,f,t] & Y[L,c]
            # reshape: pred (L, cf*T) x Y (L, C) -> (cf*T, C)
            cf, T_ = pred.shape[1], pred.shape[2]
            pred_flat = pred.reshape(L_eff, cf * T_).float()   # (L, cf*T)
            tp = pred_flat.T @ Y_t.float()                     # (cf*T, C)
            tp = tp.reshape(cf, T_, C).permute(0, 2, 1)        # (cf, C, T)
            acc_sae["TP"][f0:f1] += tp.cpu().numpy().astype(np.int32)

        # Raw neurons (absolute value)
        n = h.abs()                                      # (L, D_esm)
        for f0 in range(0, D_esm, chunk):
            f1 = min(f0 + chunk, D_esm)
            n_c = n[:, f0:f1]
            t_c = thr_neu[f0:f1]
            pred = n_c.unsqueeze(-1) > t_c.unsqueeze(0)
            pp = pred.sum(dim=0)
            acc_neu["PredPos"][f0:f1] += pp.cpu().numpy().astype(np.int64)
            cf, T_ = pred.shape[1], pred.shape[2]
            pred_flat = pred.reshape(L_eff, cf * T_).float()
            tp = pred_flat.T @ Y_t.float()
            tp = tp.reshape(cf, T_, C).permute(0, 2, 1)
            acc_neu["TP"][f0:f1] += tp.cpu().numpy().astype(np.int32)

    print(f"total residues processed: {total_residues}")

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out_dir, "confusion.npz"),
        TP_sae=acc_sae["TP"],
        PredPos_sae=acc_sae["PredPos"],
        TP_neu=acc_neu["TP"],
        PredPos_neu=acc_neu["PredPos"],
        label_pos=label_pos,
        total_residues=np.int64(total_residues),
        thr_sae=thr_sae.cpu().numpy(),
        thr_neu=thr_neu.cpu().numpy(),
    )
    with open(os.path.join(args.out_dir, "concepts.pkl"), "wb") as f:
        pickle.dump({"concepts": concepts, "label_pos": label_pos.tolist(),
                     "total_residues": int(total_residues)}, f)
    print(f"wrote confusion tensors to {args.out_dir}/confusion.npz")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm_dir", default="/data/zhenqian/models/esm2_t33_650M_UR50D")
    ap.add_argument("--sae_dir", required=True)
    ap.add_argument("--layer", type=int, default=24)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_len", type=int, default=1022)
    ap.add_argument("--max_proteins", type=int, default=2000)
    ap.add_argument("--sample_size", type=int, default=200)
    ap.add_argument("--n_thresholds", type=int, default=20)
    ap.add_argument("--feat_chunk", type=int, default=512)
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
