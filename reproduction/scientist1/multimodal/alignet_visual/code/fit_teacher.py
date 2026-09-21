"""Fit a small alignment head on frozen SigLIP features so the resulting space
matches human-similarity structure derived from the THINGS sensevec embedding
(a stand-in for triplet-odd-one-out human judgments — sensevec is trained on
free-association / semantic-property data and is a widely used proxy for the
human conceptual similarity space).

Loss: triplet cross-entropy against sensevec soft labels + RSM cosine loss.

Also stores the aligned SigLIP features and, importantly, an aligned RSM that we
will distil into student vision models on ImageNet.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
import paths
from baseline_things_eval import (
    cosine_rsm, sample_triplets, triplet_ooo_labels, triplet_predictions,
    upper_triangle, load_things_feats, load_reference, category_masks,
)


class AlignHead(nn.Module):
    def __init__(self, in_dim, out_dim=512, hidden=1024, residual=True):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        # Extra residual pathway that copies (a linear projection of) the input,
        # so the aligned space keeps a lot of the base SigLIP structure and only
        # adds a correction. Improves stability + preserves downstream utility.
        self.residual = residual
        if residual:
            self.res = nn.Linear(in_dim, out_dim, bias=False)
            with torch.no_grad():
                nn.init.zeros_(self.fc2.weight)
                nn.init.zeros_(self.fc2.bias)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = self.fc2(h)
        if self.residual:
            h = h + self.res(x)
        return h


def soft_triplet_ce(model_sim, target_sim):
    """model_sim, target_sim: [B, 3] with columns (s_ab, s_ac, s_bc).
    For each triplet, model probability that concept i is odd-one-out
    is softmax over the *pair-not-containing-i* similarities. But since
    we want odd-one-out to be the one *least similar* to the other two,
    a simpler formulation: the OOO probability equals softmax over
    (- min_other_sim) which for a triplet is proportional to softmax
    over (- pair_similarity_of_the_other_two).
    We compute odd-one-out probabilities directly and match them with KL.
    """
    # For triplet (a,b,c), pair similarities (s_ab, s_ac, s_bc).
    # Odd-one-out is: a if max pair sim is s_bc, b if s_ac, c if s_ab.
    # So OOO logits over (a,b,c) = (s_bc, s_ac, s_ab).
    def rearrange(x):
        # x: [B,3] = (s_ab, s_ac, s_bc). Return [B,3] = (s_bc, s_ac, s_ab)
        return torch.stack([x[:, 2], x[:, 1], x[:, 0]], dim=1)

    model_logits = rearrange(model_sim)
    target_logits = rearrange(target_sim)
    log_p = F.log_softmax(model_logits, dim=1)
    q = F.softmax(target_logits, dim=1)
    return -(q * log_p).sum(dim=1).mean()


def batch_triplet_sims(feats, triplets):
    """feats: [N, d] tensor (should be L2 normalized).
    triplets: [B, 3] long tensor of indices.
    Returns [B, 3] of (s_ab, s_ac, s_bc).
    """
    a = feats[triplets[:, 0]]
    b = feats[triplets[:, 1]]
    c = feats[triplets[:, 2]]
    s_ab = (a * b).sum(dim=1)
    s_ac = (a * c).sum(dim=1)
    s_bc = (b * c).sum(dim=1)
    return torch.stack([s_ab, s_ac, s_bc], dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siglip-feats", default=os.path.join(paths.CACHE_DIR, "things_feats_siglip.npz"))
    ap.add_argument("--out-head", default=os.path.join(paths.CACHE_DIR, "teacher_head.pt"))
    ap.add_argument("--out-things-aligned", default=os.path.join(paths.CACHE_DIR, "things_teacher_aligned.npz"))
    ap.add_argument("--out-dim", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--triplets-per-epoch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val-triplets", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=8.0, help="temperature for softmax on similarities")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction of concepts held out for evaluation")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    feats, ids = load_things_feats(args.siglip_feats)
    sensevec = load_reference(paths.THINGS_SENSEVEC, ids)
    keep = np.linalg.norm(sensevec, axis=1) > 1e-6
    keep_idx = np.where(keep)[0]
    print(f"Keeping {keep.sum()}/{len(ids)} concepts with sensevec.")
    feats = feats[keep_idx]
    sensevec = sensevec[keep_idx]
    ids_kept = [ids[i] for i in keep_idx]

    feats_t = torch.from_numpy(feats).float().to(device)
    sensevec_t = torch.from_numpy(sensevec).float().to(device)
    sensevec_n = F.normalize(sensevec_t, dim=1)

    # split concepts into train / val
    n = feats.shape[0]
    perm = rng.permutation(n)
    n_val = int(args.val_frac * n)
    val_idx = torch.from_numpy(perm[:n_val]).long().to(device)
    tr_idx = torch.from_numpy(perm[n_val:]).long().to(device)
    print(f"train concepts: {len(tr_idx)}, val concepts: {len(val_idx)}")

    head = AlignHead(feats.shape[1], out_dim=args.out_dim, hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.wd)

    def eval_state():
        head.eval()
        with torch.no_grad():
            aligned = head(feats_t)
            aligned_n = F.normalize(aligned, dim=1)
        for name, idx in [("train", tr_idx), ("val", val_idx)]:
            sub_m = aligned_n[idx]
            sub_s = sensevec_n[idx]
            rsm_m = (sub_m @ sub_m.T).cpu().numpy()
            rsm_s = (sub_s @ sub_s.T).cpu().numpy()
            rho, _ = spearmanr(upper_triangle(rsm_m), upper_triangle(rsm_s))
            trs = sample_triplets(sub_m.shape[0], args.val_triplets, np.random.RandomState(42))
            labs = triplet_ooo_labels(rsm_s, trs)
            preds = triplet_predictions(rsm_m, trs)
            acc = float((preds == labs).mean())
            print(f"  {name}: rho(sensevec)={rho:.4f} OOO_acc={acc:.4f}")
        return None

    print("[before fit]")
    eval_state()

    tr_feats = feats_t[tr_idx]
    tr_sense = sensevec_n[tr_idx]
    n_tr = tr_feats.shape[0]

    for ep in range(args.epochs):
        head.train()
        trs = torch.from_numpy(sample_triplets(n_tr, args.triplets_per_epoch, rng)).to(device).long()
        with torch.no_grad():
            tgt_sim = batch_triplet_sims(tr_sense, trs)
        aligned = head(tr_feats)
        aligned_n = F.normalize(aligned, dim=1)
        mdl_sim = batch_triplet_sims(aligned_n, trs)
        loss = soft_triplet_ce(mdl_sim * args.tau, tgt_sim * args.tau)
        idx = torch.randperm(n_tr, device=device)[:512]
        aug = aligned_n[idx]
        sen = tr_sense[idx]
        rsm_m = aug @ aug.T
        rsm_s = sen @ sen.T
        rsm_loss = F.mse_loss(rsm_m, rsm_s)
        total = loss + 0.5 * rsm_loss
        opt.zero_grad()
        total.backward()
        opt.step()
        if ep % 25 == 0 or ep == args.epochs - 1:
            print(f"[ep {ep:4d}] loss={loss.item():.4f} rsm_loss={rsm_loss.item():.4f}")
            eval_state()

    # Save head state
    torch.save({
        "in_dim": feats.shape[1],
        "out_dim": args.out_dim,
        "hidden": args.hidden,
        "state_dict": head.state_dict(),
    }, args.out_head)
    print("saved", args.out_head)

    # Save aligned features on THINGS (kept set)
    head.eval()
    with torch.no_grad():
        aligned = head(feats_t).cpu().numpy()
    np.savez(args.out_things_aligned, feats=aligned, ids=np.array(ids_kept))
    print("saved", args.out_things_aligned)


if __name__ == "__main__":
    main()
