"""M1 — Fit a triplet-similarity alignment head on frozen SigLIP-So400m features.

Pipeline:
  1. Cache SigLIP-So400m pooled image features for all 1852 usable THINGS concepts (float16).
  2. Define a small 2-layer MLP head: 1152 -> 1152 (residual) trained to make cosine similarity
     of head-projected embeddings agree with human triplet odd-one-out judgments.
  3. Train on trainset.txt triplets with a KL / cross-entropy loss on the softmax over the 3
     pairwise similarities, with the odd-one-out being the *least* similar pair partner.
  4. Evaluate on testset1.txt (heldout): report `teacher_triplet_accuracy` for
     (i) the fit teacher head, (ii) the unaligned SigLIP baseline, (iii) chance (1/3).
     Compute bootstrap CI95 on all three.

Verifies Claim 1a. Uses SigLIP-So400m's *frozen* image tower — only the head trains.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import (
    load_siglip_vision,
    load_things_concepts,
    things_image_paths,
    load_triplets,
    ThingsImageDataset,
    seed_everything,
)


class AlignmentHead(nn.Module):
    """Small MLP head: proj = layernorm(gelu(fc1(x))) + x   (residual to preserve base info)."""

    def __init__(self, dim: int = 1152, hidden: int = 1152):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.fc2(F.gelu(self.fc1(self.ln1(x))))
        return self.ln2(x + h)


def cache_siglip_features(device: str, batch_size: int = 32) -> tuple[torch.Tensor, list[int]]:
    """Return (features [N, 1152] fp32, usable_concept_ids)."""
    print("[cache] loading SigLIP-So400m ...")
    model, proc = load_siglip_vision(dtype=torch.float16, device=device)
    ds = ThingsImageDataset(proc)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=False)
    feats = []
    cids = []
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(dl, desc="siglip forward"):
            pv = batch["pixel_values"].to(device, dtype=torch.float16)
            out = model.vision_model(pv)
            pool = out.pooler_output.float().cpu()
            feats.append(pool)
            cids.extend(batch["concept_id"].tolist())
    feats = torch.cat(feats, dim=0)
    print(f"[cache] {len(cids)} features, {feats.shape}, took {time.time()-t0:.1f}s")
    return feats, cids


def compute_triplet_choice_matrix(triplets: np.ndarray, sim_matrix: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """For each triplet [a, b, c] where c is the odd-one-out (ground truth),
    compute the model's predicted odd-one-out = argmin over i∈{a,b,c} of avg similarity
    to the other two.

    Returns (predicted_ooo, ground_truth_ooo) as int arrays of length N.
    """
    # sim: [1854, 1854]; but sim only indexed over usable ids
    a = triplets[:, 0]; b = triplets[:, 1]; c = triplets[:, 2]
    sab = sim_matrix[a, b].cpu().numpy()
    sac = sim_matrix[a, c].cpu().numpy()
    sbc = sim_matrix[b, c].cpu().numpy()
    # for candidate a: mean sim to (b,c) = (sab + sac) / 2 ; if small, a is odd
    # For each row, the odd-one-out has the SMALLEST average similarity to the other two.
    ma = (sab + sac) / 2
    mb = (sab + sbc) / 2
    mc = (sac + sbc) / 2
    means = np.stack([ma, mb, mc], axis=1)     # [N, 3]
    pred_row = np.argmin(means, axis=1)         # 0 -> a is odd, 1 -> b, 2 -> c
    # Map back to concept id
    pred_ooo = np.where(pred_row == 0, a, np.where(pred_row == 1, b, c))
    return pred_ooo, c  # c is the ground-truth odd-one-out (per dataset_description.txt)


def bootstrap_ci95(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = x[idx].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def evaluate_triplet_accuracy(head: AlignmentHead | None, feats: torch.Tensor, cid_to_row: dict,
                              triplets: np.ndarray, device: str) -> dict:
    """Evaluate triplet accuracy using given features (optionally transformed by head)."""
    with torch.no_grad():
        if head is not None:
            h = head.to(device).eval()
            # feats: [N_usable, 1152] fp32
            proj = h(feats.to(device))
        else:
            proj = feats.to(device)
        # normalize
        proj = F.normalize(proj, dim=-1)
        sim_usable = proj @ proj.T   # [N_u, N_u]
        # Build a global 1854x1854 dense sim tensor, filling missing rows/cols with -inf
        sim_global = torch.full((1854, 1854), float("-inf"), device=device)
        rows = torch.tensor([cid_to_row[c] for c in range(1854) if c in cid_to_row], dtype=torch.long)
        usable_cids = torch.tensor([c for c in range(1854) if c in cid_to_row], dtype=torch.long)
        # Fill in
        idx = usable_cids.to(device)
        sim_global[idx.unsqueeze(1), idx.unsqueeze(0)] = sim_usable
    # Restrict triplets to those where all three ids are usable
    usable_set = set(cid_to_row.keys())
    mask = np.array([(a in usable_set) and (b in usable_set) and (c in usable_set) for a, b, c in triplets])
    tr_use = triplets[mask]
    if len(tr_use) == 0:
        return {"accuracy": float("nan"), "n": 0}
    pred, gt = compute_triplet_choice_matrix(tr_use, sim_global)
    correct = (pred == gt).astype(np.int32)
    acc = float(correct.mean())
    lo, hi = bootstrap_ci95(correct)
    return {
        "accuracy": acc,
        "bootstrap_ci_95": [lo, hi],
        "n_triplets_used": int(len(tr_use)),
        "n_triplets_input": int(len(triplets)),
        "n_dropped_missing_image": int((~mask).sum()),
    }


def train_head(feats: torch.Tensor, cid_to_row: dict, triplets_train: np.ndarray,
               device: str, epochs: int = 3, batch_size: int = 4096, lr: float = 1e-3,
               train_subsample: int | None = 300_000, temperature: float = 1.0) -> AlignmentHead:
    """Train the alignment head on THINGS training triplets.

    We compute a KL-style cross-entropy: for each triplet [a,b,c] with c the odd-one-out,
    the target distribution over the 3 candidate-odd-one-outs is [0, 0, 1] (c is the odd one).
    Model logits are the *negative* mean-similarities (smaller = more likely odd).
    """
    seed_everything(42)
    head = AlignmentHead(dim=feats.shape[1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    n = len(triplets_train)
    if train_subsample is not None and train_subsample < n:
        rng = np.random.default_rng(42)
        pick = rng.choice(n, size=train_subsample, replace=False)
        tr = triplets_train[pick]
    else:
        tr = triplets_train
    # Filter to usable triplets
    usable_set = set(cid_to_row.keys())
    mask = np.array([(a in usable_set) and (b in usable_set) and (c in usable_set) for a, b, c in tr])
    tr = tr[mask]
    print(f"[train] using {len(tr)} triplets ({(~mask).sum()} dropped for missing images)")

    # Map concept ids to feature-row indices
    rows_a = torch.tensor([cid_to_row[int(x)] for x in tr[:, 0]], dtype=torch.long)
    rows_b = torch.tensor([cid_to_row[int(x)] for x in tr[:, 1]], dtype=torch.long)
    rows_c = torch.tensor([cid_to_row[int(x)] for x in tr[:, 2]], dtype=torch.long)
    F_dev = feats.to(device)
    losses_ep = []
    accs_ep = []
    for ep in range(epochs):
        perm = torch.randperm(len(tr))
        rows_a_p = rows_a[perm]; rows_b_p = rows_b[perm]; rows_c_p = rows_c[perm]
        total_loss = 0.0
        n_batch = 0
        n_correct = 0
        n_total = 0
        for i in tqdm(range(0, len(tr), batch_size), desc=f"epoch {ep+1}"):
            ba = rows_a_p[i:i+batch_size].to(device)
            bb = rows_b_p[i:i+batch_size].to(device)
            bc = rows_c_p[i:i+batch_size].to(device)
            head.train()
            fa = head(F_dev[ba])
            fb = head(F_dev[bb])
            fc = head(F_dev[bc])
            fa = F.normalize(fa, dim=-1); fb = F.normalize(fb, dim=-1); fc = F.normalize(fc, dim=-1)
            sab = (fa*fb).sum(-1); sac = (fa*fc).sum(-1); sbc = (fb*fc).sum(-1)
            # Odd-one-out score = negative average similarity to the other two.
            # For each item in triplet {a,b,c}, mean sim to the other two:
            ma = (sab + sac) / 2
            mb = (sab + sbc) / 2
            mc = (sac + sbc) / 2
            # logits: -mean * (1/T) then softmax => higher for odd
            logits = -torch.stack([ma, mb, mc], dim=1) / temperature
            # target: position 2 (c is odd)
            target = torch.full((len(ba),), 2, dtype=torch.long, device=device)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach())
            n_batch += 1
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                n_correct += int((pred == target).sum().item())
                n_total += int(len(ba))
        avg = total_loss / max(1, n_batch)
        acc = n_correct / max(1, n_total)
        losses_ep.append(avg)
        accs_ep.append(acc)
        print(f"  ep{ep+1}: loss={avg:.4f} train-triplet-acc={acc:.4f}")
    head.eval()
    return head, {"train_loss_per_epoch": losses_ep, "train_acc_per_epoch": accs_ep}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--split_train", default="train")
    p.add_argument("--split_eval", default="heldout")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train_subsample", type=int, default=300_000,
                   help="cap on training triplets (they are cheap so this is soft — full ~4M works but wastes wall-clock)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--feat_cache", default=None,
                   help="if set, load/save SigLIP feats to this path to skip re-forward")
    p.add_argument("--sanity_only", action="store_true", help="tiny run: 2 epochs 5k triplets subsample, for pipeline sanity")
    args = p.parse_args()

    seed_everything(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Feature cache
    if args.feat_cache and Path(args.feat_cache).exists():
        print(f"[cache] loading cached features from {args.feat_cache}")
        blob = torch.load(args.feat_cache, map_location="cpu")
        feats = blob["feats"]; cids = blob["cids"]
    else:
        feats, cids = cache_siglip_features(device=device)
        if args.feat_cache:
            torch.save({"feats": feats, "cids": cids}, args.feat_cache)
            print(f"[cache] saved to {args.feat_cache}")

    cid_to_row = {int(c): i for i, c in enumerate(cids)}

    # 2) Baseline eval (unaligned)
    print("[eval] unaligned SigLIP triplet accuracy on", args.split_eval)
    triplets_eval = load_triplets(args.split_eval)
    baseline = evaluate_triplet_accuracy(head=None, feats=feats, cid_to_row=cid_to_row,
                                          triplets=triplets_eval, device=device)
    print("  baseline:", baseline)

    # 3) Train head
    triplets_train = load_triplets(args.split_train)
    if args.sanity_only:
        args.epochs = 1
        args.train_subsample = 5000
    head, train_stats = train_head(
        feats=feats, cid_to_row=cid_to_row, triplets_train=triplets_train,
        device=device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        train_subsample=args.train_subsample, temperature=args.temperature,
    )
    torch.save(head.state_dict(), out / "teacher_head.pt")
    print(f"[save] teacher_head.pt -> {out/'teacher_head.pt'}")

    # 4) Evaluate fit head
    print("[eval] fit head triplet accuracy on", args.split_eval)
    fitted = evaluate_triplet_accuracy(head=head, feats=feats, cid_to_row=cid_to_row,
                                        triplets=triplets_eval, device=device)
    print("  fit:", fitted)

    # 5) Chance baseline (1/3)
    n_use = fitted["n_triplets_used"]
    chance = {"accuracy": 1.0 / 3.0, "bootstrap_ci_95": [0.333, 0.333], "n_triplets_used": n_use}

    # 6) Write summary
    result = {
        "milestone": "M1",
        "verifies": ["C1a"],
        "teacher_triplet_accuracy": fitted["accuracy"],
        "teacher_triplet_accuracy_ci95": fitted["bootstrap_ci_95"],
        "unaligned_siglip_triplet_accuracy": baseline["accuracy"],
        "unaligned_siglip_triplet_accuracy_ci95": baseline["bootstrap_ci_95"],
        "chance_baseline": chance["accuracy"],
        "n_triplets_heldout": n_use,
        "n_triplets_dropped_missing_image": fitted["n_dropped_missing_image"],
        "train_stats": train_stats,
        "config": {k: v for k, v in vars(args).items()},
    }
    with open(out / "eval_heldout.json", "w") as f:
        json.dump(result, f, indent=2)
    print("[done] eval_heldout.json ->", out/"eval_heldout.json")
    print(json.dumps({k: v for k, v in result.items() if k not in ("train_stats", "config")}, indent=2))


if __name__ == "__main__":
    main()
