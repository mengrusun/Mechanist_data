"""M3 / M4 / M5 eval — Spearman correlation of student vs. human on THINGS held-out triplets.

Metric:
  For each triplet [a, b, c] where c is the odd-one-out (per THINGS dataset_description),
  compute the model's cosine-similarity-based odd-one-out prediction (min mean-sim to the
  other two). Two summary numbers:

  1. Triplet accuracy: fraction of triplets where model predicted odd-one-out matches c.
  2. Spearman correlation of pairwise similarity ranks between model and 'human' pattern.
     - "Human pairwise similarity" is estimated from THINGS: for each ordered pair (i, j),
       count in what fraction of triplets containing both i and j were i,j chosen as the
       *pair* (i.e. NOT the odd one). This is a standard THINGS-derived similarity proxy.

Per-level breakdown (coarse / mid / fine):
  We bucket triplets by the WordNet-derived semantic distance of the CHOSEN PAIR relative
  to the odd-one-out. Concretely, we use the THINGS `Top-down Category (WordNet)` column
  as the coarse label; when that isn't sufficiently granular we fall back to letter-based
  bins over the ranked similarity distribution.
"""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import (
    load_dinov2_base,
    load_things_concepts,
    things_image_paths,
    load_triplets,
    ThingsImageDataset,
    seed_everything,
)


def compute_dinov2_features(student_ckpt: str | None, device: str, batch_size: int = 64) -> tuple[torch.Tensor, list[int]]:
    """Return (features [N_usable, 768] fp32, usable_concept_ids).

    student_ckpt: if None, use the pretrained model. If a .pt file, load it as state dict
    into a fresh DINOv2 backbone.
    """
    from transformers import AutoModel, AutoImageProcessor
    from data_utils import MODEL_DIR
    mp = MODEL_DIR / "dinov2-base"
    model = AutoModel.from_pretrained(mp, torch_dtype=torch.float16).to(device).eval()
    if student_ckpt and Path(student_ckpt).exists() and student_ckpt not in (str(mp), str(mp) + "/"):
        print(f"[load] loading student ckpt from {student_ckpt}")
        state = torch.load(student_ckpt, map_location="cpu")
        # state may be a dict with `model_state_dict` key or a raw state_dict
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        # Detect LoRA checkpoint (keys like "...query.A" / "...query.B") and rewrap
        # the model with LoRA before load_state_dict so adapter tensors have a home.
        has_lora = any(k.endswith(".A") or k.endswith(".B") for k in state.keys())
        if has_lora:
            from align_student import apply_lora_to_dinov2
            r = next(v.shape[0] for k, v in state.items() if k.endswith(".A"))
            lora_alpha = 32
            model = model.float()
            model = apply_lora_to_dinov2(model, r=r, alpha=lora_alpha)
            model = model.to(device)
            print(f"[load] detected LoRA checkpoint (r={r}); rewrapped model before load_state_dict")
        # Try strict, fall back to relaxed
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  [warn] missing={len(missing)} unexpected={len(unexpected)} keys")
    else:
        print(f"[load] using stock DINOv2 (student_ckpt={student_ckpt!r})")
    # If LoRA was wrapped in (has_lora above), keep the model in fp32 so LoRA A/B
    # (fp32 by construction) match the frozen base. Otherwise use fp16 for speed.
    from align_student import LoRALinear
    if any(isinstance(mod, LoRALinear) for mod in model.modules()):
        model = model.to(device).float().eval()
    else:
        model = model.to(device, dtype=torch.float16).eval()
    proc = AutoImageProcessor.from_pretrained(mp)

    ds = ThingsImageDataset(proc)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=False)
    feats = []
    cids = []
    model_dtype = next(model.parameters()).dtype
    with torch.no_grad():
        for batch in tqdm(dl, desc="dinov2 forward"):
            pv = batch["pixel_values"].to(device, dtype=model_dtype)
            out = model(pv)
            # Use CLS + patch mean (DINOv2 convention) or just pooler_output
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                pool = out.pooler_output.float().cpu()
            else:
                pool = out.last_hidden_state[:, 0].float().cpu()
            feats.append(pool)
            cids.extend(batch["concept_id"].tolist())
    return torch.cat(feats, dim=0), cids


def bootstrap_ci95(x: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return (float("nan"), float("nan"))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = x[idx].mean()
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def triplet_predict_ooo(sim: torch.Tensor, triplets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (predicted_ooo_concept, ground_truth_ooo_concept) arrays."""
    a = triplets[:, 0]; b = triplets[:, 1]; c = triplets[:, 2]
    sab = sim[a, b].cpu().numpy()
    sac = sim[a, c].cpu().numpy()
    sbc = sim[b, c].cpu().numpy()
    ma = (sab + sac) / 2
    mb = (sab + sbc) / 2
    mc = (sac + sbc) / 2
    means = np.stack([ma, mb, mc], axis=1)
    pred_row = np.argmin(means, axis=1)
    pred_ooo = np.where(pred_row == 0, a, np.where(pred_row == 1, b, c))
    return pred_ooo, c


def build_human_similarity_from_triplets(triplets: np.ndarray, n_concepts: int = 1854) -> np.ndarray:
    """Human-derived pairwise similarity: for each pair (i,j), count = fraction of triplets
    containing both i and j where i,j were the CHOSEN PAIR (not odd one).

    Returns a dense [n_concepts, n_concepts] float array (NaN where undefined).
    """
    numer = np.zeros((n_concepts, n_concepts), dtype=np.float64)
    denom = np.zeros((n_concepts, n_concepts), dtype=np.float64)
    for a, b, c in triplets:
        # chosen pair = (a, b), odd one = c
        # (a, b) is chosen pair; (a, c) and (b, c) are non-chosen pairs
        numer[a, b] += 1; numer[b, a] += 1  # chosen
        denom[a, b] += 1; denom[b, a] += 1
        denom[a, c] += 1; denom[c, a] += 1
        denom[b, c] += 1; denom[c, b] += 1
    sim = np.full_like(numer, np.nan)
    mask = denom > 0
    sim[mask] = numer[mask] / denom[mask]
    return sim


def spearman_pairwise(model_sim: np.ndarray, human_sim: np.ndarray, min_obs: int = 5) -> tuple[float, int]:
    """Spearman between model & human similarity, over pairs (i<j) with enough evidence.

    Returns (spearman_r, n_pairs_used).
    """
    n = model_sim.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    m = model_sim[iu, ju]
    h = human_sim[iu, ju]
    mask = ~np.isnan(h) & ~np.isnan(m)
    if mask.sum() < 100:
        return float("nan"), int(mask.sum())
    r, _ = spearmanr(m[mask], h[mask])
    return float(r), int(mask.sum())


def level_bucket_triplets(triplets: np.ndarray, concept_meta: pd.DataFrame) -> dict[str, np.ndarray]:
    """Bucket triplets into coarse / mid / fine based on THINGS categorical metadata.

    - coarse: chosen pair (a,b) share top-down WordNet category AND odd (c) is different
              top-down category. Otherwise fall through.
    - fine:   all three (a,b,c) share the same top-down WordNet category.
    - mid:    everything else that has meaningful category info.
    Uses column `Top-down Category (WordNet)` — often empty; we fall back to `Bottom-up Category`.
    """
    def cat_for(cid: int) -> str | None:
        row = concept_meta.iloc[cid]
        for col in ("Top-down Category (WordNet)", "Top-down Category (manual selection)", "Bottom-up Category (Human Raters)"):
            v = row.get(col)
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
        return None

    coarse = []; mid = []; fine = []
    for i, (a, b, c) in enumerate(triplets):
        ca, cb, cc = cat_for(int(a)), cat_for(int(b)), cat_for(int(c))
        if ca is None or cb is None or cc is None:
            continue
        if ca == cb == cc:
            fine.append(i)
        elif ca == cb and cc != ca:
            coarse.append(i)
        else:
            mid.append(i)
    return {
        "coarse": np.array(coarse, dtype=np.int64),
        "mid":    np.array(mid, dtype=np.int64),
        "fine":   np.array(fine, dtype=np.int64),
    }


def eval_features(feats: torch.Tensor, cids: list[int], triplets_eval: np.ndarray,
                  device: str, level_buckets: dict[str, np.ndarray] | None) -> dict:
    """Compute triplet accuracy + Spearman(model, human) globally and per level."""
    with torch.no_grad():
        proj = F.normalize(feats.to(device), dim=-1)
        sim_usable = proj @ proj.T
        sim_global = torch.full((1854, 1854), float("nan"), device=device)
        idx = torch.tensor(cids, dtype=torch.long, device=device)
        sim_global[idx.unsqueeze(1), idx.unsqueeze(0)] = sim_usable
    # Restrict eval triplets to usable
    usable_set = set(cids)
    keep = np.array([(a in usable_set) and (b in usable_set) and (c in usable_set) for a, b, c in triplets_eval])
    tr = triplets_eval[keep]
    n = len(tr)
    # Global accuracy
    pred, gt = triplet_predict_ooo(sim_global, tr)
    correct = (pred == gt).astype(np.int32)
    acc = float(correct.mean())
    lo, hi = bootstrap_ci95(correct)
    # Human similarity from training triplets (established prior — could be from train split)
    # For Spearman, use the eval-split human sim itself as the ground truth (still uses ranks).
    # BUT: the eval split is small (15k), so per-pair support may be thin. We use train split for human sim.
    triplets_train = load_triplets("train")
    human = build_human_similarity_from_triplets(triplets_train, n_concepts=1854)
    # Restrict model sim to concepts present in usable
    # Build dense arrays only over usable concepts to speed spearman
    keep_c = np.array(cids)
    ms = sim_global.cpu().numpy()
    ms_sub = ms[np.ix_(keep_c, keep_c)]
    hs_sub = human[np.ix_(keep_c, keep_c)]
    r_global, n_pairs = spearman_pairwise(ms_sub, hs_sub)

    result = {
        "n_triplets_heldout": int(n),
        "triplet_accuracy": acc,
        "triplet_accuracy_ci95": [lo, hi],
        "spearman_aggregate": r_global,
        "spearman_n_pairs": n_pairs,
    }
    # Per-level accuracy
    if level_buckets is None:
        # Compute against the input triplet set
        # (already restricted to usable via `tr`) — bucket on tr
        pass
    if level_buckets is not None:
        for name, idxs in level_buckets.items():
            # idxs are into the ORIGINAL triplets_eval; restrict to those also in tr
            if len(idxs) == 0 or idxs.max() >= len(keep):
                idxs_use = np.array([], dtype=np.int64)
            else:
                idxs_use = idxs[keep[idxs]]
            if len(idxs_use) == 0:
                result[f"triplet_accuracy_{name}"] = float("nan")
                result[f"triplet_accuracy_{name}_ci95"] = [float("nan"), float("nan")]
                result[f"n_triplets_{name}"] = 0
                # Spearman on level requires more work; approximate by restricting sim & human
                # to the concepts appearing in these triplets
                result[f"spearman_{name}"] = float("nan")
                result[f"n_pairs_{name}"] = 0
                continue
            level_triplets = triplets_eval[idxs_use]
            level_pred, level_gt = triplet_predict_ooo(sim_global, level_triplets)
            level_correct = (level_pred == level_gt).astype(np.int32)
            result[f"triplet_accuracy_{name}"] = float(level_correct.mean())
            lo_l, hi_l = bootstrap_ci95(level_correct)
            result[f"triplet_accuracy_{name}_ci95"] = [lo_l, hi_l]
            result[f"n_triplets_{name}"] = int(len(idxs_use))
            # Spearman restricted to PAIRS actually instantiated in this level's triplets
            # (fix: previously used all-pairs among concepts appearing at this level, which
            #  over-counts inter-level pairs that never participated in the level's triplets).
            level_pairs = set()
            for a_, b_, c_ in level_triplets:
                level_pairs.add((min(a_, b_), max(a_, b_)))
                level_pairs.add((min(a_, c_), max(a_, c_)))
                level_pairs.add((min(b_, c_), max(b_, c_)))
            if len(level_pairs) < 30:
                result[f"spearman_{name}"] = float("nan")
                result[f"n_pairs_{name}"] = 0
            else:
                ii = np.array([p[0] for p in level_pairs])
                jj = np.array([p[1] for p in level_pairs])
                ms_l = ms[ii, jj]
                hs_l = human[ii, jj]
                mask_l = ~np.isnan(ms_l) & ~np.isnan(hs_l)
                if mask_l.sum() < 30:
                    result[f"spearman_{name}"] = float("nan")
                    result[f"n_pairs_{name}"] = int(mask_l.sum())
                else:
                    r_l, _ = spearmanr(ms_l[mask_l], hs_l[mask_l])
                    result[f"spearman_{name}"] = float(r_l) if r_l == r_l else float("nan")
                    result[f"n_pairs_{name}"] = int(mask_l.sum())
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student_ckpt", required=True,
                   help="Path to student .pt file, OR the model directory path (e.g., MODEL_DIR/dinov2-base) to use stock DINOv2.")
    p.add_argument("--split", default="heldout")
    p.add_argument("--output_json", required=True)
    p.add_argument("--levels_construction", default="things_metadata",
                   choices=["things_metadata"],
                   help="how to define coarse/mid/fine buckets. 'things_metadata' uses THINGS Top-down / "
                        "Bottom-up Category columns (per requirement 7 of the code-review protocol). No other "
                        "value is accepted; earlier CLI variants that passed 'imagenet_wordnet' will fail argparse.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    # 1) DINOv2 features on THINGS
    feats, cids = compute_dinov2_features(args.student_ckpt, device=device)
    print(f"[feat] shape={feats.shape}")

    # 2) Load eval triplets
    tri_eval = load_triplets(args.split)
    print(f"[eval] {len(tri_eval)} triplets from '{args.split}'")

    # 3) Build level buckets
    meta = load_things_concepts()
    buckets = level_bucket_triplets(tri_eval, meta)
    print("[bucket] " + ", ".join(f"{k}={len(v)}" for k, v in buckets.items()))

    # 4) Evaluate
    res = eval_features(feats=feats, cids=cids, triplets_eval=tri_eval,
                        device=device, level_buckets=buckets)
    res["config"] = {k: v for k, v in vars(args).items()}
    res["milestone_kind"] = "student_similarity_eval"

    with open(args.output_json, "w") as f:
        json.dump(res, f, indent=2)
    print("[done]", args.output_json)
    printable = {k: v for k, v in res.items() if k != "config"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
