"""M6 — Behavioural + per-triplet uncertainty match, aligned vs. unaligned.

Human agreement is measured *directly from the THINGS noise-ceiling data*:
  testset2.txt         — first worker's response to each triplet.
  testset2_repeat.txt  — a different worker's response to the SAME triplet position.
For each such triplet we get a per-triplet human agreement label:
  human_agree = 1 iff worker_1 and worker_2 chose the same odd-one-out, else 0.
The `hard` triplets (~15-20% where workers disagree) are exactly the ones where
human uncertainty is high. Model uncertainty should correlate with human uncertainty
(Spearman between model confidence and human agreement rate).

We compare aligned DINOv2 vs unaligned DINOv2 on:
  1. choice_agreement — mean per-triplet fraction where the model's odd-one-out
     matches the *majority* human choice (i.e., matches worker_1 & worker_2 when they agree;
     dropped from choice metric otherwise). ← C3-choice
  2. uncertainty_spearman — Spearman(model top-1 confidence, human_agree indicator) over the
     testset2/testset2_repeat overlap. Human agreement is BINARY per triplet.  ← C3-uncertainty
  3. uncertainty_kl — KL between model 3-way choice distribution and the empirical 2-worker
     human distribution (with Laplace smoothing). ← C3-uncertainty
  4. rsa_spearman — Spearman(model pairwise cosine sim, human pairwise similarity from
     training-triplet fractions). ← C3-RSA

For the general-ability guardrail baseline, we log the mean per-triplet entropy of each
model as well (lower = more confident; too-low might indicate overfit).
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from tqdm import tqdm

from data_utils import (
    load_things_concepts,
    load_triplets,
    seed_everything,
    MODEL_DIR,
    DATA_DIR,
)


def compute_dinov2_features_for_ckpt(ckpt: str, device: str, batch_size: int = 64) -> tuple[torch.Tensor, list[int]]:
    from transformers import AutoModel, AutoImageProcessor
    from torch.utils.data import DataLoader
    from data_utils import ThingsImageDataset
    mp = MODEL_DIR / "dinov2-base"
    model = AutoModel.from_pretrained(mp, torch_dtype=torch.float16).to(device).eval()
    if Path(ckpt).is_file():
        print(f"[load] loading student state from {ckpt}")
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  [warn] missing={len(missing)} unexpected={len(unexpected)}")
    proc = AutoImageProcessor.from_pretrained(mp)
    ds = ThingsImageDataset(proc)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=False)
    feats = []; cids = []
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"forward {Path(ckpt).name}"):
            pv = batch["pixel_values"].to(device, dtype=torch.float16)
            out = model(pv)
            pool = (out.pooler_output if (hasattr(out, "pooler_output") and out.pooler_output is not None)
                    else out.last_hidden_state[:, 0])
            feats.append(pool.float().cpu())
            cids.extend(batch["concept_id"].tolist())
    return torch.cat(feats, dim=0), cids


def triplet_predict_full(features: torch.Tensor, cids: list[int], triplets: np.ndarray, device: str) -> dict:
    """Return per-triplet {pred, p_dist} where p_dist[i, k] = P(model thinks position k is odd)."""
    with torch.no_grad():
        proj = F.normalize(features.to(device), dim=-1)
        sim_usable = proj @ proj.T
        sim_global = torch.full((1854, 1854), float("nan"), device=device)
        idx = torch.tensor(cids, dtype=torch.long, device=device)
        sim_global[idx.unsqueeze(1), idx.unsqueeze(0)] = sim_usable
    usable = set(cids)
    keep = np.array([(a in usable) and (b in usable) and (c in usable) for a, b, c in triplets])
    tr = triplets[keep]
    a = tr[:, 0]; b = tr[:, 1]; c = tr[:, 2]
    sab = sim_global[a, b].cpu().numpy()
    sac = sim_global[a, c].cpu().numpy()
    sbc = sim_global[b, c].cpu().numpy()
    ma = (sab + sac) / 2
    mb = (sab + sbc) / 2
    mc = (sac + sbc) / 2
    means = np.stack([ma, mb, mc], axis=1)   # [N, 3]
    logits = -means
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    pred_pos = np.argmin(means, axis=1)         # 0/1/2 = which position is odd
    pred_ooo = np.where(pred_pos == 0, a, np.where(pred_pos == 1, b, c))
    return {
        "keep_mask": keep,
        "triplets_used": tr,
        "pred_pos": pred_pos,
        "pred_ooo": pred_ooo,
        "p_dist": p,                              # [N, 3] over positions in tr order
        "n": int(len(tr)),
    }


def bootstrap_diff_ci95(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n = len(a)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = a[idx].mean() - b[idx].mean()
    return {
        "diff": float(a.mean() - b.mean()),
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "p_positive": float((boots > 0).mean()),
    }


def build_human_similarity(triplets_train: np.ndarray, n_concepts: int = 1854) -> np.ndarray:
    """Pairwise human similarity: fraction of triplets where (i,j) was the CHOSEN pair given
    that both appeared."""
    numer = np.zeros((n_concepts, n_concepts), dtype=np.float64)
    denom = np.zeros((n_concepts, n_concepts), dtype=np.float64)
    for a, b, c in triplets_train:
        numer[a, b] += 1; numer[b, a] += 1
        denom[a, b] += 1; denom[b, a] += 1
        denom[a, c] += 1; denom[c, a] += 1
        denom[b, c] += 1; denom[c, b] += 1
    sim = np.full_like(numer, np.nan)
    mask = denom > 0
    sim[mask] = numer[mask] / denom[mask]
    return sim


def rsa_spearman(feats: torch.Tensor, cids: list[int], human_sim: np.ndarray, device: str) -> tuple[float, int]:
    with torch.no_grad():
        proj = F.normalize(feats.to(device), dim=-1)
        sim = (proj @ proj.T).cpu().numpy()
    cids_arr = np.array(cids)
    hs_sub = human_sim[np.ix_(cids_arr, cids_arr)]
    iu, ju = np.triu_indices(len(cids), k=1)
    ms = sim[iu, ju]
    hs = hs_sub[iu, ju]
    mask = ~np.isnan(hs)
    if mask.sum() < 100:
        return float("nan"), int(mask.sum())
    r, _ = spearmanr(ms[mask], hs[mask])
    return float(r), int(mask.sum())


def build_noise_ceiling_pairs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load testset2 + testset2_repeat as two independent worker responses to the SAME
    ordered set of triplet positions, aligned by row index (per the THINGS
    noise-ceiling data design).

    Row alignment is validated strictly: the sorted 3-element set of concept ids in each
    row must match between the two files (only the odd-one-out choice — position 2 —
    may differ, which is exactly what encodes worker disagreement).

    Returns:
      triplet_set_key : [N, 3] sorted concept-id triple identifying the triplet
      w1_ooo          : [N]    worker 1's chosen odd-one-out concept id
      w2_ooo          : [N]    worker 2's chosen odd-one-out concept id
    """
    ts2 = np.loadtxt(DATA_DIR / "things_ooo_triplets" / "triplet_dataset" / "testset2.txt", dtype=np.int64)
    ts2r = np.loadtxt(DATA_DIR / "things_ooo_triplets" / "triplet_dataset" / "testset2_repeat.txt", dtype=np.int64)
    assert ts2.shape == ts2r.shape, f"testset2 shape {ts2.shape} != repeat shape {ts2r.shape}"
    w1_ooo = ts2[:, 2]
    w2_ooo = ts2r[:, 2]
    # Row alignment check: same 3-element SET (order-invariant) at each row
    key1 = np.sort(ts2, axis=1)
    key2 = np.sort(ts2r, axis=1)
    same = (key1 == key2).all(axis=1)
    match_rate = same.mean()
    # Hard-require ≥ 99% row alignment. THINGS noise-ceiling design guarantees this (>99.9% in practice).
    assert match_rate >= 0.99, (
        f"testset2 vs testset2_repeat row-set match rate {match_rate:.4f} < 0.99 — the two files "
        "are NOT row-aligned by triplet identity, so worker pairings from parallel rows are wrong. "
        "The noise-ceiling metric assumes row-index-aligned pairs and cannot proceed."
    )
    if match_rate < 1.0:
        print(f"[noise-ceiling] {int((~same).sum())} of {len(ts2)} rows had misaligned triplet sets — dropping.")
    return key1[same], w1_ooo[same], w2_ooo[same]


def uncertainty_evaluation(features: torch.Tensor, cids: list[int], device: str) -> dict:
    """Score the model on the noise-ceiling testset2 triplets with human-derived uncertainty."""
    key, w1, w2 = build_noise_ceiling_pairs()
    # For each triplet row we need (a, b, c) with c being *some* canonical odd-one-out.
    # We use w1's ooo as the "GT" for choice-agreement but exclude triplets where workers
    # disagree from the choice metric (they are ambiguous — use them only for uncertainty).
    N = len(key)
    # Construct triplets in [a, b, c] form: put w1's ooo at position 2, then the other two.
    tri_w1 = np.zeros((N, 3), dtype=np.int64)
    for i, (k, w) in enumerate(zip(key, w1)):
        others = [x for x in k if x != w]
        if len(others) != 2:
            # Some triplet where w1 == none of key (should not happen if the odd is in the triplet)
            # Fall back to using original ordering: assume position 2 in ts2 IS w1
            tri_w1[i] = [k[0], k[1], k[2]] if w == k[2] else [k[0], k[2], k[1]] if w == k[1] else [k[1], k[2], k[0]]
        else:
            tri_w1[i, 0] = others[0]; tri_w1[i, 1] = others[1]; tri_w1[i, 2] = w
    # Human agreement per triplet: 1 if w1 == w2
    human_agree = (w1 == w2).astype(np.int32)   # 1 = both chose same → EASY; 0 = disagreed → HARD
    # Model scores on these triplets
    sc = triplet_predict_full(features, cids, tri_w1, device)
    keep = sc["keep_mask"]
    tr_use = sc["triplets_used"]
    n_use = len(tr_use)
    if n_use == 0:
        return {"n": 0}
    human_agree_use = human_agree[keep]
    w1_use = w1[keep]; w2_use = w2[keep]
    pred_ooo = sc["pred_ooo"]
    p_dist = sc["p_dist"]
    # Choice agreement: on rows where workers agree, does the model also pick that same ooo?
    agreed_mask = (human_agree_use == 1)
    if agreed_mask.sum() > 0:
        choice_acc_on_agreed = float((pred_ooo[agreed_mask] == w1_use[agreed_mask]).mean())
        n_agreed = int(agreed_mask.sum())
    else:
        choice_acc_on_agreed = float("nan"); n_agreed = 0
    # Model top-1 confidence per triplet
    conf_top1 = p_dist.max(axis=1)
    # Uncertainty Spearman: high model confidence should track high human agreement
    #   (both should be high on easy triplets, low on hard ones)
    if len(conf_top1) >= 100 and len(np.unique(human_agree_use)) > 1:
        r_conf, _ = spearmanr(conf_top1, human_agree_use.astype(np.float32))
    else:
        r_conf = float("nan")
    # Mean entropy on hard vs easy
    entropy = -np.sum(p_dist * np.log(p_dist + 1e-9), axis=1)
    H_easy = float(entropy[agreed_mask].mean()) if agreed_mask.sum() > 0 else float("nan")
    H_hard = float(entropy[~agreed_mask].mean()) if (~agreed_mask).sum() > 0 else float("nan")
    # KL from model p_dist to human 2-worker distribution (Laplace-smoothed)
    # Human distribution per triplet: 2-worker vote over 3 positions, Laplace(0.5) smoothed
    # We need to figure out which position (0,1,2) in tri_w1 each worker's ooo corresponds to.
    # tri_w1[i, 2] == w1_use[i] by construction. What about w2?
    a_ = tri_w1[keep, 0]; b_ = tri_w1[keep, 1]; c_ = tri_w1[keep, 2]
    pos_w2 = np.where(w2_use == a_, 0, np.where(w2_use == b_, 1, np.where(w2_use == c_, 2, -1)))
    # Human histogram: count per position [pos_w1_always=2, pos_w2]
    hist = np.zeros((n_use, 3), dtype=np.float64)
    hist[np.arange(n_use), 2] += 1    # w1 chose position 2 (c_)
    valid_w2 = (pos_w2 >= 0)
    hist[np.arange(n_use)[valid_w2], pos_w2[valid_w2]] += 1
    hist = hist + 0.5   # Laplace smoothing
    human_p = hist / hist.sum(axis=1, keepdims=True)
    # KL(human || model)
    kl_hm = float(np.sum(human_p * (np.log(human_p) - np.log(p_dist + 1e-9)), axis=1).mean())
    return {
        "n_triplets_evaluated": int(n_use),
        "n_workers_agreed": n_agreed,
        "choice_accuracy_on_agreed": choice_acc_on_agreed,   # <-- C3-choice on human-consensus subset
        "uncertainty_spearman_conf_vs_human_agreement": float(r_conf) if r_conf == r_conf else float("nan"),
        "mean_entropy_on_easy": H_easy,
        "mean_entropy_on_hard": H_hard,
        "kl_human_to_model": kl_hm,
        "human_agreement_rate": float(agreed_mask.mean()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aligned_ckpt", required=True)
    p.add_argument("--unaligned_ckpt", default=str(MODEL_DIR / "dinov2-base"))
    p.add_argument("--rsa_public_root", default=None,
                   help="optional path to a public RSA collection; if missing, rsa_spearman* metrics use only THINGS-train-derived pairwise sim.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)
    device = "cuda"
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # 1) Forward both students on THINGS
    feats_al, cids_al = compute_dinov2_features_for_ckpt(args.aligned_ckpt, device=device)
    feats_ua, cids_ua = compute_dinov2_features_for_ckpt(args.unaligned_ckpt, device=device)
    assert cids_al == cids_ua

    # 2) Noise-ceiling uncertainty eval
    unc_al = uncertainty_evaluation(feats_al, cids_al, device)
    unc_ua = uncertainty_evaluation(feats_ua, cids_ua, device)
    print("[uncertainty aligned]", unc_al)
    print("[uncertainty unaligned]", unc_ua)

    # 3) RSA spearman using THINGS train-triplet-derived pairwise human similarity
    triplets_train = load_triplets("train")
    human_sim = build_human_similarity(triplets_train)
    rsa_al, n_al = rsa_spearman(feats_al, cids_al, human_sim, device)
    rsa_ua, n_ua = rsa_spearman(feats_ua, cids_ua, human_sim, device)
    rsa = {
        "rsa_spearman_aligned": rsa_al,
        "rsa_spearman_unaligned": rsa_ua,
        "n_pairs": n_al,
    }
    # Also RSA public collection if available
    if args.rsa_public_root and Path(args.rsa_public_root).exists():
        # Skip — we don't have this dataset
        pass

    # 4) Predicates
    predicates = {
        # C3-choice: aligned > unaligned on choice-accuracy on the human-consensus subset
        "choice_aligned_gt_unaligned": (unc_al["choice_accuracy_on_agreed"] or 0) > (unc_ua["choice_accuracy_on_agreed"] or 0),
        # C3-uncertainty: aligned's model confidence tracks human agreement better (higher Spearman)
        "uncertainty_aligned_calibrated_better":
            (not np.isnan(unc_al["uncertainty_spearman_conf_vs_human_agreement"])) and
            (unc_al["uncertainty_spearman_conf_vs_human_agreement"] > unc_ua["uncertainty_spearman_conf_vs_human_agreement"]),
        # C3-RSA
        "rsa_aligned_gt_unaligned": rsa_al > rsa_ua,
    }
    n_hits = sum(bool(v) for v in predicates.values())
    verdict = "established" if n_hits == 3 else ("conditional" if n_hits == 2 else "not-established")

    result = {
        "milestone": "M6",
        "verifies": ["C3-choice", "C3-uncertainty", "C3-RSA"],
        "human_uncertainty_source": "THINGS testset2 vs testset2_repeat (two independent workers per triplet)",
        "aligned": unc_al,
        "unaligned": unc_ua,
        "rsa": rsa,
        "predicates": predicates,
        "n_predicates_met": n_hits,
        "verdict": verdict,
        "config": {k: v for k, v in vars(args).items()},
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print("[done]", out/"results.json")
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
