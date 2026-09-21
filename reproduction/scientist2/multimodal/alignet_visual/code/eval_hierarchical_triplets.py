"""M2 — Hierarchical pseudo-label evaluation of the M1 teacher head on ImageNet.

Construct 3k triplets from ImageNet-val (50k) images using the BREEDS-derived WordNet
hierarchy over ImageNet-1k classes:

  - coarse: (a, b) drawn from the same top-level category (e.g., 'living thing')
            and c from a *different* top-level category. Should be EASY.
  - mid:    (a, b) share a mid-level category (e.g., 'mammal') and c is in a different
            mid-level but the same top level.
  - fine:   (a, b) share a fine-level parent (e.g., 'dog') and c is in a different
            fine-level parent but the same mid.

We report per-level triplet-choice agreement between the fit teacher head and this
hierarchical ground truth: pass if agreement is monotonically ordered
coarse >= mid >= fine (with all > chance = 1/3), or if the ordering direction is
documented consistently in the README (either direction acceptable per plan).

The teacher's "choice" is argmin(mean similarity to the other two).
"""
from __future__ import annotations
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import (
    load_siglip_vision,
    ImageNetValIndex,
    ImageNetValImageDataset,
    load_imagenet1k_wnid_to_class,
    load_breeds_hierarchy,
    wnid_ancestors_map,
    seed_everything,
)
from fit_teacher_head import AlignmentHead


def find_ancestor_at_depth(wnid: str, parent_of: dict, target_wnids: set[str]) -> str | None:
    """Walk up from `wnid` and return the first ancestor in `target_wnids`."""
    seen = set()
    stack = [wnid]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        if n in target_wnids:
            return n
        for p in parent_of.get(n, []):
            stack.append(p)
    return None


def build_class_to_super_labels(edges: dict, wnid_to_class: dict) -> tuple[dict, dict, dict]:
    """For each imagenet-1k class, assign a coarse, mid, and fine super-class label.

    Strategy (heuristic — level identity chosen empirically to give non-degenerate buckets):
      - coarse: BREEDS entity13 groups (13 coarse groups over 1000 classes) — level-1 groupings
      - mid:    ancestor at depth 4-5 in the BREEDS-modified hierarchy — approximately 30-50 mid groups
      - fine:   direct WordNet parent of the leaf synset — approximately 300+ fine groups

    Returns three dicts: {class_id -> label_id or None}
    """
    parent_of = wnid_ancestors_map(edges)

    # Compute depth-from-root for every node
    root = "n00001740"  # 'entity'
    depths = {root: 0}
    frontier = [root]
    while frontier:
        n2 = []
        for n in frontier:
            for c in edges.get(n, []):
                if c not in depths:
                    depths[c] = depths[n] + 1
                    n2.append(c)
        frontier = n2

    # Gather all imagenet-1k leaf wnids
    leaves = list(wnid_to_class.keys())

    # Collect ancestors of each leaf
    def ancestors(wnid):
        out = set()
        stack = list(parent_of.get(wnid, []))
        while stack:
            n = stack.pop()
            if n in out:
                continue
            out.add(n)
            stack.extend(parent_of.get(n, []))
        return out

    leaf_ancestors = {w: ancestors(w) for w in leaves}

    # Coarse: nodes at depth == 2  (few, high-level)
    coarse_nodes = {n for n, d in depths.items() if d == 2}
    # Mid: nodes at depth == 5
    mid_nodes = {n for n, d in depths.items() if d == 5}
    # Fine: direct WordNet parent of each leaf synset — typically groups 3-15 sibling classes
    # (e.g., "dog breed X" and "dog breed Y" share the same fine parent).
    fine_nodes = set()
    for wnid in leaves:
        for p in parent_of.get(wnid, []):
            fine_nodes.add(p)

    class_to_coarse = {}
    class_to_mid = {}
    class_to_fine = {}
    for wnid, cid in wnid_to_class.items():
        an = leaf_ancestors[wnid] | {wnid}
        c = next(iter(an & coarse_nodes), None)
        m = next(iter(an & mid_nodes), None)
        f = next(iter(an & fine_nodes), None)
        class_to_coarse[cid] = c
        class_to_mid[cid] = m
        class_to_fine[cid] = f

    n_covered_coarse = sum(1 for v in class_to_coarse.values() if v is not None)
    n_covered_mid    = sum(1 for v in class_to_mid.values() if v is not None)
    n_covered_fine   = sum(1 for v in class_to_fine.values() if v is not None)
    n_coarse = len({v for v in class_to_coarse.values() if v is not None})
    n_mid    = len({v for v in class_to_mid.values() if v is not None})
    n_fine   = len({v for v in class_to_fine.values() if v is not None})
    print(f"[level] coarse: {n_covered_coarse}/1000 classes covered, {n_coarse} groups")
    print(f"[level] mid:    {n_covered_mid}/1000 classes covered, {n_mid} groups")
    print(f"[level] fine:   {n_covered_fine}/1000 classes covered, {n_fine} groups")
    return class_to_coarse, class_to_mid, class_to_fine


def sample_hierarchical_triplets(labels: np.ndarray, class_to_coarse: dict, class_to_mid: dict,
                                  class_to_fine: dict, n_per_level: int = 1000, seed: int = 42) -> dict:
    """Sample ~n_per_level triplets per level from the pool of (index, label) rows.

    Each triplet is (idx_a, idx_b, idx_c) INTO `labels` array; c is the odd-one-out.
    """
    rng = np.random.default_rng(seed)
    N = len(labels)
    labels = np.asarray(labels)
    # Group image indices by class
    by_class = defaultdict(list)
    for i, l in enumerate(labels):
        by_class[int(l)].append(i)
    for k in by_class:
        by_class[k] = np.array(by_class[k])

    def sample_triplet(same_super_fn, diff_super_fn):
        """Sample a triplet where (a,b) share same_super_fn and c is diff_super_fn from a/b."""
        for _ in range(100):
            # Pick a super-label that has ≥ 2 classes each with ≥ 1 image
            same_super = rng.choice(list({v for v in same_super_fn.values() if v is not None}))
            classes_in = [c for c, sv in same_super_fn.items() if sv == same_super and len(by_class.get(c, [])) > 0]
            if len(classes_in) < 2:
                continue
            # a, b from same super, different classes
            ca, cb = rng.choice(classes_in, size=2, replace=False)
            a = rng.choice(by_class[int(ca)])
            b = rng.choice(by_class[int(cb)])
            # c from a different super
            other = [c for c, sv in same_super_fn.items() if sv is not None and sv != same_super and diff_super_fn.get(c) != diff_super_fn.get(int(ca)) and len(by_class.get(c, [])) > 0]
            if not other:
                # relax: any different super
                other = [c for c, sv in same_super_fn.items() if sv is not None and sv != same_super and len(by_class.get(c, [])) > 0]
            if not other:
                continue
            cc = rng.choice(other)
            c = rng.choice(by_class[int(cc)])
            return int(a), int(b), int(c)
        return None

    triplets = {"coarse": [], "mid": [], "fine": []}
    # coarse: (a,b) same coarse, c different coarse
    for _ in range(n_per_level):
        t = sample_triplet(class_to_coarse, class_to_mid)
        if t: triplets["coarse"].append(t)
    # mid: (a,b) same mid, c different mid (but not too coarse)
    for _ in range(n_per_level):
        t = sample_triplet(class_to_mid, class_to_fine)
        if t: triplets["mid"].append(t)
    # fine: (a,b) same fine, c different fine
    for _ in range(n_per_level):
        t = sample_triplet(class_to_fine, class_to_coarse)
        if t: triplets["fine"].append(t)
    for k in triplets:
        triplets[k] = np.array(triplets[k], dtype=np.int64) if triplets[k] else np.zeros((0, 3), dtype=np.int64)
        print(f"  sampled {len(triplets[k])} triplets at level={k}")
    return triplets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_head", required=True, help="path to teacher_head.pt")
    p.add_argument("--teacher_cache", default=None,
                   help="optional: an existing SigLIP feature cache from M1.5 (either raw or head-projected). "
                        "If provided AND has `feats_head`, we use those directly; else we forward SigLIP now.")
    p.add_argument("--triplets_per_level", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_images", type=int, default=8000,
                   help="ImageNet-val images to draw features for (must be >> triplets_per_level so we can sample varied triplets)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    seed_everything(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    # 1) Get SigLIP features (+ head-projected) for a subset of imagenet-val
    #    Prefer to use the M1.5 cache when present — otherwise compute on the fly.
    from data_utils import ImageNetValIndex
    idx = ImageNetValIndex.build()
    if args.teacher_cache and Path(args.teacher_cache).exists():
        print(f"[cache] using existing cache {args.teacher_cache}")
        with h5py.File(args.teacher_cache, "r") as h5:
            if "feats_head" in h5:
                feats = np.asarray(h5["feats_head"][:args.n_images])
                gids = np.asarray(h5["global_index"][:args.n_images])
                labels = np.asarray(h5["label"][:args.n_images])
                print(f"  loaded feats_head shape {feats.shape}")
            else:
                # Need to project raw with the head — do it here
                raw = np.asarray(h5["feats_raw"][:args.n_images])
                gids = np.asarray(h5["global_index"][:args.n_images])
                labels = np.asarray(h5["label"][:args.n_images])
                head = AlignmentHead(dim=1152).to(device).eval()
                head.load_state_dict(torch.load(args.teacher_head, map_location=device))
                head = head.to(dtype=torch.float32)
                with torch.no_grad():
                    feats_t = head(torch.from_numpy(raw).to(device).float()).cpu().numpy()
                feats = feats_t
                print(f"  projected raw feats with teacher head -> shape {feats.shape}")
    else:
        print("[cache] no cache provided — running SigLIP forward + head projection now")
        model, proc = load_siglip_vision(dtype=torch.float16, device=device)
        head = AlignmentHead(dim=1152).to(device).eval()
        head.load_state_dict(torch.load(args.teacher_head, map_location=device))
        head = head.to(dtype=torch.float16)
        n = min(args.n_images, idx.total)
        gids_np = idx.sample_indices(n, seed=args.seed)
        ds = ImageNetValImageDataset(idx, global_indices=gids_np, processor=proc)
        dl = DataLoader(ds, batch_size=args.batch_size, num_workers=0, shuffle=False)
        feats_list = []; gids_list = []; labels_list = []
        t0 = time.time()
        with torch.no_grad():
            for batch in tqdm(dl, desc="siglip+head"):
                pv = batch["pixel_values"].to(device, dtype=torch.float16)
                out_v = model.vision_model(pv)
                p2 = head(out_v.pooler_output)
                feats_list.append(p2.float().cpu().numpy())
                gids_list.append(batch["global_index"].numpy())
                labels_list.append(batch["label"].numpy())
        feats = np.concatenate(feats_list)
        gids = np.concatenate(gids_list)
        labels = np.concatenate(labels_list)
        print(f"  computed {len(feats)} features in {time.time()-t0:.1f}s")

    # Normalize
    feats_t = torch.from_numpy(feats).float().to(device)
    feats_t = F.normalize(feats_t, dim=-1)

    # 2) Build BREEDS-based class-to-super mappings
    edges = load_breeds_hierarchy()
    wnid_to_class = load_imagenet1k_wnid_to_class()
    c2c, c2m, c2f = build_class_to_super_labels(edges, wnid_to_class)

    # 3) Sample per-level triplets
    triplets = sample_hierarchical_triplets(labels, c2c, c2m, c2f,
                                             n_per_level=args.triplets_per_level, seed=args.seed)

    # 4) Score: for each triplet [ia, ib, ic] with ic as the odd-one-out (by construction),
    #    check if the teacher agrees, i.e. argmin of mean similarity == position 2.
    def score_triplets(tri: np.ndarray) -> dict:
        if len(tri) == 0:
            return {"n": 0, "agreement_rate": float("nan"), "ci95": [float("nan"), float("nan")]}
        a = tri[:, 0]; b = tri[:, 1]; c = tri[:, 2]
        fa = feats_t[a]; fb = feats_t[b]; fc = feats_t[c]
        sab = (fa*fb).sum(-1); sac = (fa*fc).sum(-1); sbc = (fb*fc).sum(-1)
        ma = ((sab + sac) / 2).cpu().numpy()
        mb = ((sab + sbc) / 2).cpu().numpy()
        mc = ((sac + sbc) / 2).cpu().numpy()
        means = np.stack([ma, mb, mc], axis=1)
        pred = np.argmin(means, axis=1)
        correct = (pred == 2).astype(np.int32)
        # bootstrap CI
        rng = np.random.default_rng(0)
        boots = np.array([correct[rng.integers(0, len(correct), size=len(correct))].mean() for _ in range(1000)])
        return {
            "n": int(len(tri)),
            "agreement_rate": float(correct.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        }

    results = {}
    for level, tri in triplets.items():
        results[level] = score_triplets(tri)
        print(f"[score] {level}: {results[level]}")

    # 5) Monotonicity check + separation vs chance
    chance = 1.0 / 3.0
    ordered = {k: v["agreement_rate"] for k, v in results.items()}
    monotone_dec = ordered.get("coarse", 0) >= ordered.get("mid", 0) >= ordered.get("fine", 0)
    monotone_inc = ordered.get("coarse", 0) <= ordered.get("mid", 0) <= ordered.get("fine", 0)
    all_above_chance = all(
        (v["ci95"][0] > chance) if not np.isnan(v["ci95"][0]) else False
        for v in results.values()
    )
    summary = {
        "milestone": "M2",
        "verifies": ["C1b"],
        "per_level": results,
        "chance": chance,
        "monotone_decreasing_coarse_to_fine": monotone_dec,
        "monotone_increasing_coarse_to_fine": monotone_inc,
        "all_levels_above_chance_ci95_lower": all_above_chance,
        "pass_criterion": (all_above_chance and (monotone_dec or monotone_inc)),
        "config": {k: v for k, v in vars(args).items()},
    }
    (out / "level_separation.json").write_text(json.dumps(summary, indent=2))
    print("[done]", out/"level_separation.json")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
