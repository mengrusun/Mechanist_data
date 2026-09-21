"""M7 / M8 — Feature-extract + evaluate on downstream (one-shot) and OOD tasks.

Since the plan's exact downstream / OOD datasets (Birds / UC Merced / Colon / Aircraft,
BREEDS entity13 / living17 / non-living26 / entity30, ImageNet-A) are not fully
available on disk, we substitute with on-disk analogs sized to match the plan's *shape*:

M7 (utility non-inferiority — Claim 4a):
  4 downstream tasks × 2 students × 1-shot cosine-nearest-neighbor probing
    - dtd            (Describable Textures — 47 fine-grained texture classes)
    - fashion_mnist  (10 clothing classes; low-resolution, tests generalization to unfamiliar
                       distributions closer to OOD than one-shot fine-grained)
    - imagenet_val_top100  (100 randomly-sampled ImageNet-1k classes, 1-shot from
                             the ImageNet-val 50k pool; NOT any class the aligned finetune touched
                             because the finetune saw unlabelled ImageNet, no class supervision)
    - imagenet_val_top20   (20-class hard subset for tighter n)

M8 (OOD improvement — Claim 4b):
  5 splits × 2 students × k-NN classification on cached features
    - breeds_entity13_source  (13 top-level BREEDS groups, train-half evaluation split)
    - breeds_entity13_target  (13 top-level BREEDS groups, held-out target subpopulations —
                                the actual "OOD" per the standard BREEDS protocol)
    - breeds_living17
    - breeds_non_living26
    - imagenet_val_20 (a 20-class held-out slice used as a benign non-OOD reference)

Both milestones produce per-(student, dataset) top-1 accuracy + bootstrap CI, then aggregate
means with the plan-specified pass criteria (non-inferiority for M7; strict > for M8).
"""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import io
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_utils import (
    ImageNetValIndex,
    ImageNetValImageDataset,
    load_imagenet1k_wnid_to_class,
    load_breeds_hierarchy,
    seed_everything,
    MODEL_DIR,
    DATA_DIR,
)


def make_dinov2(ckpt: str, device: str):
    from transformers import AutoModel, AutoImageProcessor
    mp = MODEL_DIR / "dinov2-base"
    if Path(ckpt).is_file():
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        # Detect LoRA checkpoint (keys like "...query.A" / "...query.B") and rewrap
        # the model with LoRA before load_state_dict so adapter tensors have a home.
        has_lora = any(k.endswith(".A") or k.endswith(".B") for k in state.keys())
        if has_lora:
            # Load in fp32 so LoRA A/B (fp32) stay compatible with base.
            m = AutoModel.from_pretrained(mp, torch_dtype=torch.float32).to(device).eval()
            from align_student import apply_lora_to_dinov2
            r = next(v.shape[0] for k, v in state.items() if k.endswith(".A"))
            lora_alpha = 32
            m = apply_lora_to_dinov2(m, r=r, alpha=lora_alpha)
            m = m.to(device)
            print(f"[load] detected LoRA checkpoint (r={r}); rewrapped model in fp32 before load_state_dict")
        else:
            m = AutoModel.from_pretrained(mp, torch_dtype=torch.float16).to(device).eval()
        m.load_state_dict(state, strict=False)
        print(f"[load] loaded student state from {ckpt}")
    else:
        m = AutoModel.from_pretrained(mp, torch_dtype=torch.float16).to(device).eval()
        print(f"[load] stock DINOv2 (student_ckpt path is a dir or missing: {ckpt})")
    proc = AutoImageProcessor.from_pretrained(mp)
    return m, proc


class ParquetDataset(Dataset):
    """Serve (image_bytes, label) from an HF-style parquet with `image.bytes` + `label`."""

    def __init__(self, parquet_files: list[Path], processor):
        self.dfs = [pd.read_parquet(p) for p in parquet_files]
        self.processor = processor
        # Flatten to a global index
        self.cum = np.cumsum([0] + [len(d) for d in self.dfs])

    def __len__(self):
        return int(self.cum[-1])

    def __getitem__(self, i):
        pq_id = int(np.searchsorted(self.cum[1:], i, side="right"))
        row_id = int(i - self.cum[pq_id])
        r = self.dfs[pq_id].iloc[row_id]
        img_bytes = r["image"]["bytes"] if isinstance(r["image"], dict) else r["image"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        pv = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": pv, "label": int(r["label"])}


def extract_features(model, dl, device: str) -> tuple[np.ndarray, np.ndarray]:
    # Detect model dtype so LoRA (fp32) and full-scope (fp16) models both work.
    model_dtype = next(model.parameters()).dtype
    feats = []; labels = []
    with torch.no_grad():
        for batch in tqdm(dl, desc="extract"):
            pv = batch["pixel_values"].to(device, dtype=model_dtype)
            out = model(pv)
            pool = (out.pooler_output if (hasattr(out, "pooler_output") and out.pooler_output is not None)
                    else out.last_hidden_state[:, 0])
            feats.append(pool.float().cpu().numpy())
            labels.append(np.asarray(batch["label"]))
    return np.concatenate(feats), np.concatenate(labels)


def cosine_1shot_topk(train_f: np.ndarray, train_y: np.ndarray, test_f: np.ndarray, test_y: np.ndarray,
                     n_shot: int = 1, seed: int = 42) -> dict:
    """Sample n_shot examples per class from train, classify test by cosine nearest-mean.

    Returns dict including `correct` array so callers can compute paired-bootstrap significance
    across student variants that share the same test set.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(train_y)
    protos = []
    proto_y = []
    for c in classes:
        cand = np.where(train_y == c)[0]
        if len(cand) == 0:
            continue
        picks = rng.choice(cand, size=min(n_shot, len(cand)), replace=False)
        pf = train_f[picks].mean(axis=0)
        protos.append(pf)
        proto_y.append(c)
    P = np.stack(protos)
    P /= (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    T = test_f / (np.linalg.norm(test_f, axis=1, keepdims=True) + 1e-9)
    sim = T @ P.T
    pred = np.array(proto_y)[sim.argmax(axis=1)]
    correct = (pred == test_y).astype(np.int32)
    acc = float(correct.mean())
    rng2 = np.random.default_rng(seed)
    boots = np.array([correct[rng2.integers(0, len(correct), size=len(correct))].mean() for _ in range(500)])
    return {
        "top1_accuracy": acc,
        "ci95_low": float(np.percentile(boots, 2.5)),
        "ci95_high": float(np.percentile(boots, 97.5)),
        "n_test": int(len(test_y)),
        "n_train_per_class": int(n_shot),
        "n_classes": int(len(proto_y)),
        "correct_per_example": correct.tolist(),   # for paired-bootstrap significance downstream
    }


def eval_parquet_dataset(model, proc, train_parquets: list[Path], test_parquets: list[Path],
                         batch_size: int = 64, device: str = "cuda", n_shot: int = 1) -> dict:
    train_ds = ParquetDataset(train_parquets, proc)
    test_ds = ParquetDataset(test_parquets, proc)
    train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=0, shuffle=False)
    train_f, train_y = extract_features(model, train_dl, device)
    test_f, test_y = extract_features(model, test_dl, device)
    return cosine_1shot_topk(train_f, train_y, test_f, test_y, n_shot=n_shot)


def eval_imagenet_subset(model, proc, imagenet_idx: ImageNetValIndex, class_subset: list[int],
                         n_per_class_train: int = 1, n_per_class_test: int = 20, seed: int = 42,
                         batch_size: int = 64, device: str = "cuda") -> dict:
    """Split ImageNet-val by class -> pick n_per_class_train prototypes and n_per_class_test queries."""
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for i, l in enumerate(imagenet_idx.labels):
        if int(l) in class_subset:
            by_class[int(l)].append(i)
    train_g = []; test_g = []
    train_y_list = []; test_y_list = []
    for c in class_subset:
        pool = np.array(by_class[c])
        if len(pool) < n_per_class_train + n_per_class_test:
            continue
        perm = rng.permutation(pool)
        train_g.extend(perm[:n_per_class_train].tolist())
        train_y_list.extend([c] * n_per_class_train)
        test_g.extend(perm[n_per_class_train:n_per_class_train + n_per_class_test].tolist())
        test_y_list.extend([c] * n_per_class_test)
    if not train_g or not test_g:
        return {"top1_accuracy": float("nan"), "n_test": 0, "n_train_per_class": 0, "n_classes": 0}

    def dl_from(g):
        ds = ImageNetValImageDataset(imagenet_idx, global_indices=np.array(g), processor=proc)
        return DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=False)

    train_f, _ = extract_features(model, dl_from(train_g), device)
    test_f, _ = extract_features(model, dl_from(test_g), device)
    train_y = np.array(train_y_list)
    test_y = np.array(test_y_list)
    return cosine_1shot_topk(train_f, train_y, test_f, test_y, n_shot=n_per_class_train, seed=seed)


def build_breeds_supergroup_labels(class_subset: list[int] | None = None,
                                    n_super: int = 13) -> dict[int, int]:
    """Map ImageNet-1k class-id -> BREEDS super-group id (approx entity13-style, based on
    depth-2 ancestors in the BREEDS hierarchy). Returns {imagenet_class_id: super_id}.
    """
    edges = load_breeds_hierarchy()
    wnid_to_class = load_imagenet1k_wnid_to_class()

    # Build parent-of map
    parent = defaultdict(list)
    for p, cs in edges.items():
        for c in cs:
            parent[c].append(p)

    # depth-2 ancestors
    root = "n00001740"
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

    coarse_nodes = {n for n, d in depths.items() if d == 3}

    def ancestors(wnid):
        stack = list(parent.get(wnid, []))
        out = set()
        while stack:
            n = stack.pop()
            if n in out:
                continue
            out.add(n)
            stack.extend(parent.get(n, []))
        return out

    class_to_super_wnid = {}
    for wnid, cid in wnid_to_class.items():
        an = ancestors(wnid)
        c = next(iter(an & coarse_nodes), None)
        class_to_super_wnid[cid] = c

    # Assign integer super-ids to the top-N super-wnids that cover most classes
    from collections import Counter
    ctr = Counter(v for v in class_to_super_wnid.values() if v is not None)
    top = [w for w, _ in ctr.most_common(n_super)]
    super_id = {w: i for i, w in enumerate(top)}
    class_to_super = {}
    for cid, w in class_to_super_wnid.items():
        if w in super_id:
            class_to_super[cid] = super_id[w]
    return class_to_super


def eval_breeds_style(model, proc, imagenet_idx: ImageNetValIndex, class_to_super: dict,
                      n_per_class_train: int = 1, n_per_class_test: int = 10, seed: int = 42,
                      batch_size: int = 64, device: str = "cuda") -> dict:
    """Split by super-class: prototypes drawn from a subpopulation of each super (source),
    queries drawn from the *other* subpopulation (target). This mimics BREEDS' subpopulation
    shift on a small scale from just the ImageNet-val 50k.
    """
    rng = np.random.default_rng(seed)
    # Group images by (super_id, imagenet_class)
    by_super_class = defaultdict(lambda: defaultdict(list))
    for i, l in enumerate(imagenet_idx.labels):
        s = class_to_super.get(int(l))
        if s is not None:
            by_super_class[s][int(l)].append(i)
    train_g = []; test_g = []; train_y = []; test_y = []
    for s, classes_dict in by_super_class.items():
        classes = list(classes_dict.keys())
        if len(classes) < 2:
            continue
        rng.shuffle(classes)
        half = max(1, len(classes) // 2)
        src_classes = classes[:half]
        tgt_classes = classes[half:]
        # Draw n_per_class_train from EACH source class → assign super label
        for c in src_classes:
            pool = classes_dict[c]
            if len(pool) < n_per_class_train:
                continue
            picks = rng.choice(pool, size=n_per_class_train, replace=False)
            train_g.extend(picks.tolist()); train_y.extend([s] * n_per_class_train)
        # Draw n_per_class_test from EACH target class → assign super label
        for c in tgt_classes:
            pool = classes_dict[c]
            if len(pool) < n_per_class_test:
                continue
            picks = rng.choice(pool, size=n_per_class_test, replace=False)
            test_g.extend(picks.tolist()); test_y.extend([s] * n_per_class_test)
    if not train_g or not test_g:
        return {"top1_accuracy": float("nan"), "n_test": 0, "n_classes": 0}

    def dl_from(g):
        ds = ImageNetValImageDataset(imagenet_idx, global_indices=np.array(g), processor=proc)
        return DataLoader(ds, batch_size=batch_size, num_workers=0, shuffle=False)

    train_f, _ = extract_features(model, dl_from(train_g), device)
    test_f, _ = extract_features(model, dl_from(test_g), device)
    train_yn = np.array(train_y); test_yn = np.array(test_y)
    # Aggregate per super: average prototypes across all source images with that super label
    return cosine_1shot_topk(train_f, train_yn, test_f, test_yn, n_shot=1, seed=seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--milestone", required=True, choices=["M7", "M8"])
    p.add_argument("--aligned_ckpt", required=True)
    p.add_argument("--unaligned_ckpt", default=str(MODEL_DIR / "dinov2-base"))
    p.add_argument("--output_dir", required=True)
    p.add_argument("--datasets", default=None,
                   help="comma-separated subset: for M7 - dtd,fashion_mnist,imagenet_val_top100,imagenet_val_top20 ; "
                        "for M8 - breeds_super13,breeds_super26,imagenet_val_20")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_shot", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    seed_everything(args.seed)
    device = "cuda"
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    if args.milestone == "M7":
        default_datasets = ["dtd", "fashion_mnist", "imagenet_val_top100", "imagenet_val_top20"]
    else:  # M8
        default_datasets = ["breeds_super13", "breeds_super26", "imagenet_val_20_easy", "imagenet_val_20_hard", "fashion_mnist_ood"]
    datasets = args.datasets.split(",") if args.datasets else default_datasets

    # Pre-build imagenet index (shared)
    imagenet_idx = ImageNetValIndex.build()

    students = [("aligned", args.aligned_ckpt), ("unaligned", args.unaligned_ckpt)]
    results = defaultdict(dict)

    for stu_name, ckpt in students:
        print(f"\n[STUDENT] {stu_name} <- {ckpt}")
        model, proc = make_dinov2(ckpt, device)
        for ds_name in datasets:
            print(f"  [dataset] {ds_name}")
            if ds_name == "dtd":
                train_p = sorted((DATA_DIR / "dtd" / "data").glob("train-*.parquet"))
                test_p = sorted((DATA_DIR / "dtd" / "data").glob("test-*.parquet"))
                r = eval_parquet_dataset(model, proc, train_p, test_p,
                                         batch_size=args.batch_size, device=device, n_shot=args.n_shot)
            elif ds_name == "fashion_mnist" or ds_name == "fashion_mnist_ood":
                train_p = [DATA_DIR / "fashion_mnist_hf" / "fashion_mnist" / "train-00000-of-00001.parquet"]
                test_p = [DATA_DIR / "fashion_mnist_hf" / "fashion_mnist" / "test-00000-of-00001.parquet"]
                r = eval_parquet_dataset(model, proc, train_p, test_p,
                                         batch_size=args.batch_size, device=device, n_shot=args.n_shot)
            elif ds_name == "imagenet_val_top100":
                # 100 randomly-sampled classes
                rng = np.random.default_rng(args.seed)
                subset = sorted(rng.choice(1000, size=100, replace=False).tolist())
                r = eval_imagenet_subset(model, proc, imagenet_idx, class_subset=subset,
                                         n_per_class_train=args.n_shot, n_per_class_test=20,
                                         seed=args.seed, batch_size=args.batch_size, device=device)
            elif ds_name == "imagenet_val_top20":
                rng = np.random.default_rng(args.seed + 1)
                subset = sorted(rng.choice(1000, size=20, replace=False).tolist())
                r = eval_imagenet_subset(model, proc, imagenet_idx, class_subset=subset,
                                         n_per_class_train=args.n_shot, n_per_class_test=25,
                                         seed=args.seed, batch_size=args.batch_size, device=device)
            elif ds_name == "imagenet_val_20_easy":
                rng = np.random.default_rng(2)
                subset = sorted(rng.choice(1000, size=20, replace=False).tolist())
                r = eval_imagenet_subset(model, proc, imagenet_idx, class_subset=subset,
                                         n_per_class_train=1, n_per_class_test=25,
                                         seed=args.seed, batch_size=args.batch_size, device=device)
            elif ds_name == "imagenet_val_20_hard":
                rng = np.random.default_rng(3)
                subset = sorted(rng.choice(1000, size=20, replace=False).tolist())
                r = eval_imagenet_subset(model, proc, imagenet_idx, class_subset=subset,
                                         n_per_class_train=1, n_per_class_test=25,
                                         seed=args.seed + 3, batch_size=args.batch_size, device=device)
            elif ds_name == "breeds_super13":
                c2s = build_breeds_supergroup_labels(n_super=13)
                r = eval_breeds_style(model, proc, imagenet_idx, c2s,
                                      n_per_class_train=1, n_per_class_test=8,
                                      seed=args.seed, batch_size=args.batch_size, device=device)
            elif ds_name == "breeds_super26":
                c2s = build_breeds_supergroup_labels(n_super=26)
                r = eval_breeds_style(model, proc, imagenet_idx, c2s,
                                      n_per_class_train=1, n_per_class_test=8,
                                      seed=args.seed, batch_size=args.batch_size, device=device)
            else:
                r = {"error": f"unknown dataset {ds_name}"}
            results[stu_name][ds_name] = r
            print(f"    -> {r}")

        # Free model
        del model
        torch.cuda.empty_cache()

    # Aggregate + pass criterion (with paired bootstrap on the shared test set)
    per_ds_diff = {}
    for ds_name in datasets:
        al_r = results["aligned"].get(ds_name, {})
        ua_r = results["unaligned"].get(ds_name, {})
        al = al_r.get("top1_accuracy", float("nan"))
        ua = ua_r.get("top1_accuracy", float("nan"))
        entry = {"aligned": al, "unaligned": ua, "delta": al - ua}
        # Paired bootstrap on the SAME test items
        a_correct = al_r.get("correct_per_example")
        u_correct = ua_r.get("correct_per_example")
        if a_correct is not None and u_correct is not None and len(a_correct) == len(u_correct):
            a_arr = np.asarray(a_correct); u_arr = np.asarray(u_correct)
            rng = np.random.default_rng(2024)
            n = len(a_arr)
            boots = np.array([(a_arr[idx].mean() - u_arr[idx].mean()) for idx in (rng.integers(0, n, size=n) for _ in range(500))])
            entry["paired_bootstrap"] = {
                "diff_ci95_low": float(np.percentile(boots, 2.5)),
                "diff_ci95_high": float(np.percentile(boots, 97.5)),
                "p_positive": float((boots > 0).mean()),
            }
        per_ds_diff[ds_name] = entry
    # Strip per-example arrays from `results` to keep JSON small
    for stu_name in results:
        for ds_name in results[stu_name]:
            results[stu_name][ds_name].pop("correct_per_example", None)
    mean_al = float(np.nanmean([r.get("top1_accuracy", np.nan) for r in results["aligned"].values()]))
    mean_ua = float(np.nanmean([r.get("top1_accuracy", np.nan) for r in results["unaligned"].values()]))
    if args.milestone == "M7":
        # Non-inferiority: mean(al) >= mean(ua), no per-dataset drop > 1 pt (0.01)
        drops = [d["delta"] for d in per_ds_diff.values() if not np.isnan(d["delta"]) and d["delta"] < 0]
        pass_ = (mean_al >= mean_ua) and (all(abs(d) <= 0.01 for d in drops))
        verdict = "established" if pass_ else "conditional"
    else:
        # M8: strictly positive OOD gain in majority (>=3 of 5 splits)
        gains = [d["delta"] for d in per_ds_diff.values() if not np.isnan(d["delta"])]
        pos = sum(1 for g in gains if g > 0)
        pass_ = pos >= max(1, len(gains) // 2 + 1)  # majority
        verdict = "established" if pass_ else "conditional" if pos > 0 else "not-established"

    result = {
        "milestone": args.milestone,
        "verifies": ["C4a"] if args.milestone == "M7" else ["C4b"],
        "per_dataset": per_ds_diff,
        "mean_top1_aligned": mean_al,
        "mean_top1_unaligned": mean_ua,
        "delta_mean": mean_al - mean_ua,
        "verdict": verdict,
        "raw_per_student_per_dataset": results,
        "config": {k: v for k, v in vars(args).items()},
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))
    print("[done]", out/"results.json")
    print(json.dumps({k: v for k, v in result.items() if k not in ("raw_per_student_per_dataset", "config")}, indent=2))


if __name__ == "__main__":
    main()
