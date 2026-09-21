"""M2.b — Build target-specific Fisher masks and per-head candidate ranking.

Mask_target = top-0.1% F_target AND NOT top-1% F_knowledge (over the parameter
tensors kept in the M2a Fisher .pt files).  Aggregate mass per attention head =
count of masked parameters in that head's Q/K/V rows + dense columns.
"""
import argparse, json, os
import torch
from belief_lib import build_target_mask, aggregate_head_mass, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--f_target", required=True, help="path to M2a Fisher .pt for the target signal")
    ap.add_argument("--f_knowledge", required=True, help="path to M2a Fisher .pt for F_knowledge")
    ap.add_argument("--top_target_pct", type=float, default=0.001)
    ap.add_argument("--exclude_knowledge_pct", type=float, default=0.01)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[M2b] model={args.model} f_target={args.f_target}", flush=True)
    ft = torch.load(args.f_target, map_location="cpu", weights_only=False)["fisher_by_pname"]
    fk = torch.load(args.f_knowledge, map_location="cpu", weights_only=False)["fisher_by_pname"]

    masks = build_target_mask(ft, fk, args.top_target_pct, args.exclude_knowledge_pct)
    n_total = sum(m.numel() for m in masks.values())
    n_masked = sum(int(m.sum().item()) for m in masks.values())
    print(f"[M2b] mask footprint: {n_masked}/{n_total} params  "
          f"({100 * n_masked / n_total:.4f}%)", flush=True)

    head_mass = aggregate_head_mass(masks, args.model)
    ranking = sorted(head_mass.items(), key=lambda kv: -kv[1])
    ranking = [{"layer": int(L), "head": int(H), "per_head_mass": int(m)} for (L, H), m in ranking]

    save_json({
        "model": args.model,
        "f_target_path": args.f_target,
        "f_knowledge_path": args.f_knowledge,
        "top_target_pct": args.top_target_pct,
        "exclude_knowledge_pct": args.exclude_knowledge_pct,
        "mask_footprint_params": n_masked,
        "total_kept_params": n_total,
        "head_ranking": ranking,   # descending mass; nonzero-mass heads first
    }, args.out)
    print(f"[M2b] top-5 candidate heads: {ranking[:5]}", flush=True)


if __name__ == "__main__":
    main()
