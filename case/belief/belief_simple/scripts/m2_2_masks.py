"""M2.2: Mask construction and candidate-head ranking.

For a (model, target) cell:
  1. Read F_target and F_knowledge from M2.1 outputs.
  2. Materialize parameter-level AND-NOT mask:
        Mask_target = top-0.1% of F_target  AND  NOT top-1% of F_knowledge
     across ALL attention parameters (qkv.weight, qkv.bias, dense.weight, all layers).
  3. Compute per-head candidate score = fraction of the head's parameters that fall in the mask.
  4. Rank heads by candidate score descending; ties broken by aggregated Fisher magnitude descending.
  5. Optional jackknife stability: read the two half-Fisher tensors from M2.1 --jackknife;
     compute per-head candidate scores on each half separately (using the SAME F_knowledge
     top-1% for both halves); report Spearman ρ of the per-head rankings and set `stable`
     flag when ρ ≥ 0.6.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import numpy as np
from scipy.stats import spearmanr

from belief_utils import (
    load_model_and_tokenizer,
    model_arch_info,
    head_param_names,
    save_json,
)


def _param_family_layout(net, param_names):
    """Return (offsets, sizes, names_ordered) so we can flatten a set of tensors into one
    long vector and slice per parameter."""
    sizes = []
    for name in param_names:
        p = dict(net.named_parameters())[name]
        sizes.append(int(p.numel()))
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    return offsets, sizes, param_names


def _flatten_fisher(fisher_state: dict, param_names):
    """Concatenate the fisher values in param_names order into a single fp32 vector."""
    parts = []
    for name in param_names:
        parts.append(fisher_state[name].reshape(-1).float())
    return torch.cat(parts, dim=0)


def _topk_mask(flat_scores: torch.Tensor, frac: float) -> torch.Tensor:
    """Return a bool tensor of the same length: True on the top-frac positions."""
    k = max(1, int(round(frac * flat_scores.numel())))
    if k >= flat_scores.numel():
        return torch.ones_like(flat_scores, dtype=torch.bool)
    # top-k threshold
    thresh = torch.topk(flat_scores, k, largest=True).values[-1]
    return flat_scores >= thresh


def _head_candidate_scores(net, mask_flat: torch.Tensor, offsets, param_names, head_agg_fisher_flat: torch.Tensor):
    """Compute per-head candidate score = (# masked params in head) / (# total params in head).
    Also return the head's total aggregated Fisher for tie-breaking."""
    info = model_arch_info(net)
    n_layers, n_heads, d, hidden = info["n_layers"], info["n_heads"], info["head_dim"], info["hidden_size"]

    # Build reverse-map: for each layer, find which param_names entries are the
    # qkv.weight / qkv.bias / dense.weight for that layer, and their offsets.
    def _layer_slots(layer):
        names = head_param_names(layer)
        slot = {}
        for i, name in enumerate(param_names):
            for key in ("qkv_weight", "qkv_bias", "dense_weight"):
                if name == names[key]:
                    slot[key] = (i, offsets[i], offsets[i + 1])
        return slot

    per_head = {}
    for l in range(n_layers):
        slots = _layer_slots(l)
        qkv_w_off = slots["qkv_weight"][1:]
        qkv_b_off = slots["qkv_bias"][1:]
        dense_w_off = slots["dense_weight"][1:]
        qkv_w = mask_flat[qkv_w_off[0]: qkv_w_off[1]].view(3 * hidden, hidden)
        qkv_b = mask_flat[qkv_b_off[0]: qkv_b_off[1]]  # [3*hidden]
        dense_w = mask_flat[dense_w_off[0]: dense_w_off[1]].view(hidden, hidden)

        fisher_qkv_w = head_agg_fisher_flat[qkv_w_off[0]: qkv_w_off[1]].view(3 * hidden, hidden)
        fisher_qkv_b = head_agg_fisher_flat[qkv_b_off[0]: qkv_b_off[1]]
        fisher_dense_w = head_agg_fisher_flat[dense_w_off[0]: dense_w_off[1]].view(hidden, hidden)

        for h in range(n_heads):
            q_start, v_end = h * 3 * d, h * 3 * d + 3 * d
            n_masked = (qkv_w[q_start:v_end, :].sum() + qkv_b[q_start:v_end].sum()
                        + dense_w[:, h * d : (h + 1) * d].sum()).item()
            n_total = (3 * d) * hidden + (3 * d) + hidden * d
            f_tot = (fisher_qkv_w[q_start:v_end, :].sum().item()
                     + fisher_qkv_b[q_start:v_end].sum().item()
                     + fisher_dense_w[:, h * d : (h + 1) * d].sum().item())
            per_head[(l, h)] = {"score": n_masked / n_total, "n_masked": int(n_masked),
                                "n_total": n_total, "aggregated_fisher": f_tot}
    return per_head


def _build_mask_and_ranking(net, F_target: dict, F_knowledge: dict, top_target: float, top_knowledge: float):
    info = model_arch_info(net)
    # enumerate attention parameter names in a fixed order
    param_names = []
    for l in range(info["n_layers"]):
        names = head_param_names(l)
        param_names.extend([names["qkv_weight"], names["qkv_bias"], names["dense_weight"]])
    offsets, sizes, _ = _param_family_layout(net, param_names)
    F_t_flat = _flatten_fisher(F_target, param_names)
    F_k_flat = _flatten_fisher(F_knowledge, param_names)
    print(f"[m2.2] total attention params: {F_t_flat.numel():,}")

    mask_target = _topk_mask(F_t_flat, top_target)
    mask_knowledge = _topk_mask(F_k_flat, top_knowledge)
    mask_and_not = mask_target & (~mask_knowledge)
    print(f"[m2.2] top-{top_target:.4f} of F_target: {mask_target.sum().item():,} params")
    print(f"[m2.2] top-{top_knowledge:.4f} of F_knowledge: {mask_knowledge.sum().item():,} params")
    print(f"[m2.2] AND-NOT: {mask_and_not.sum().item():,} params")

    per_head = _head_candidate_scores(net, mask_and_not.float(), offsets, param_names, F_t_flat)
    # rank: score desc, then aggregated_fisher desc
    ranked = sorted(per_head.items(),
                    key=lambda kv: (-kv[1]["score"], -kv[1]["aggregated_fisher"]))
    return mask_and_not, per_head, ranked, F_t_flat


def _jackknife_stability(net, F_target_a: dict, F_target_b: dict, F_knowledge: dict,
                         top_target: float, top_knowledge: float):
    _, per_head_a, ranked_a, _ = _build_mask_and_ranking(net, F_target_a, F_knowledge, top_target, top_knowledge)
    _, per_head_b, ranked_b, _ = _build_mask_and_ranking(net, F_target_b, F_knowledge, top_target, top_knowledge)
    heads = sorted(per_head_a.keys())
    a_scores = np.array([per_head_a[h]["score"] for h in heads])
    b_scores = np.array([per_head_b[h]["score"] for h in heads])
    rho, p = spearmanr(a_scores, b_scores)
    return {"spearman_rho": float(rho), "p_value": float(p), "stable": bool(rho >= 0.6),
            "heads_evaluated": len(heads)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", required=True, choices=["personal", "attributed"])
    ap.add_argument("--fisher-target", required=True)
    ap.add_argument("--fisher-knowledge", required=True)
    ap.add_argument("--top-target", type=float, default=0.001)
    ap.add_argument("--top-knowledge", type=float, default=0.01)
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--output-ranking", required=True)
    ap.add_argument("--jackknife-half-a", default=None)
    ap.add_argument("--jackknife-half-b", default=None)
    ap.add_argument("--jackknife-report", default=None)
    ap.add_argument("--device", default="cpu", help="mask math can run on CPU cheaply")
    args = ap.parse_args()

    print(f"[m2.2] {args.model} × {args.target}")
    net, _ = load_model_and_tokenizer(args.model_root, args.model, dtype="fp16", device=args.device)
    for p in net.parameters():
        p.requires_grad_(False)

    F_target = torch.load(args.fisher_target, map_location="cpu")
    F_knowledge = torch.load(args.fisher_knowledge, map_location="cpu")
    mask, per_head, ranked, F_t_flat = _build_mask_and_ranking(
        net, F_target, F_knowledge, args.top_target, args.top_knowledge
    )

    ranked_list = []
    for (l, h), info in ranked:
        ranked_list.append({"layer": int(l), "head": int(h), **{k: float(v) if isinstance(v, (int, float)) else v
                                                                 for k, v in info.items()}})
    mask_summary = {
        "model": args.model, "target": args.target,
        "top_target": args.top_target, "top_knowledge": args.top_knowledge,
        "n_params_total": int(F_t_flat.numel()),
        "n_params_top_target": int(_topk_mask(F_t_flat, args.top_target).sum().item()),
        "n_params_masked_and_not": int(mask.sum().item()),
        "n_heads_with_any_masked_param": sum(1 for _, info in per_head.items() if info["n_masked"] > 0),
    }
    save_json(args.output, mask_summary)
    save_json(args.output_ranking, {"model": args.model, "target": args.target, "ranked": ranked_list})
    print(f"[m2.2] wrote {args.output}, {args.output_ranking}")

    if args.jackknife_half_a and args.jackknife_half_b and args.jackknife_report:
        F_a = torch.load(args.jackknife_half_a, map_location="cpu")
        F_b = torch.load(args.jackknife_half_b, map_location="cpu")
        report = _jackknife_stability(net, F_a, F_b, F_knowledge, args.top_target, args.top_knowledge)
        report["model"] = args.model
        report["target"] = args.target
        save_json(args.jackknife_report, report)
        print(f"[m2.2] jackknife ρ={report['spearman_rho']:.3f}, stable={report['stable']} → {args.jackknife_report}")


if __name__ == "__main__":
    main()
