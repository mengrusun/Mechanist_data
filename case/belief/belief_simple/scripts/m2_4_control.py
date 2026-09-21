"""M2.4: 20 random-head + 20 random-mask controls per successful H*.

For a single control run: given `--kind` ∈ {random_head, random_mask} and `--seed`,
draw one random subset and evaluate the same 4 metrics as M2.3.

random_head — subset of |H*| attention heads drawn uniformly without replacement from the
    full attention-head set. Ablation via forward hook (same mechanism as M2.3).

random_mask — subset of parameters drawn uniformly without replacement across the
    {qkv.weight, qkv.bias, dense.weight} parameters in every layer, of the SAME total count
    as the |H*| union parameter count. Ablation via in-place parameter zeroing +
    restoration after the eval.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from belief_utils import (
    load_model_and_tokenizer, load_task, load_json, save_json, evaluate_task_accuracy,
    evaluate_ppl, install_head_scaling_hooks, remove_hooks,
    sample_random_head_subset, sample_random_param_mask,
    apply_param_zero_mask, restore_params, set_seed,
)


BELIEF_TASK = {"personal": "personal_belief", "attributed": "attributed_belief"}
OTHER_TASK = {"personal": "attributed_belief", "attributed": "personal_belief"}


def _evaluate_all(net, tok, examples_by_task, ppl_tokens, device: str) -> dict:
    out = {}
    for task, examples in examples_by_task.items():
        r = evaluate_task_accuracy(net, tok, examples, device=device)
        out[task] = {"acc": r["acc"], "correct_count": r["correct_count"], "total": r["total"],
                     "wilson_ci_low": r["wilson_ci_low"], "wilson_ci_high": r["wilson_ci_high"]}
    ppl = evaluate_ppl(net, ppl_tokens, window=1024, device=device)
    out["ppl"] = ppl
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", required=True, choices=["personal", "attributed"])
    ap.add_argument("--hstar", required=True, help="path to H_{target}.json produced by M2.3")
    ap.add_argument("--kind", required=True, choices=["random_head", "random_mask"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--ppl-sample", required=True)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    net, tok = load_model_and_tokenizer(args.model_root, args.model, dtype=args.dtype, device=args.device)
    for p in net.parameters():
        p.requires_grad_(False)

    hstar = load_json(args.hstar)
    heads = [tuple(h) for h in hstar["hstar_heads"]]
    if not heads or hstar.get("status") != "localized":
        raise RuntimeError(f"H* not localized: {hstar.get('status')}")
    print(f"[m2.4] {args.model} × {args.target} × {args.kind} seed={args.seed} |H*|={len(heads)}")

    target_task = BELIEF_TASK[args.target]
    other_task = OTHER_TASK[args.target]
    examples_by_task = {
        target_task: load_task(args.data_root, target_task),
        other_task: load_task(args.data_root, other_task),
        "world_knowledge": load_task(args.data_root, "world_knowledge"),
    }
    ppl_tokens = torch.load(args.ppl_sample)
    if isinstance(ppl_tokens, dict):
        ppl_tokens = ppl_tokens["tokens"]

    t0 = time.time()
    result = {"model": args.model, "target": args.target, "kind": args.kind, "seed": args.seed,
              "hstar_size": len(heads)}

    if args.kind == "random_head":
        rand_heads = sample_random_head_subset(net, len(heads), args.seed)
        result["control_heads"] = [list(h) for h in rand_heads]
        scale = {(l, h): 0.0 for (l, h) in rand_heads}
        handles = install_head_scaling_hooks(net, scale)
        try:
            ablated = _evaluate_all(net, tok, examples_by_task, ppl_tokens, device=args.device)
        finally:
            remove_hooks(handles)
    else:  # random_mask
        mask = sample_random_param_mask(net, heads, args.seed)
        n_zeroed = sum(int(v.numel()) for v in mask.values())
        result["n_params_zeroed"] = n_zeroed
        snap = apply_param_zero_mask(net, mask)
        try:
            ablated = _evaluate_all(net, tok, examples_by_task, ppl_tokens, device=args.device)
        finally:
            restore_params(net, snap)

    result["ablated"] = ablated
    result["elapsed_sec"] = time.time() - t0
    save_json(args.output, result)
    print(f"[m2.4] done seed={args.seed} target_drop_vs_ablated_acc={ablated[target_task]['acc']:.4f} "
          f"({time.time()-t0:.1f}s) → {args.output}")


if __name__ == "__main__":
    main()
