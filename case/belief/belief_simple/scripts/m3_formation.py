"""M3: Behavioral + causal-intervention measurements at one pythia-1b intermediate checkpoint.

Given a step id, load pythia-1b at that checkpoint (dir = `checkpoints/step{STEP}/`),
apply H*_personal / H*_attributed head sets (identified on step143000), and record 9 accuracies:
  3 behavioural (WK / PB / AB, no ablation)
  3 with H*_personal ablated (WK / PB / AB)
  3 with H*_attributed ablated (WK / PB / AB)

If a target's H* is unavailable (M2 not localized), the corresponding causal trajectory is
recorded as `not_applicable`.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from belief_utils import (
    load_model_and_tokenizer, load_task, evaluate_task_accuracy, load_json, save_json,
    install_head_scaling_hooks, remove_hooks, set_seed,
)


TASKS = ["world_knowledge", "personal_belief", "attributed_belief"]


def _eval_all(net, tok, examples_by_task, device):
    return {task: evaluate_task_accuracy(net, tok, examples_by_task[task], device=device)
            for task in TASKS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--model", default="pythia-1b")
    ap.add_argument("--checkpoint", required=True, help="e.g. step10000")
    ap.add_argument("--hstar-personal", required=True, help="path to H_personal.json")
    ap.add_argument("--hstar-attributed", required=True, help="path to H_attributed.json")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    set_seed(0)
    checkpoint_dir = os.path.join(args.model_root, f"{args.model}-checkpoints", args.checkpoint)
    net, tok = load_model_and_tokenizer(args.model_root, args.model, dtype=args.dtype,
                                        device=args.device, checkpoint_dir=checkpoint_dir)
    for p in net.parameters():
        p.requires_grad_(False)

    examples_by_task = {t: load_task(args.data_root, t) for t in TASKS}

    result = {"model": args.model, "checkpoint": args.checkpoint, "checkpoint_dir": checkpoint_dir}

    # Behavioural (no ablation)
    t0 = time.time()
    print(f"[m3] {args.checkpoint} behavioural...")
    behav = _eval_all(net, tok, examples_by_task, args.device)
    result["behavioural"] = {t: {"acc": behav[t]["acc"], "wilson_ci_low": behav[t]["wilson_ci_low"],
                                  "wilson_ci_high": behav[t]["wilson_ci_high"],
                                  "correct_count": behav[t]["correct_count"], "total": behav[t]["total"]}
                              for t in TASKS}
    print(f"[m3]   WK={behav['world_knowledge']['acc']:.3f} PB={behav['personal_belief']['acc']:.3f} "
          f"AB={behav['attributed_belief']['acc']:.3f} ({time.time()-t0:.1f}s)")

    # Load H* sets
    def _load_hstar(path):
        try:
            h = load_json(path)
            if h.get("status") != "localized":
                return None
            return [(int(l), int(hh)) for l, hh in h["hstar_heads"]]
        except FileNotFoundError:
            return None

    Hp = _load_hstar(args.hstar_personal)
    Ha = _load_hstar(args.hstar_attributed)

    for name, heads in (("hstar_personal_ablated", Hp), ("hstar_attributed_ablated", Ha)):
        if heads is None:
            result[name] = {t: "not_applicable" for t in TASKS}
            continue
        t0 = time.time()
        scale = {(l, h): 0.0 for (l, h) in heads}
        handles = install_head_scaling_hooks(net, scale)
        try:
            abl = _eval_all(net, tok, examples_by_task, args.device)
        finally:
            remove_hooks(handles)
        result[name] = {t: {"acc": abl[t]["acc"], "wilson_ci_low": abl[t]["wilson_ci_low"],
                             "wilson_ci_high": abl[t]["wilson_ci_high"],
                             "correct_count": abl[t]["correct_count"], "total": abl[t]["total"]}
                         for t in TASKS}
        result[f"{name}_head_count"] = len(heads)
        print(f"[m3]   {name}: WK={abl['world_knowledge']['acc']:.3f} PB={abl['personal_belief']['acc']:.3f} "
              f"AB={abl['attributed_belief']['acc']:.3f} ({time.time()-t0:.1f}s)")

    save_json(args.output, result)
    print(f"[m3] {args.checkpoint} → {args.output}")


if __name__ == "__main__":
    main()
