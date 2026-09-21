"""M2.3: Smallest-head-set search via zero-ablation (greedy-add + greedy-remove).

For a (model, target) cell that cleared M1's above-chance gate:
  Read ranked candidate heads (M2.2 output).
  Compute the "clean" baselines for the target task, other belief task, world_knowledge,
    and PPL (each measured once with hooks-off).
  Greedy-add: add ranked heads one at a time; after each addition, evaluate the 4 metrics
    (target-acc drop, other-belief drop, world_knowledge drop, PPL ratio) and check three of
    the four criteria (C2a, C2c, C2d — C2b needs M2.4's control drops so it's deferred).
  Stop at the first passing set S⁺ (or terminate at |S|=30 → status not_localized).
  Greedy-remove sanity check: attempt to shrink S⁺ by dropping each head; if the reduced
    set still passes {C2a, C2c, C2d}, take it. Iterate to fixed point.
  Write:
    hstar/H_${target}.json — the final H* (heads, size, status)
    hstar/${target}_search_log.json — the per-iteration criteria log
    ablation/${target}/main.json — the four metrics for H* (for M2.4 acceptance)
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from belief_utils import (
    load_model_and_tokenizer, load_task, evaluate_task_accuracy, load_json, save_json,
    install_head_scaling_hooks, remove_hooks, evaluate_ppl, set_seed,
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


def _criteria_from(clean: dict, ablated: dict, target_task: str, other_task: str,
                   thr_target=0.30, thr_off=0.10, thr_ppl=1.05):
    drop_target = clean[target_task]["acc"] - ablated[target_task]["acc"]
    drop_other = clean[other_task]["acc"] - ablated[other_task]["acc"]
    drop_wk = clean["world_knowledge"]["acc"] - ablated["world_knowledge"]["acc"]
    ppl_ratio = ablated["ppl"]["ppl"] / max(1e-6, clean["ppl"]["ppl"])
    return {
        "drop_target": drop_target,
        "drop_other_belief": drop_other,
        "drop_world_knowledge": drop_wk,
        "ppl_ratio": ppl_ratio,
        "C2a_pass": drop_target >= thr_target,
        "C2c_pass": drop_other <= thr_off and drop_wk <= thr_off,
        "C2d_pass": ppl_ratio <= thr_ppl,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", required=True, choices=["personal", "attributed"])
    ap.add_argument("--ranked-heads", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--ppl-sample", required=True)
    ap.add_argument("--max-heads", type=int, default=30)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--search-log", required=True)
    ap.add_argument("--output-hstar", required=True)
    ap.add_argument("--output-main-metrics", required=True)
    ap.add_argument("--drop-target-thresh", type=float, default=0.30)
    ap.add_argument("--drop-off-thresh", type=float, default=0.10)
    ap.add_argument("--ppl-ratio-thresh", type=float, default=1.05)
    args = ap.parse_args()

    set_seed(0)
    print(f"[m2.3] {args.model} × {args.target}")
    net, tok = load_model_and_tokenizer(args.model_root, args.model, dtype=args.dtype, device=args.device)
    for p in net.parameters():
        p.requires_grad_(False)

    # load tasks
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
    print(f"[m2.3] tasks loaded: target={len(examples_by_task[target_task])}, other={len(examples_by_task[other_task])}, wk={len(examples_by_task['world_knowledge'])}, ppl_tokens={ppl_tokens.numel()}")

    ranked = load_json(args.ranked_heads)["ranked"]
    print(f"[m2.3] ranked candidates: {len(ranked)}")

    # Clean baselines
    t0 = time.time()
    print("[m2.3] measuring clean baselines...")
    clean = _evaluate_all(net, tok, examples_by_task, ppl_tokens, device=args.device)
    print(f"[m2.3] clean: target={clean[target_task]['acc']:.4f}, other={clean[other_task]['acc']:.4f}, "
          f"wk={clean['world_knowledge']['acc']:.4f}, ppl={clean['ppl']['ppl']:.3f} ({time.time()-t0:.1f}s)")

    search_log = {"model": args.model, "target": args.target, "clean_baselines": clean, "iterations": []}

    def _eval_with_heads(S):
        scale = {(l, h): 0.0 for (l, h) in S}
        handles = install_head_scaling_hooks(net, scale)
        try:
            out = _evaluate_all(net, tok, examples_by_task, ppl_tokens, device=args.device)
        finally:
            remove_hooks(handles)
        return out

    # ---------- Greedy-add ----------
    S = []
    ablated = None
    crit = None
    status = "not_localized"
    for step_i, item in enumerate(ranked):
        head = (int(item["layer"]), int(item["head"]))
        if head in S:
            continue
        S.append(head)
        t1 = time.time()
        ablated = _eval_with_heads(S)
        crit = _criteria_from(clean, ablated, target_task, other_task,
                              args.drop_target_thresh, args.drop_off_thresh, args.ppl_ratio_thresh)
        iter_log = {
            "phase": "greedy_add", "step": step_i, "|S|": len(S), "heads": [list(h) for h in S],
            "added_head": [head[0], head[1]], "criteria": crit,
            "target_acc_ablated": ablated[target_task]["acc"],
            "other_acc_ablated": ablated[other_task]["acc"],
            "wk_acc_ablated": ablated["world_knowledge"]["acc"],
            "ppl_ablated": ablated["ppl"]["ppl"],
            "elapsed_sec": time.time() - t1,
        }
        search_log["iterations"].append(iter_log)
        print(f"[m2.3] add head=(L{head[0]},H{head[1]}) |S|={len(S)} drop_t={crit['drop_target']:.3f} "
              f"drop_o={crit['drop_other_belief']:.3f} drop_wk={crit['drop_world_knowledge']:.3f} "
              f"ppl_r={crit['ppl_ratio']:.3f} pass(a,c,d)=({crit['C2a_pass']},{crit['C2c_pass']},{crit['C2d_pass']}) "
              f"({time.time()-t1:.1f}s)")
        if crit["C2a_pass"] and crit["C2c_pass"] and crit["C2d_pass"]:
            status = "greedy_add_passed"
            break
        if len(S) >= args.max_heads:
            status = "not_localized"
            break

    # ---------- Greedy-remove (only if greedy-add succeeded) ----------
    if status == "greedy_add_passed":
        if len(S) > 1:
            changed = True
            remove_round = 0
            while changed and len(S) > 1:
                changed = False
                remove_round += 1
                # try removing each head in turn (from the least-recently-added first, end of S)
                for i in list(range(len(S)))[::-1]:
                    candidate = [S[j] for j in range(len(S)) if j != i]
                    t1 = time.time()
                    abl2 = _eval_with_heads(candidate)
                    cr2 = _criteria_from(clean, abl2, target_task, other_task,
                                         args.drop_target_thresh, args.drop_off_thresh, args.ppl_ratio_thresh)
                    keeps = cr2["C2a_pass"] and cr2["C2c_pass"] and cr2["C2d_pass"]
                    iter_log = {
                        "phase": "greedy_remove", "round": remove_round,
                        "removed_head": [S[i][0], S[i][1]],
                        "|S|_after_removal": len(candidate), "criteria_after_removal": cr2,
                        "kept_smaller": keeps, "elapsed_sec": time.time() - t1,
                    }
                    search_log["iterations"].append(iter_log)
                    if keeps:
                        S = candidate
                        ablated = abl2
                        crit = cr2
                        changed = True
                        print(f"[m2.3] remove (L{iter_log['removed_head'][0]},H{iter_log['removed_head'][1]}) → |S|={len(S)} (kept)")
                        break
        # regardless of whether greedy-remove found anything or |S|=1, the set passes all criteria
        status = "localized"

    # ---------- Write outputs ----------
    hstar_out = {
        "model": args.model, "target": args.target, "status": status,
        "hstar_heads": [list(h) for h in S],
        "hstar_size": len(S),
        "iterations_greedy_add": sum(1 for it in search_log["iterations"] if it.get("phase") == "greedy_add"),
        "iterations_greedy_remove": sum(1 for it in search_log["iterations"] if it.get("phase") == "greedy_remove"),
    }
    if status == "localized":
        hstar_out["final_criteria"] = crit
        hstar_out["final_metrics"] = {
            "target_acc_clean": clean[target_task]["acc"],
            "target_acc_ablated": ablated[target_task]["acc"],
            "other_acc_clean": clean[other_task]["acc"],
            "other_acc_ablated": ablated[other_task]["acc"],
            "wk_acc_clean": clean["world_knowledge"]["acc"],
            "wk_acc_ablated": ablated["world_knowledge"]["acc"],
            "ppl_clean": clean["ppl"]["ppl"],
            "ppl_ablated": ablated["ppl"]["ppl"],
        }

    save_json(args.output_hstar, hstar_out)
    save_json(args.search_log, search_log)

    main_metrics = {
        "model": args.model, "target": args.target, "status": status,
        "hstar_heads": [list(h) for h in S], "hstar_size": len(S),
        "clean_baselines": clean, "ablated": ablated if ablated is not None else None,
        "criteria": crit if crit is not None else None,
    }
    save_json(args.output_main_metrics, main_metrics)
    print(f"[m2.3] status={status} |H*|={len(S)} → {args.output_hstar}")


if __name__ == "__main__":
    main()
