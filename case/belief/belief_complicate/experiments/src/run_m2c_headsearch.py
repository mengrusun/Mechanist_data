"""M2.c — Smallest-passing-head-set search via zero-ablation.

Optimization: PPL is expensive (~10.24M forward-token pass); only run PPL
when accuracy criteria already pass — otherwise the PPL check is wasted.

Search protocol per plan:
  * Start with empty head set; incrementally add heads in candidate-ranking order.
  * At each k, evaluate accuracy on target / other-belief / world_knowledge
    (third-person subset for belief frames; full 227 for wk).
  * If target-drop >= 0.30 AND off-targets <= 0.10 AND wk-drop <= 0.10 → run Pile PPL.
      * If Pile PPL <= 1.05× clean → PASS at this k, record head-set as
        smallest-passing set and stop.
  * If off-target drop begins exceeding 0.10 while target-drop keeps growing,
    KEEP going (we might find a larger passing set is impossible — but the
    plan defines "partially localized" via the accuracy criteria — record and
    continue up to max_heads_search).

Fixed thresholds (task.md-pinned; DO NOT relax):
  * target_drop_min = 0.30
  * offtarget_max   = 0.10
  * ppl_ratio_max   = 1.05
"""
import argparse, json, os
import torch
from belief_lib import (BELIEF_CORE_DIR, MODEL_META, evaluate_frame, load_frame,
                        load_model, make_head_zero_ablation, pile_ppl, save_json,
                        third_person_filter)


BELIEF_FRAME_ALT = {"personal_belief": "attributed_belief",
                    "attributed_belief": "personal_belief"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True, help="M2.b output JSON")
    ap.add_argument("--target_frame", required=True, choices=["personal_belief", "attributed_belief"])
    ap.add_argument("--max_heads_search", type=int, default=50)
    ap.add_argument("--target_drop_min", type=float, default=0.30)
    ap.add_argument("--offtarget_max", type=float, default=0.10)
    ap.add_argument("--ppl_ratio_max", type=float, default=1.05)
    ap.add_argument("--pile_docs", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--pile_windows", type=int, default=5000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.candidate, "r") as fh:
        cand = json.load(fh)
    head_ranking = [(int(h["layer"]), int(h["head"]), int(h["per_head_mass"]))
                    for h in cand["head_ranking"] if int(h["per_head_mass"]) > 0]
    print(f"[M2c] {len(head_ranking)} candidate heads (nonzero mass); "
          f"max_heads_search={args.max_heads_search}", flush=True)

    other = BELIEF_FRAME_ALT[args.target_frame]
    tgt_items = third_person_filter(load_frame(args.target_frame))
    other_items = third_person_filter(load_frame(other))
    wk_items = load_frame("world_knowledge")

    device = "cuda"
    model, tok, meta = load_model(args.model, device=device, dtype=torch.float16)

    # 1. Clean baseline accuracies
    print("[M2c] clean baseline accuracies …", flush=True)
    clean_tgt = evaluate_frame(model, tok, tgt_items, device=device, show_progress=False)
    clean_other = evaluate_frame(model, tok, other_items, device=device, show_progress=False)
    clean_wk = evaluate_frame(model, tok, wk_items, device=device, show_progress=False)
    print(f"[M2c] clean acc: tgt={clean_tgt['accuracy']:.4f} "
          f"other={clean_other['accuracy']:.4f} wk={clean_wk['accuracy']:.4f}", flush=True)

    # Clean PPL — computed lazily on first candidate PASS to save time; compute now.
    pile_docs = [int(x) for x in args.pile_docs.split(",")]
    print("[M2c] clean Pile PPL …", flush=True)
    clean_ppl = pile_ppl(model, tok, doc_indices=pile_docs, n_windows=args.pile_windows,
                        device=device)
    print(f"[M2c] clean pile_ppl={clean_ppl:.4f}", flush=True)

    # 2. Incremental search
    trace = []
    passing_set = None

    k_max = min(args.max_heads_search, len(head_ranking))
    for k in range(1, k_max + 1):
        head_set = [(L, H) for (L, H, _m) in head_ranking[:k]]
        with make_head_zero_ablation(model, head_set):
            abl_tgt = evaluate_frame(model, tok, tgt_items, device=device, show_progress=False)
            abl_other = evaluate_frame(model, tok, other_items, device=device, show_progress=False)
            abl_wk = evaluate_frame(model, tok, wk_items, device=device, show_progress=False)
            tgt_drop = clean_tgt["accuracy"] - abl_tgt["accuracy"]
            other_drop = clean_other["accuracy"] - abl_other["accuracy"]
            wk_drop = clean_wk["accuracy"] - abl_wk["accuracy"]
            pass_target = tgt_drop >= args.target_drop_min
            pass_other = other_drop <= args.offtarget_max
            pass_wk = wk_drop <= args.offtarget_max

            # PPL check ONLY when accuracy criteria (T/O/W) all pass; otherwise skip PPL
            if pass_target and pass_other and pass_wk:
                print(f"[M2c] k={k:2d}  accuracy criteria all pass → running Pile PPL …", flush=True)
                abl_ppl = pile_ppl(model, tok, doc_indices=pile_docs, n_windows=args.pile_windows,
                                  device=device)
                ppl_ratio = abl_ppl / clean_ppl
                pass_ppl = ppl_ratio <= args.ppl_ratio_max
            else:
                abl_ppl = None
                ppl_ratio = None
                pass_ppl = None

        step = {
            "k": k, "head_set": head_set,
            "target_drop": float(tgt_drop), "other_drop": float(other_drop),
            "wk_drop": float(wk_drop),
            "target_acc_clean": clean_tgt["accuracy"],
            "target_acc_ablated": abl_tgt["accuracy"],
            "other_acc_clean": clean_other["accuracy"],
            "other_acc_ablated": abl_other["accuracy"],
            "wk_acc_clean": clean_wk["accuracy"],
            "wk_acc_ablated": abl_wk["accuracy"],
            "pile_ppl_clean": clean_ppl,
            "pile_ppl_ablated": abl_ppl,
            "pile_ppl_ratio": ppl_ratio,
            "pass_target": bool(pass_target),
            "pass_other": bool(pass_other),
            "pass_wk": bool(pass_wk),
            "pass_ppl": (bool(pass_ppl) if pass_ppl is not None else None),
        }
        trace.append(step)
        print(f"[M2c] k={k:2d}  tgt_drop={tgt_drop:.3f}  "
              f"other_drop={other_drop:.3f}  wk_drop={wk_drop:.3f}"
              + (f"  ppl_ratio={ppl_ratio:.3f}" if ppl_ratio is not None else ""),
              flush=True)

        if pass_target and pass_other and pass_wk and pass_ppl:
            passing_set = head_set
            print(f"[M2c] SMALLEST PASSING SET at k={k}", flush=True)
            break

    # 3. Verdict rules
    verdict = "not_localized"
    max_target_drop = max(s["target_drop"] for s in trace) if trace else 0.0
    if passing_set is not None:
        verdict = "localized"
    elif max_target_drop >= args.target_drop_min:
        verdict = "partially_localized"

    # Best partial step: earliest k with target-drop >= threshold, else last
    best_step_for_report = None
    for s in trace:
        if s["target_drop"] >= args.target_drop_min:
            best_step_for_report = s
            break
    if best_step_for_report is None and trace:
        best_step_for_report = trace[-1]

    out = {
        "model": args.model,
        "target_frame": args.target_frame,
        "other_belief_frame": other,
        "verdict": verdict,
        "head_set": passing_set,
        "head_set_size": len(passing_set) if passing_set else None,
        "clean_accuracies": {
            "target": clean_tgt["accuracy"],
            "other": clean_other["accuracy"],
            "world_knowledge": clean_wk["accuracy"],
        },
        "clean_pile_ppl": clean_ppl,
        "search_trace": trace,
        "best_reported_step": best_step_for_report,
        "candidate_source": args.candidate,
        "pile_docs": pile_docs,
        "pile_windows": args.pile_windows,
        "thresholds": {
            "target_drop_min": args.target_drop_min,
            "offtarget_max": args.offtarget_max,
            "ppl_ratio_max": args.ppl_ratio_max,
        },
    }
    save_json(out, args.out)
    print(f"[M2c] DONE  verdict={verdict}  head_set_size={out['head_set_size']}", flush=True)


if __name__ == "__main__":
    main()
