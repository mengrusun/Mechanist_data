"""M3.b — Causal-intervention trajectory: at each pythia-1b intermediate
checkpoint, zero-ablate the Claim-2 head set (from final pythia-1b) and
evaluate accuracy on all three frames (third-person subset for belief).

Head-set transfer: architectural (same layer/head grid) — see FINAL_PROPOSAL R6.
"""
import argparse, json, os
import torch
from belief_lib import (evaluate_frame, load_frame, load_model,
                        make_head_zero_ablation, save_json, third_person_filter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-1b")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--headset_json", required=True,
                    help="path to M2.c pythia-1b headset_<frame>.json")
    ap.add_argument("--headset_frame", required=True,
                    choices=["personal_belief", "attributed_belief"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.headset_json, "r") as fh:
        m2c = json.load(fh)
    hs = m2c.get("head_set")
    if not hs:
        # localized failure — record the trajectory as skipped
        save_json({"model": args.model, "step": args.step,
                   "headset_frame": args.headset_frame,
                   "headset_json": args.headset_json,
                   "skipped_reason": f"M2.c verdict={m2c.get('verdict')}; no localized head set."},
                  args.out)
        print(f"[M3b] SKIP step={args.step} (no head set)", flush=True)
        return
    head_set = [tuple(x) for x in hs]

    print(f"[M3b] model={args.model} step={args.step} "
          f"|HeadSet_{args.headset_frame}|={len(head_set)}", flush=True)
    device = "cuda"
    model, tok, meta = load_model(args.model, ckpt_step=args.step,
                                 device=device, dtype=torch.float16)

    results = {}
    for frame in ["world_knowledge", "personal_belief", "attributed_belief"]:
        items = load_frame(frame)
        if frame in ("personal_belief", "attributed_belief"):
            items = third_person_filter(items)
        # clean
        clean = evaluate_frame(model, tok, items, device=device, show_progress=False)
        # ablated
        with make_head_zero_ablation(model, head_set):
            abl = evaluate_frame(model, tok, items, device=device, show_progress=False)
        results[frame] = {
            "n_items": clean["n_items"],
            "clean_accuracy": clean["accuracy"],
            "ablated_accuracy": abl["accuracy"],
            "drop": clean["accuracy"] - abl["accuracy"],
        }
        print(f"  [M3b] {frame}: clean={clean['accuracy']:.3f} "
              f"abl={abl['accuracy']:.3f} drop={results[frame]['drop']:.3f}", flush=True)

    save_json({"model": args.model, "step": args.step,
               "headset_frame": args.headset_frame,
               "head_set": head_set, "head_set_size": len(head_set),
               "frames": results}, args.out)
    print(f"[M3b] DONE step={args.step}", flush=True)


if __name__ == "__main__":
    main()
