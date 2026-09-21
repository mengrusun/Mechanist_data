"""M2.a — Empirical Fisher over attention parameters, per belief signal.

Per plan: F_target on the belief-frame's James+Mary subset (454 items) for
personal_belief / attributed_belief; F_knowledge on all 227 world_knowledge items.
"""
import argparse, os
import torch
from belief_lib import (BELIEF_CORE_DIR, MODEL_META, SIGNAL_FRAME,
                        fisher_completion_over_frame, load_frame, load_model,
                        third_person_filter, save_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--signal", required=True, choices=list(SIGNAL_FRAME))
    ap.add_argument("--out", required=True)
    ap.add_argument("--third_person_only", action="store_true", default=True)
    args = ap.parse_args()

    frame = SIGNAL_FRAME[args.signal]
    print(f"[M2a] model={args.model} signal={args.signal} frame={frame}", flush=True)

    # world_knowledge stays full (no person filter meaningful)
    items = load_frame(frame, root=BELIEF_CORE_DIR)
    if frame in ("personal_belief", "attributed_belief") and args.third_person_only:
        items = third_person_filter(items)
    print(f"[M2a] using {len(items)} items", flush=True)

    device = "cuda"
    model, tok, meta = load_model(args.model, device=device, dtype=torch.float16)

    fisher = fisher_completion_over_frame(model, tok, items, device=device)

    # Save as .pt (large fp32 tensors on CPU)
    save_dict = {k: v.cpu() for k, v in fisher.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "model": args.model, "signal": args.signal, "frame": frame,
        "n_items": len(items),
        "fisher_by_pname": save_dict,
    }, args.out)
    print(f"[M2a] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
