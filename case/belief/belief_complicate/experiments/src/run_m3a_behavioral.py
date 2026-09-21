"""M3.a — Behavioral trajectory: per pythia-1b intermediate checkpoint,
evaluate accuracy on world_knowledge / personal_belief / attributed_belief on
full belief_core.
"""
import argparse, os
import torch
from belief_lib import (BELIEF_CORE_DIR, MODEL_META, evaluate_frame,
                        load_frame, load_model, save_json)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pythia-1b")
    ap.add_argument("--step", type=int, required=True,
                    help="checkpoint step id (e.g. 143000)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[M3a] model={args.model} step={args.step}", flush=True)
    device = "cuda"
    model, tok, meta = load_model(args.model, ckpt_step=args.step,
                                 device=device, dtype=torch.float16)

    results = {}
    for frame in ["world_knowledge", "personal_belief", "attributed_belief"]:
        items = load_frame(frame)
        res = evaluate_frame(model, tok, items, device=device, show_progress=False)
        results[frame] = {"n_items": res["n_items"], "n_correct": res["n_correct"],
                          "accuracy": res["accuracy"]}
        print(f"  [M3a] {frame} acc={res['accuracy']:.4f}", flush=True)

    save_json({"model": args.model, "step": args.step, "frames": results}, args.out)
    print(f"[M3a] DONE step={args.step}", flush=True)


if __name__ == "__main__":
    main()
