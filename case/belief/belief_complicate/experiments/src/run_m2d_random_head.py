"""M2.d — Random-head baseline (20 seeds, head-count matched).

Sample 20 head sets of size K = |HeadSet_target| uniformly from all
(n_layers × n_heads) heads.  For each, zero-ablate and evaluate target /
other-belief / world_knowledge accuracy on the third-person subset.
Report mean ± 2σ of each metric.
"""
import argparse, json, os, random
import numpy as np
import torch
from belief_lib import (MODEL_META, evaluate_frame, load_frame, load_model,
                        make_head_zero_ablation, save_json, set_seed,
                        third_person_filter)

BELIEF_FRAME_ALT = {"personal_belief": "attributed_belief",
                    "attributed_belief": "personal_belief"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target_frame", required=True, choices=["personal_belief", "attributed_belief"])
    ap.add_argument("--headset_json", required=True, help="M2.c output JSON")
    ap.add_argument("--headset_size", type=int, default=None,
                    help="Override head-count K; defaults to |head_set| from M2.c "
                         "or best_reported_step's k if no passing set")
    ap.add_argument("--n_random", type=int, default=20)
    ap.add_argument("--seeds", default="42,200,201")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.headset_json, "r") as fh:
        m2c = json.load(fh)
    K = args.headset_size
    if K is None:
        if m2c.get("head_set_size"):
            K = int(m2c["head_set_size"])
        elif m2c.get("best_reported_step"):
            K = int(m2c["best_reported_step"]["k"])
        else:
            raise ValueError("cannot infer K")
    print(f"[M2d] K={K} n_random={args.n_random}", flush=True)

    other = BELIEF_FRAME_ALT[args.target_frame]
    tgt_items = third_person_filter(load_frame(args.target_frame))
    other_items = third_person_filter(load_frame(other))
    wk_items = load_frame("world_knowledge")

    device = "cuda"
    model, tok, meta = load_model(args.model, device=device, dtype=torch.float16)

    # Clean baselines (same as M2.c would compute, but recompute for standalone use)
    print("[M2d] clean baselines …", flush=True)
    clean_tgt = evaluate_frame(model, tok, tgt_items, device=device, show_progress=False)
    clean_other = evaluate_frame(model, tok, other_items, device=device, show_progress=False)
    clean_wk = evaluate_frame(model, tok, wk_items, device=device, show_progress=False)

    all_heads = [(L, H) for L in range(meta.n_layers) for H in range(meta.n_heads)]
    # Aggregate seeds -> use each seed to draw n_random samples? Per plan, "20 random
    # sets"; seeds are for reproducibility. We fix the total sample count to args.n_random
    # and seed from the first seed for reproducibility.
    seeds = [int(x) for x in args.seeds.split(",")]

    per_sample = []
    rng = random.Random(seeds[0])
    for i in range(args.n_random):
        set_seed(seeds[0] + i)  # ensure a fresh RNG state per sample (deterministic)
        rng = random.Random(seeds[0] + i)
        sample = rng.sample(all_heads, K)
        with make_head_zero_ablation(model, sample):
            abl_tgt = evaluate_frame(model, tok, tgt_items, device=device, show_progress=False)
            abl_other = evaluate_frame(model, tok, other_items, device=device, show_progress=False)
            abl_wk = evaluate_frame(model, tok, wk_items, device=device, show_progress=False)
        per_sample.append({
            "sample_idx": i, "seed_used": seeds[0] + i, "head_set": sample,
            "target_drop": clean_tgt["accuracy"] - abl_tgt["accuracy"],
            "other_drop": clean_other["accuracy"] - abl_other["accuracy"],
            "wk_drop": clean_wk["accuracy"] - abl_wk["accuracy"],
            "target_acc_ablated": abl_tgt["accuracy"],
            "other_acc_ablated": abl_other["accuracy"],
            "wk_acc_ablated": abl_wk["accuracy"],
        })
        if (i + 1) % 5 == 0:
            print(f"[M2d] sample {i+1}/{args.n_random}", flush=True)

    def band(name):
        vals = np.array([s[name] for s in per_sample])
        return {"mean": float(vals.mean()), "std": float(vals.std(ddof=0)),
                "band_2sigma_lo": float(vals.mean() - 2 * vals.std(ddof=0)),
                "band_2sigma_hi": float(vals.mean() + 2 * vals.std(ddof=0))}

    out = {
        "model": args.model, "target_frame": args.target_frame,
        "K": K, "n_random": args.n_random, "seeds": seeds,
        "clean_accuracies": {"target": clean_tgt["accuracy"],
                             "other": clean_other["accuracy"],
                             "world_knowledge": clean_wk["accuracy"]},
        "target_drop_band": band("target_drop"),
        "other_drop_band": band("other_drop"),
        "wk_drop_band": band("wk_drop"),
        "per_sample": per_sample,
    }
    save_json(out, args.out)
    b = out["target_drop_band"]
    print(f"[M2d] DONE  target_drop mean={b['mean']:.4f}  2σ=[{b['band_2sigma_lo']:.4f},{b['band_2sigma_hi']:.4f}]", flush=True)


if __name__ == "__main__":
    main()
