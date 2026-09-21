"""M2.e — Random-mask baseline (20 seeds, parameter-count matched).

Sample 20 random parameter masks of size P = mask_footprint from the
attention weight matrices only (same family as Fisher masks: query_key_value.weight
and attention.dense.weight, all layers).  For each, zero those parameters
(temporarily), evaluate accuracies + Pile PPL, then restore.
"""
import argparse, json, os
import numpy as np
import torch
from belief_lib import (MODEL_META, evaluate_frame, head_slices_for_model,
                        load_frame, load_model, pile_ppl, save_json, set_seed,
                        third_person_filter)

BELIEF_FRAME_ALT = {"personal_belief": "attributed_belief",
                    "attributed_belief": "personal_belief"}


def get_attention_pnames_and_sizes(model):
    """Return list of (pname, numel) restricted to
       query_key_value.weight + attention.dense.weight, across all layers."""
    out = []
    for name, p in model.named_parameters():
        if ("attention.query_key_value.weight" in name
                or "attention.dense.weight" in name):
            out.append((name, p.numel(), tuple(p.shape)))
    return out


def sample_random_mask(pnames_sizes, target_P: int, rng: np.random.Generator):
    """Sample `target_P` flat indices uniformly across the pool.  Returns
    dict[pname] -> list of flat indices to zero."""
    total = sum(sz for _, sz, _ in pnames_sizes)
    if target_P > total:
        target_P = total
    picks = rng.choice(total, size=target_P, replace=False)
    picks = np.sort(picks)
    # Assign each flat index to the correct pname bucket
    cutoffs = np.cumsum([sz for _, sz, _ in pnames_sizes])
    starts = np.concatenate([[0], cutoffs[:-1]])
    per_pname = {}
    idx_bucket = np.searchsorted(cutoffs, picks, side="right")  # which pname
    for b in range(len(pnames_sizes)):
        sel = picks[idx_bucket == b] - starts[b]
        if len(sel):
            per_pname[pnames_sizes[b][0]] = sel
    return per_pname


def apply_zero_mask(model, per_pname):
    """Save originals and zero the sampled indices in-place.  Returns
    dict for restoration."""
    saved = {}
    for pname, flat_idx in per_pname.items():
        p = dict(model.named_parameters())[pname]
        with torch.no_grad():
            flat = p.data.view(-1)
            saved[pname] = (flat_idx, flat[flat_idx].clone())
            flat[flat_idx] = 0
    return saved


def restore_from_saved(model, saved):
    for pname, (flat_idx, values) in saved.items():
        p = dict(model.named_parameters())[pname]
        with torch.no_grad():
            flat = p.data.view(-1)
            flat[flat_idx] = values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target_frame", required=True, choices=["personal_belief", "attributed_belief"])
    ap.add_argument("--candidate", required=True, help="M2.b output JSON — read mask_footprint_params")
    ap.add_argument("--n_random", type=int, default=20)
    ap.add_argument("--seed_root", type=int, default=42)
    ap.add_argument("--eval_ppl", action="store_true", default=False,
                    help="also evaluate Pile PPL per sample (slow)")
    ap.add_argument("--pile_docs", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--pile_windows", type=int, default=5000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.candidate, "r") as fh:
        cand = json.load(fh)
    P = int(cand["mask_footprint_params"])
    print(f"[M2e] parameter-count-matched P={P}  n_random={args.n_random}", flush=True)

    other = BELIEF_FRAME_ALT[args.target_frame]
    tgt_items = third_person_filter(load_frame(args.target_frame))
    other_items = third_person_filter(load_frame(other))
    wk_items = load_frame("world_knowledge")

    device = "cuda"
    model, tok, meta = load_model(args.model, device=device, dtype=torch.float16)

    # Clean baselines
    print("[M2e] clean baselines …", flush=True)
    clean_tgt = evaluate_frame(model, tok, tgt_items, device=device, show_progress=False)
    clean_other = evaluate_frame(model, tok, other_items, device=device, show_progress=False)
    clean_wk = evaluate_frame(model, tok, wk_items, device=device, show_progress=False)
    if args.eval_ppl:
        pile_docs = [int(x) for x in args.pile_docs.split(",")]
        clean_ppl = pile_ppl(model, tok, doc_indices=pile_docs,
                            n_windows=args.pile_windows, device=device)
    else:
        clean_ppl = None

    pnames_sizes = get_attention_pnames_and_sizes(model)

    per_sample = []
    for i in range(args.n_random):
        seed = args.seed_root + i
        set_seed(seed)
        rng = np.random.default_rng(seed)
        per_pname = sample_random_mask(pnames_sizes, P, rng)
        saved = apply_zero_mask(model, per_pname)
        try:
            abl_tgt = evaluate_frame(model, tok, tgt_items, device=device, show_progress=False)
            abl_other = evaluate_frame(model, tok, other_items, device=device, show_progress=False)
            abl_wk = evaluate_frame(model, tok, wk_items, device=device, show_progress=False)
            abl_ppl = None
            if args.eval_ppl:
                abl_ppl = pile_ppl(model, tok, doc_indices=pile_docs,
                                  n_windows=args.pile_windows, device=device)
        finally:
            restore_from_saved(model, saved)
        per_sample.append({
            "sample_idx": i, "seed_used": seed,
            "target_drop": clean_tgt["accuracy"] - abl_tgt["accuracy"],
            "other_drop": clean_other["accuracy"] - abl_other["accuracy"],
            "wk_drop": clean_wk["accuracy"] - abl_wk["accuracy"],
            "target_acc_ablated": abl_tgt["accuracy"],
            "other_acc_ablated": abl_other["accuracy"],
            "wk_acc_ablated": abl_wk["accuracy"],
            "pile_ppl_ablated": abl_ppl,
        })
        if (i + 1) % 5 == 0:
            print(f"[M2e] sample {i+1}/{args.n_random}", flush=True)

    def band(name):
        vals = np.array([s[name] for s in per_sample])
        return {"mean": float(vals.mean()), "std": float(vals.std(ddof=0)),
                "band_2sigma_lo": float(vals.mean() - 2 * vals.std(ddof=0)),
                "band_2sigma_hi": float(vals.mean() + 2 * vals.std(ddof=0))}

    out = {
        "model": args.model, "target_frame": args.target_frame,
        "P": P, "n_random": args.n_random,
        "clean_accuracies": {"target": clean_tgt["accuracy"],
                             "other": clean_other["accuracy"],
                             "world_knowledge": clean_wk["accuracy"]},
        "clean_pile_ppl": clean_ppl,
        "target_drop_band": band("target_drop"),
        "other_drop_band": band("other_drop"),
        "wk_drop_band": band("wk_drop"),
        "per_sample": per_sample,
    }
    save_json(out, args.out)
    b = out["target_drop_band"]
    print(f"[M2e] DONE  target_drop mean={b['mean']:.4f}  "
          f"2σ=[{b['band_2sigma_lo']:.4f},{b['band_2sigma_hi']:.4f}]", flush=True)


if __name__ == "__main__":
    main()
