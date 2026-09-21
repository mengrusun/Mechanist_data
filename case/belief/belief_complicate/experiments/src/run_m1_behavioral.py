"""M1 — behavioral eval (Claim 1). One (model × frame) invocation."""
import argparse, os
from belief_lib import (BELIEF_CORE_DIR, MODEL_META, evaluate_frame, load_frame, load_model,
                        binomial_pvalue_above_chance, third_person_filter, save_json)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--frame", required=True, choices=["world_knowledge", "personal_belief", "attributed_belief"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = ap.parse_args()
    import torch
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    device = "cuda"
    print(f"[M1] model={args.model} frame={args.frame} -> {args.out}", flush=True)
    model, tok, meta = load_model(args.model, device=device, dtype=dtype)
    items = load_frame(args.frame, root=BELIEF_CORE_DIR)
    print(f"[M1] loaded {len(items)} items from {args.frame}", flush=True)

    # Full belief_core (all persons) per plan (M1 uses full data for Claim 1)
    res = evaluate_frame(model, tok, items, device=device)

    # Above-chance test:
    # world_knowledge is single-completion factual — the plan/HARD-CONSTRAINT threshold
    # (n=245/454 = p<0.05 one-sided binomial vs 0.5 chance) is defined on the third-person
    # subset (454 items) for the Claim-2 eligibility gate.  Report both full-set p-value
    # and third-person-subset p-value.
    p_full = binomial_pvalue_above_chance(res["n_correct"], res["n_items"], p0=0.5)

    # Third-person subset for above_chance eligibility
    third_items = third_person_filter(items) if args.frame in ("personal_belief", "attributed_belief") else items
    if args.frame in ("personal_belief", "attributed_belief"):
        third_correct = sum(1 for x, it in zip(res["per_item"], items)
                            if str(it.get("person", "")).lower() in ("james", "mary") and x["correct"])
        third_n = sum(1 for it in items if str(it.get("person", "")).lower() in ("james", "mary"))
    else:
        third_correct = res["n_correct"]
        third_n = res["n_items"]
    p_third = binomial_pvalue_above_chance(third_correct, third_n, p0=0.5) if third_n else 1.0

    summary = {
        "model": args.model,
        "frame": args.frame,
        "n_items": res["n_items"],
        "n_correct": res["n_correct"],
        "accuracy": res["accuracy"],
        "third_person_subset_n_items": third_n,
        "third_person_subset_n_correct": third_correct,
        "third_person_subset_accuracy": third_correct / third_n if third_n else 0.0,
        "binomial_p_full": p_full,
        "binomial_p_third_person": p_third,
        "above_chance_full": p_full < 0.05,
        "above_chance_third_person": p_third < 0.05,
        "per_item": res["per_item"],
    }
    save_json(summary, args.out)
    print(f"[M1] DONE  acc(full)={res['accuracy']:.4f}  "
          f"acc(third)={summary['third_person_subset_accuracy']:.4f}  p_third={p_third:.4g}", flush=True)


if __name__ == "__main__":
    main()
