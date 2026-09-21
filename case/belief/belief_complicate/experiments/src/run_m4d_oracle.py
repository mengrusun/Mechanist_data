"""M4.d — Oracle prompt-hint baseline.

The oracle uses the ground-truth frame to prepend a frame-specific instruction
to the prompt.  Under identical eval conditions (pinned log-prob metric),
compute per-item baseline correctness and oracle-prompt-hint correctness →
recovery / degradation / net_improvement per frame, world_knowledge delta.

Prompt hints (pinned per task.md):
  * world_knowledge:  "Answer with the true state of the world about this proposition. \n\n" + prompt
  * personal_belief:  "Ignore other characters' beliefs; answer according to your own belief about reality. \n\n" + prompt
  * attributed_belief: "Track the named person's belief and answer according to that person's belief. \n\n" + prompt
"""
import argparse, os, json
import numpy as np
import torch
from belief_lib import (BELIEF_HOLDOUT_DIR, load_frame, load_model, save_json,
                        item_is_correct)


HINTS = {
    "world_knowledge":  "Answer with the true state of the world about this proposition. \n\n",
    "personal_belief":  "Ignore other characters' beliefs; answer according to your own belief about reality. \n\n",
    "attributed_belief": "Track the named person's belief and answer according to that person's belief. \n\n",
}
FRAME_GOLD_NAMES = {"world_knowledge": "reality",
                    "personal_belief": "believe_truth",
                    "attributed_belief": "follow_belief"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[M4d] model={args.model}", flush=True)
    device = "cuda"
    model, tok, _ = load_model(args.model, device=device, dtype=torch.float16)

    # Load full belief_holdout
    all_items = []
    for frame in ["world_knowledge", "personal_belief", "attributed_belief"]:
        items = load_frame(frame, root=BELIEF_HOLDOUT_DIR)
        all_items.extend(items)
    print(f"[M4d] belief_holdout: {len(all_items)} items", flush=True)

    per_item_baseline = []
    per_item_oracle = []
    for i, it in enumerate(all_items):
        # Ground-truth frame is embedded in the gold-vs-distractor semantics of the item:
        # believe_truth → personal_belief, follow_belief → attributed_belief, reality → world_knowledge
        gold_frame_str = {"reality": "world_knowledge",
                          "believe_truth": "personal_belief",
                          "follow_belief": "attributed_belief"}[it["frame"]]
        # Baseline: unmodified forward
        ok_b, lp_g_b, lp_d_b = item_is_correct(
            model, tok, it["prompt"], it["gold"], it["distractor"], device)
        # Oracle: prepend the ground-truth frame's hint
        hinted_prompt = HINTS[gold_frame_str] + it["prompt"]
        ok_o, lp_g_o, lp_d_o = item_is_correct(
            model, tok, hinted_prompt, it["gold"], it["distractor"], device)
        per_item_baseline.append({"idx": i, "gold_frame": it["frame"],
                                  "logp_gold": lp_g_b, "logp_distractor": lp_d_b,
                                  "correct": ok_b})
        per_item_oracle.append({"idx": i, "gold_frame": it["frame"],
                                "logp_gold": lp_g_o, "logp_distractor": lp_d_o,
                                "correct": ok_o})
        if (i + 1) % 100 == 0:
            print(f"[M4d] {i+1}/{len(all_items)}", flush=True)

    # Item-level metrics per frame
    per_frame = {}
    for frame in ["world_knowledge", "personal_belief", "attributed_belief"]:
        gold_name = FRAME_GOLD_NAMES[frame]
        b = [x for x in per_item_baseline if x["gold_frame"] == gold_name]
        o = [x for x in per_item_oracle if x["gold_frame"] == gold_name]
        n = len(b)
        if n == 0:
            per_frame[frame] = None
            continue
        bc = np.array([x["correct"] for x in b])
        oc = np.array([x["correct"] for x in o])
        recovery = int(((~bc) & oc).sum())
        degradation = int((bc & (~oc)).sum())
        per_frame[frame] = {
            "n_items": n,
            "baseline_correct": int(bc.sum()),
            "oracle_correct": int(oc.sum()),
            "baseline_accuracy": float(bc.mean()),
            "oracle_accuracy": float(oc.mean()),
            "recovery": recovery,
            "degradation": degradation,
            "net_improvement": recovery - degradation,
            "recovery_rate": recovery / n,
            "degradation_rate": degradation / n,
            "net_improvement_rate": (recovery - degradation) / n,
        }

    out = {
        "model": args.model,
        "n_holdout_items": len(all_items),
        "hint_templates": HINTS,
        "per_frame": per_frame,
        "baseline_per_item": per_item_baseline,
        "oracle_per_item": per_item_oracle,
    }
    save_json(out, args.out)
    print(f"[M4d] DONE personal_belief_net={per_frame['personal_belief']['net_improvement'] if per_frame['personal_belief'] else 'N/A'} "
          f"attributed_belief_net={per_frame['attributed_belief']['net_improvement'] if per_frame['attributed_belief'] else 'N/A'}",
          flush=True)


if __name__ == "__main__":
    main()
