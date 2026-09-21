"""M2.4 acceptance: aggregate the 40 controls per (model, target) and compute the final
Claim-2 verdict.

Reads:
  refine-logs/artifacts/ablation/${model}/${target}/main.json  (from M2.3)
  refine-logs/artifacts/ablation/${model}/${target}/random_head/seed{100..119}.json
  refine-logs/artifacts/ablation/${model}/${target}/random_mask/seed{200..219}.json
Writes:
  refine-logs/artifacts/hstar/${model}/H_{target}_acceptance.json
"""

import argparse
import glob
import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from belief_utils import load_json, save_json


TASK_KEY = {"personal": "personal_belief", "attributed": "attributed_belief"}
OTHER_KEY = {"personal": "attributed_belief", "attributed": "personal_belief"}


def _collect(root: str, model: str, target: str, kind: str):
    d = os.path.join(root, model, target, kind)
    files = sorted(glob.glob(os.path.join(d, "seed*.json")))
    out = []
    for f in files:
        r = load_json(f)
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", required=True, choices=["personal", "attributed"])
    ap.add_argument("--ablation-root", default="refine-logs/artifacts/ablation")
    ap.add_argument("--output", required=True)
    ap.add_argument("--drop-target-thresh", type=float, default=0.30)
    ap.add_argument("--drop-off-thresh", type=float, default=0.10)
    ap.add_argument("--ppl-ratio-thresh", type=float, default=1.05)
    args = ap.parse_args()

    target_task = TASK_KEY[args.target]
    other_task = OTHER_KEY[args.target]

    # main H* metrics from M2.3
    main_path = os.path.join(args.ablation_root, args.model, args.target, "main.json")
    main = load_json(main_path)
    clean = main["clean_baselines"]
    ablated = main["ablated"]
    if ablated is None:
        # not localized
        acceptance = {
            "model": args.model, "target": args.target, "status": main["status"],
            "message": "H* not localized in M2.3; no controls applicable",
            "final_verdict": "not_localized",
        }
        save_json(args.output, acceptance)
        return

    drop_target = clean[target_task]["acc"] - ablated[target_task]["acc"]
    drop_other = clean[other_task]["acc"] - ablated[other_task]["acc"]
    drop_wk = clean["world_knowledge"]["acc"] - ablated["world_knowledge"]["acc"]
    ppl_ratio = ablated["ppl"]["ppl"] / max(1e-6, clean["ppl"]["ppl"])

    # collect controls
    rh = _collect(args.ablation_root, args.model, args.target, "random_head")
    rm = _collect(args.ablation_root, args.model, args.target, "random_mask")
    if len(rh) < 20 or len(rm) < 20:
        print(f"[m2.4-acc] WARNING: expected 20+20 controls, got {len(rh)}+{len(rm)}")

    def _drop(r, task): return clean[task]["acc"] - r["ablated"][task]["acc"]

    rh_target_drops = [_drop(r, target_task) for r in rh]
    rm_target_drops = [_drop(r, target_task) for r in rm]

    rh_mean = statistics.mean(rh_target_drops) if rh_target_drops else 0.0
    rh_sigma = statistics.pstdev(rh_target_drops) if len(rh_target_drops) > 1 else 0.0
    threshold_2sigma = rh_mean + 2 * rh_sigma

    C2a_pass = drop_target >= args.drop_target_thresh
    C2b_pass = drop_target > threshold_2sigma
    C2c_pass = drop_other <= args.drop_off_thresh and drop_wk <= args.drop_off_thresh
    C2d_pass = ppl_ratio <= args.ppl_ratio_thresh
    all_pass = C2a_pass and C2b_pass and C2c_pass and C2d_pass
    verdict = "localized" if all_pass else "partially_localized" if (C2a_pass or C2b_pass) else "not_localized"

    acceptance = {
        "model": args.model, "target": args.target,
        "hstar_heads": main["hstar_heads"], "hstar_size": main["hstar_size"],
        "clean_baselines": {k: (v if k == "ppl" else {"acc": v["acc"]}) for k, v in clean.items()},
        "hstar_ablated": {k: (v if k == "ppl" else {"acc": v["acc"]}) for k, v in ablated.items()},
        "drops": {"target": drop_target, "other_belief": drop_other, "world_knowledge": drop_wk,
                  "ppl_ratio": ppl_ratio},
        "random_head": {
            "n_controls": len(rh),
            "target_drops": rh_target_drops,
            "mean": rh_mean, "sigma": rh_sigma, "threshold_2sigma": threshold_2sigma,
        },
        "random_mask": {
            "n_controls": len(rm),
            "target_drops": rm_target_drops,
            "mean": statistics.mean(rm_target_drops) if rm_target_drops else 0.0,
            "sigma": statistics.pstdev(rm_target_drops) if len(rm_target_drops) > 1 else 0.0,
        },
        "criteria": {
            "C2a_drop_target_ge_0.30": C2a_pass,
            "C2b_drop_target_gt_meanplus2sigma": C2b_pass,
            "C2c_off_target_drops_le_0.10": C2c_pass,
            "C2d_ppl_ratio_le_1.05": C2d_pass,
        },
        "final_verdict": verdict,
    }
    save_json(args.output, acceptance)
    print(f"[m2.4-acc] {args.model} × {args.target}: verdict={verdict} drop_target={drop_target:.3f} "
          f"threshold={threshold_2sigma:.3f} → {args.output}")


if __name__ == "__main__":
    main()
