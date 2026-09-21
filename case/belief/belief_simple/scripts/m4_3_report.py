"""M4.3 report: combine baseline_no_control + controller + prompt_hint OOD evals into
per-model M4_report.json + M4_report.md."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from belief_utils import load_json, save_json


def _idx_key(pe):
    return (pe["task"], pe["prop_idx"], pe["person"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--controller-dir", required=True, help="dir with ood_baseline_no_control.json / ood_controller.json / ood_prompt_hint.json")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()

    base = load_json(os.path.join(args.controller_dir, "ood_baseline_no_control.json"))
    ctrl = load_json(os.path.join(args.controller_dir, "ood_controller.json"))
    hint = load_json(os.path.join(args.controller_dir, "ood_prompt_hint.json"))

    # Compare per-example on the shared examples (index by (task, prop_idx, person))
    base_map = {_idx_key(p): p for p in base["per_example"]}
    ctrl_map = {_idx_key(p): p for p in ctrl["per_example"]}
    hint_map = {_idx_key(p): p for p in hint["per_example"]}

    # per-task
    per_task = {}
    for task in ("world_knowledge", "personal_belief", "attributed_belief"):
        n = base["per_task"][task]["n"]
        per_task[task] = {
            "n": n,
            "acc_baseline_no_control": base["per_task"][task]["acc"],
            "acc_controller": ctrl["per_task"][task]["acc"],
            "acc_prompt_hint": hint["per_task"][task]["acc"],
        }

    # controller vs baseline: recovered/degraded on belief tasks
    belief_recovered = 0
    belief_degraded = 0
    for k, cp in ctrl_map.items():
        bp = base_map.get(k)
        if bp is None or cp["task"] == "world_knowledge":
            continue
        if bp["correct"] == 0 and cp["correct"] == 1:
            belief_recovered += 1
        elif bp["correct"] == 1 and cp["correct"] == 0:
            belief_degraded += 1
    net_impr = belief_recovered - belief_degraded

    # prompt_hint vs baseline: same
    hint_recovered = 0; hint_degraded = 0
    for k, hp in hint_map.items():
        bp = base_map.get(k)
        if bp is None or hp["task"] == "world_knowledge":
            continue
        if bp["correct"] == 0 and hp["correct"] == 1:
            hint_recovered += 1
        elif bp["correct"] == 1 and hp["correct"] == 0:
            hint_degraded += 1
    hint_net = hint_recovered - hint_degraded

    result = {
        "model": args.model,
        "alpha_personal": ctrl.get("alpha_personal"),
        "alpha_attributed": ctrl.get("alpha_attributed"),
        "per_task": per_task,
        "controller_vs_baseline": {"belief_recovered": belief_recovered, "belief_degraded": belief_degraded,
                                    "net_improvement": net_impr},
        "prompt_hint_vs_baseline": {"belief_recovered": hint_recovered, "belief_degraded": hint_degraded,
                                     "net_improvement": hint_net},
        "frame_acc_ood": ctrl.get("frame_acc_ood"),
        "ppl_clean": ctrl.get("ppl_clean"),
        "ppl_controller_ood": ctrl.get("ppl_controller_ood"),
    }
    save_json(args.output_json, result)

    lines = [f"# M4 — OOD Report ({args.model})\n",
             f"- α_personal* = **{ctrl.get('alpha_personal')}**, α_attributed* = **{ctrl.get('alpha_attributed')}**\n",
             f"- Frame classifier OOD accuracy: **{ctrl.get('frame_acc_ood', 0.0):.4f}**\n",
             ""]
    lines.append("## Per-task OOD accuracy\n")
    lines.append("| Task | n | baseline | controller | prompt_hint |")
    lines.append("|------|---:|---------:|-----------:|------------:|")
    for task in ("world_knowledge", "personal_belief", "attributed_belief"):
        pt = per_task[task]
        lines.append(f"| {task} | {pt['n']} | {pt['acc_baseline_no_control']:.4f} | {pt['acc_controller']:.4f} | {pt['acc_prompt_hint']:.4f} |")
    lines.append("")
    lines.append("## Recovered / degraded (belief tasks only, vs baseline)\n")
    lines.append(f"- Controller: recovered={belief_recovered}, degraded={belief_degraded}, net_improvement=**{net_impr}**")
    lines.append(f"- Prompt-hint baseline: recovered={hint_recovered}, degraded={hint_degraded}, net_improvement=**{hint_net}**")
    lines.append("")
    if ctrl.get("ppl_clean") and ctrl.get("ppl_controller_ood"):
        lines.append("## PPL preservation (pretraining sample)\n")
        lines.append(f"- Clean: **{ctrl['ppl_clean']['ppl']:.3f}**")
        lines.append(f"- Controller-on: **{ctrl['ppl_controller_ood']['ppl']:.3f}** (ratio {ctrl['ppl_controller_ood']['ppl']/ctrl['ppl_clean']['ppl']:.3f}×)")

    with open(args.output_md, "w") as f:
        f.write("\n".join(lines))
    print(f"[m4.3-rep] wrote {args.output_json} + {args.output_md}")


if __name__ == "__main__":
    main()
