"""M3 aggregation: read all per-checkpoint jsons, build behavioural + causal trajectories,
apply emergence criteria, write summary.json + summary.md.
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from belief_utils import load_json, save_json


TARGET_TO_TASK = {"personal": "personal_belief", "attributed": "attributed_belief"}


def _step_of(fname: str) -> int:
    m = re.search(r"step(\d+)\.json$", fname)
    return int(m.group(1)) if m else -1


def _emergence(steps, values, thresh: float, persistence: int = 2, next_k: int = 3):
    """First index i such that values[i] >= thresh AND at least `persistence` of the next
    `next_k` values (i.e., values[i+1:i+1+next_k]) are >= thresh. Returns (step, i) or (None, None)."""
    for i in range(len(values)):
        if values[i] < thresh:
            continue
        window = values[i + 1 : i + 1 + next_k]
        if len(window) < persistence:
            # not enough future points to check persistence — reject
            continue
        if sum(1 for v in window if v >= thresh) >= persistence:
            return steps[i], i
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formation-dir", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    ap.add_argument("--behav-thresh", type=float, default=0.60)
    ap.add_argument("--causal-thresh", type=float, default=0.20)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.formation_dir, "step*.json")), key=_step_of)
    files = [f for f in files if _step_of(f) >= 0]
    print(f"[m3-agg] found {len(files)} checkpoint files")

    records = []
    for f in files:
        r = load_json(f)
        step = _step_of(f)
        rec = {"step": step, "checkpoint": r.get("checkpoint")}
        rec["behavioural"] = {t: r["behavioural"][t]["acc"] for t in ("world_knowledge", "personal_belief", "attributed_belief")}
        for name in ("hstar_personal_ablated", "hstar_attributed_ablated"):
            block = r.get(name, {})
            if isinstance(block, dict) and all(v == "not_applicable" for v in block.values()):
                rec[name] = "not_applicable"
            else:
                rec[name] = {t: block[t]["acc"] if isinstance(block.get(t), dict) else None
                             for t in ("world_knowledge", "personal_belief", "attributed_belief")}
        records.append(rec)

    steps = [r["step"] for r in records]

    # Emergence detection per target
    emergences = {}
    for target, target_task in TARGET_TO_TASK.items():
        behav_vals = [r["behavioural"][target_task] for r in records]
        t_star, _ = _emergence(steps, behav_vals, args.behav_thresh)
        # causal Δ using the matching H*_target
        ablation_key = f"hstar_{target}_ablated"
        first_valid = next((r[ablation_key] for r in records if r[ablation_key] != "not_applicable"), None)
        if first_valid is None:
            t_dag = None
            causal_available = False
        else:
            causal_available = True
            delta_vals = []
            for r in records:
                if r[ablation_key] == "not_applicable":
                    delta_vals.append(-1.0)  # sentinel — treat as always below threshold
                else:
                    behav_target = r["behavioural"][target_task]
                    abl_target = r[ablation_key][target_task]
                    delta_vals.append(behav_target - abl_target)
            t_dag, _ = _emergence(steps, delta_vals, args.causal_thresh)
        window = "not observed"
        if t_star is not None and t_dag is not None:
            window = [t_star, t_dag]
        elif t_star is not None:
            window = [t_star]
        emergences[target] = {
            "behavioural_emergence_step": t_star,
            "causal_emergence_step": t_dag,
            "causal_trajectory_available": causal_available,
            "formation_window": window,
        }

    # Distinctness check
    p, a = emergences["personal"], emergences["attributed"]
    distinct = (p["behavioural_emergence_step"] != a["behavioural_emergence_step"]) or \
               (p["causal_emergence_step"] != a["causal_emergence_step"])

    summary = {
        "n_checkpoints": len(records),
        "steps": steps,
        "behav_thresh": args.behav_thresh,
        "causal_thresh": args.causal_thresh,
        "trajectories": records,
        "emergences": emergences,
        "distinct_windows": distinct,
    }
    save_json(args.output_json, summary)

    # Markdown report
    lines = []
    lines.append("# M3 — Formation Window Summary\n")
    lines.append(f"- Checkpoints evaluated: **{len(records)}** ({steps[0]} → {steps[-1]})\n")
    lines.append(f"- Behavioural emergence threshold: acc ≥ **{args.behav_thresh}** (2-of-next-3 persistence)")
    lines.append(f"- Causal emergence threshold: Δ ≥ **{args.causal_thresh}** (2-of-next-3 persistence)\n")
    for target in ("personal", "attributed"):
        e = emergences[target]
        lines.append(f"### {target}")
        lines.append(f"- Behavioural t* = **{e['behavioural_emergence_step']}**")
        lines.append(f"- Causal t† = **{e['causal_emergence_step']}** (available: {e['causal_trajectory_available']})")
        lines.append(f"- Formation window = **{e['formation_window']}**\n")
    lines.append(f"**Distinct windows across targets: {distinct}**\n")
    lines.append("\n## Behavioural trajectory (acc)\n")
    lines.append("| step | WK | personal | attributed |")
    lines.append("|-----:|---:|---------:|-----------:|")
    for r in records:
        b = r["behavioural"]
        lines.append(f"| {r['step']} | {b['world_knowledge']:.3f} | {b['personal_belief']:.3f} | {b['attributed_belief']:.3f} |")
    with open(args.output_md, "w") as f:
        f.write("\n".join(lines))
    print(f"[m3-agg] wrote {args.output_json} + {args.output_md}")


if __name__ == "__main__":
    main()
