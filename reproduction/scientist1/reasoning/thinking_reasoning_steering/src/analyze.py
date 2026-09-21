"""
Aggregate judged generations into (behaviour × alpha) dose-response tables and
compare steering-vector control vs. prompt-engineering control.

Inputs:
  - results/steered_judgments.json  (from judge_behaviour.py on steered gens)
  - results/prompt_judgments.json   (from judge_behaviour.py on prompt gens)

Outputs:
  - results/analysis.json  (tables + summary)
  - stdout table
"""
import json, argparse, os
from statistics import mean


def load_judgments(path):
    with open(path) as f: d = json.load(f)
    return d["judgments"] if "judgments" in d else d


def aggregate_steered(judgments):
    """Return {behaviour: {alpha: {intensity_mean, correct_frac, coherence_mean, n}}}."""
    out = {}
    for j in judgments:
        b = j["behaviour"]; a = j["alpha"]
        out.setdefault(b, {}).setdefault(a, []).append(j)
    summary = {}
    for b, per_alpha in out.items():
        summary[b] = {}
        for a, items in per_alpha.items():
            valid = [i for i in items if i["judge_intensity"] >= 0]
            if not valid: continue
            summary[b][a] = {
                "intensity_mean": round(mean(i["judge_intensity"] for i in valid), 3),
                "correct_frac":   round(mean(1 if i["judge_correct"] else 0 for i in valid), 3),
                "coherence_mean": round(mean(i["judge_coherence"] for i in valid), 3),
                "n": len(valid),
            }
    return summary


def aggregate_prompt(judgments):
    """Return {behaviour: {condition: {intensity_mean, correct_frac, coherence_mean, n}}}
       plus a 'none' entry for the true baseline."""
    out = {}
    for j in judgments:
        b = j["behaviour"]; c = j["condition"]
        out.setdefault(b, {}).setdefault(c, []).append(j)
    summary = {}
    for b, per_c in out.items():
        summary[b] = {}
        for c, items in per_c.items():
            valid = [i for i in items if i["judge_intensity"] >= 0]
            if not valid: continue
            summary[b][c] = {
                "intensity_mean": round(mean(i["judge_intensity"] for i in valid), 3),
                "correct_frac":   round(mean(1 if i["judge_correct"] else 0 for i in valid), 3),
                "coherence_mean": round(mean(i["judge_coherence"] for i in valid), 3),
                "n": len(valid),
            }
    return summary


def pretty_steered(summary):
    lines = []
    for b in sorted(summary):
        lines.append(f"\n### Steering-vector control : {b}")
        lines.append(f"  {'alpha':>7} | {'intensity':>10} | {'correct':>8} | {'coherence':>10} | n")
        lines.append("  " + "-" * 55)
        for a in sorted(summary[b], key=float):
            e = summary[b][a]
            lines.append(f"  {a:>+7.2f} | {e['intensity_mean']:>10.2f} | {e['correct_frac']:>8.2f} | {e['coherence_mean']:>10.2f} | {e['n']}")
    return "\n".join(lines)


def pretty_prompt(summary):
    lines = []
    for b in sorted(summary):
        lines.append(f"\n### Prompt control : {b}")
        lines.append(f"  {'cond':>10} | {'intensity':>10} | {'correct':>8} | {'coherence':>10} | n")
        lines.append("  " + "-" * 55)
        for c in sorted(summary[b]):
            e = summary[b][c]
            lines.append(f"  {c:>10} | {e['intensity_mean']:>10.2f} | {e['correct_frac']:>8.2f} | {e['coherence_mean']:>10.2f} | {e['n']}")
    return "\n".join(lines)


def dose_response(summary):
    """Return per behaviour the slope of intensity vs alpha (via simple least-squares)
       and Spearman rank correlation."""
    from statistics import mean
    def rankdata(x):
        s = sorted(range(len(x)), key=lambda i: x[i])
        r = [0]*len(x)
        for rank, idx in enumerate(s): r[idx] = rank
        return r
    def spearman(x, y):
        rx, ry = rankdata(x), rankdata(y)
        return pearson(rx, ry)
    def pearson(x, y):
        mx, my = mean(x), mean(y)
        num = sum((a-mx)*(b-my) for a, b in zip(x, y))
        dx = sum((a-mx)**2 for a in x); dy = sum((b-my)**2 for b in y)
        den = (dx*dy) ** 0.5
        return num / den if den else float("nan")
    ret = {}
    for b, per_alpha in summary.items():
        alphas = sorted(per_alpha, key=float)
        xs = [float(a) for a in alphas]
        ys = [per_alpha[a]["intensity_mean"] for a in alphas]
        # OLS slope
        mx, my = mean(xs), mean(ys)
        num = sum((a-mx)*(b-my) for a, b in zip(xs, ys))
        den = sum((a-mx)**2 for a in xs)
        slope = num/den if den else float("nan")
        ret[b] = {
            "slope_intensity_per_alpha": round(slope, 4),
            "spearman_alpha_intensity":  round(spearman(xs, ys), 4),
            "intensity_at_min_alpha":    ys[0],
            "intensity_at_max_alpha":    ys[-1],
            "intensity_range":           round(max(ys) - min(ys), 3),
            "correct_at_alpha_0":        per_alpha.get(0.0, per_alpha.get("0.0", {})).get("correct_frac", None),
        }
    return ret


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steered", default="results/steered_judgments.json")
    ap.add_argument("--prompt",  default="results/prompt_judgments.json")
    ap.add_argument("--out",     default="results/analysis.json")
    args = ap.parse_args()

    steered_summary = None; prompt_summary = None; dose = None
    if os.path.exists(args.steered):
        steered_summary = aggregate_steered(load_judgments(args.steered))
        dose = dose_response(steered_summary)
        print(pretty_steered(steered_summary))
        print("\n=== Dose-response (steering-vector) ===")
        for b, d in dose.items():
            print(f"  {b}: slope={d['slope_intensity_per_alpha']:+.3f}/α  "
                  f"spearman={d['spearman_alpha_intensity']:+.3f}  "
                  f"range={d['intensity_range']}  "
                  f"acc@α=0={d['correct_at_alpha_0']}")
    if os.path.exists(args.prompt):
        prompt_summary = aggregate_prompt(load_judgments(args.prompt))
        print(pretty_prompt(prompt_summary))

    out = {"steered": steered_summary, "prompt": prompt_summary, "dose_response": dose}
    with open(args.out, "w") as f: json.dump(out, f, indent=2)
    print(f"\n[save] wrote {args.out}")


if __name__ == "__main__":
    main()
