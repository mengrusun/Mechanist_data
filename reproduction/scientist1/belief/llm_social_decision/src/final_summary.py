"""End-to-end summary of experiment results for report writing."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/"results"

def load(name):
    p = RES/name
    if p.suffix == ".json":
        return json.load(open(p))
    return [json.loads(l) for l in open(p)]

def summarise_intervention(intv, alpha_ref=3.0):
    """Return per-kind specificity table:
       rows = injected variable, cols = effect-change on each variable."""
    VARS = ["gender","age","instruction","meeting"]
    baseline = intv["baseline_summary"]
    out = {}
    for kind in ["raw", "pure"]:
        rows = {}
        for tv in VARS:
            for sign in [1.0, -1.0]:
                a = sign * alpha_ref
                m = [c for c in intv["conditions"]
                     if c["kind"]==kind and c["variable"]==tv and c["alpha"]==a]
                if not m: continue
                # effect change on each variable
                delta = {v: m[0]["summary"][v]-baseline[v] for v in VARS}
                delta["mean"] = m[0]["summary"]["mean"] - baseline["mean"]
                rows[f"{tv}_a{a:+.1f}"] = delta
        out[kind] = rows
    return baseline, out


def spec_ratio(intv, alpha_ref=3.0):
    """For each (kind, variable), specificity ratio =
       |delta_target| / max_{v!=target} |delta_v|.
       We use pre-post baseline delta of the effect coefficient (mean E gap
       between the two values of the variable)."""
    VARS = ["gender","age","instruction","meeting"]
    baseline = intv["baseline_summary"]
    out = {}
    for kind in ["raw", "pure"]:
        rows = {}
        for tv in VARS:
            for sign in [1.0, -1.0]:
                a = sign * alpha_ref
                m = [c for c in intv["conditions"]
                     if c["kind"]==kind and c["variable"]==tv and c["alpha"]==a]
                if not m: continue
                delta_tv = m[0]["summary"][tv] - baseline[tv]
                deltas_other = [abs(m[0]["summary"][v]-baseline[v]) for v in VARS if v != tv]
                ratio = abs(delta_tv) / (max(deltas_other) + 1e-9)
                rows[f"{tv}_a{a:+.1f}"] = {"delta_target": delta_tv,
                                          "max_delta_other": max(deltas_other),
                                          "spec_ratio": ratio}
        out[kind] = rows
    return out


def main():
    # Baseline coefficients from linear regression
    from sklearn.linear_model import LinearRegression
    print("="*60)
    print("BASELINE variable effects (linear regression on E[transfer])")
    print("="*60)
    for name, path in [("Llama-3.1-8B-Instruct","baseline_llama.jsonl"),
                       ("DeepSeek-R1-Distill-Llama-8B","baseline_deepseek.jsonl")]:
        if not (RES/path).exists(): continue
        trials = [json.loads(l) for l in open(RES/path)]
        X = np.array([[1 if t["gender"]=="male" else 0,
                       1 if t["age"]=="old" else 0,
                       1 if t["instruction"]=="B" else 0,
                       1 if t["meeting"]=="meeting" else 0] for t in trials])
        y = np.array([t["E_transfer"] for t in trials])
        reg = LinearRegression().fit(X, y)
        print(f"\n[{name}] intercept={reg.intercept_:+.3f}  R^2={reg.score(X,y):.3f}")
        for n, c in zip(["male","old","instrB","meeting"], reg.coef_):
            print(f"    {n:8s} coef={c:+.3f}")
        print(f"    argmax mean = {np.mean([t['argmax'] for t in trials]):.2f}")
        print(f"    E[transfer] mean = {y.mean():.3f} (fixed endowment $20, fair-split $10)")

    print("\n" + "="*60)
    print("CLAIM 1: linear-probe accuracy on baseline 1000 dictator prompts")
    print("="*60)
    for name, path in [("Llama","probe_baseline_accs.json"),
                       ("DeepSeek","probe_baseline_deepseek.json")]:
        if not (RES/path).exists(): continue
        accs = json.load(open(RES/path))
        for v, a in accs.items():
            if isinstance(a, dict):  # deepseek format {layer: acc}
                vals = list(a.values())
            else:
                vals = a
            print(f"  [{name}] {v:12s} layer0={vals[0]:.2f}  best={max(vals):.2f}  final={vals[-1]:.2f}")

    print("\n" + "="*60)
    print("CLAIM 3+4: causal intervention")
    print("="*60)
    for name, path in [("Llama","intervention_llama.json"),
                       ("DeepSeek","intervention_deepseek.json")]:
        p = RES/path
        if not p.exists(): print(f"  {name}: not yet available"); continue
        intv = json.load(open(p))
        print(f"\n### [{name}] target layer = {intv['target_layer']}")
        base, tables = summarise_intervention(intv, alpha_ref=3.0)
        print(f"  baseline effects: {json.dumps(base, indent=2)[:400]}")
        for kind in ["raw", "pure"]:
            print(f"\n  {kind.upper()} directions, |alpha|=3 delta from baseline:")
            print(f"    {'condition':<22s} {'d_gender':>9s} {'d_age':>7s} {'d_instr':>8s} {'d_meet':>7s} {'d_mean':>7s}")
            for cond, row in tables[kind].items():
                print(f"    {cond:<22s} {row['gender']:+.3f}    {row['age']:+.3f}   {row['instruction']:+.3f}    {row['meeting']:+.3f}    {row['mean']:+.3f}")
        # specificity ratio
        sr = spec_ratio(intv, alpha_ref=3.0)
        print(f"\n  Specificity ratio |d_target| / max|d_other| at |alpha|=3:")
        for kind in ["raw", "pure"]:
            print(f"    ---{kind}---")
            for cond, r in sr[kind].items():
                print(f"    {cond:<22s} target_delta={r['delta_target']:+.3f} max_other={r['max_delta_other']:.3f} ratio={r['spec_ratio']:.2f}")


if __name__ == "__main__":
    main()
