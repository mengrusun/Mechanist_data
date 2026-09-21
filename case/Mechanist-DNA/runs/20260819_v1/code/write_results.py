"""Assemble a compact machine-readable summary of all milestones into
results/summary.json, for the human-written EXPERIMENT_RESULTS.md to draw on."""
import os, sys, json, glob
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None

def main():
    out = {}
    m1 = load(os.path.join(ROOT, "results/m1_features.json"))
    if m1:
        out["C1"] = dict(
            selection_path=m1.get("selection_path"),
            n_candidates=m1.get("n_candidates"),
            chosen_K=m1.get("chosen_K"),
            S=m1.get("S"),
            c1=m1.get("c1"),
            top_features=(m1.get("features") or [])[:8],
            helix_frac=m1.get("helix_frac"),
            ctrl_helix_auroc_bar=m1.get("ctrl_helix_auroc_bar"),
        )
    probe = load(os.path.join(ROOT, "results/ss_probe_meta.json"))
    if probe:
        out["readout"] = dict(tool="esm2_probe", **probe)
    m2 = load(os.path.join(ROOT, "results/m2_analysis.json"))
    if m2:
        out["M2"] = dict(alpha_star=m2.get("alpha_star"),
                         c2=m2.get("c2"), c3=m2.get("c3"),
                         base_valid_rate=m2.get("base_valid_rate"),
                         rows=m2.get("rows"))
    mc = load(os.path.join(ROOT, "results/mctrl_analysis.json"))
    if mc:
        out["MCTRL"] = mc
    m3 = load(os.path.join(ROOT, "results/m3_analysis.json"))
    if m3:
        out["M3"] = dict(alpha_star=m3.get("alpha_star"),
                         c2=m3.get("c2"), c3=m3.get("c3"), rows=m3.get("rows"))
    with open(os.path.join(ROOT, "results/summary.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2)[:4000])

if __name__ == "__main__":
    main()
