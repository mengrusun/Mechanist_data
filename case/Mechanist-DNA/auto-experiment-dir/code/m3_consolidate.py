"""Consolidate M3 specificity: double dissociation (helix rise for alpha_helix_S but not matched/beta;
sheet rise for beta but not S) + quality guardrail. -> results/m3_specificity_summary.json"""
import os, sys, json, glob
import numpy as np
from scipy import stats

RES = "/data/wanghaoxiong/intergene_mechanist_v6/results"
KINDS = ["alpha_helix_S", "matched_control", "beta_sheet_offtarget"]


def main():
    runs = {}
    for f in glob.glob(os.path.join(RES, "m3_*_a*_s*.json")):
        d = json.load(open(f)); c = d["config"]
        runs.setdefault((c["feature_kind"], c["alpha"]), []).append(d)
    alphas = sorted(set(k[1] for k in runs))
    table = {}
    pooled = {}  # (kind, alpha) -> {"helix":[...], "sheet":[...]}
    for kind in KINDS:
        table[kind] = {}
        for a in alphas:
            ds = runs.get((kind, a), [])
            if not ds:
                continue
            hmeans = [d["helix_hgi"]["helix_hgi_mean"] for d in ds if d["helix_hgi"].get("helix_hgi_mean") is not None]
            smeans = [d["sheet"]["sheet_mean"] for d in ds if d["sheet"].get("sheet_mean") is not None]
            orf = [d["helix_hgi"]["valid_orf_rate"] for d in ds]
            plddt = [d["helix_hgi"].get("plddt_mean") for d in ds if d["helix_hgi"].get("plddt_mean")]
            helix_s = [x["helix_hgi"] for d in ds for x in d.get("per_sample", []) if x.get("helix_hgi") is not None]
            sheet_s = [x["sheet"] for d in ds for x in d.get("per_sample", []) if x.get("sheet") is not None]
            pooled[(kind, a)] = {"helix": helix_s, "sheet": sheet_s}
            table[kind][a] = {
                "helix_mean": float(np.mean(hmeans)) if hmeans else None,
                "sheet_mean": float(np.mean(smeans)) if smeans else None,
                "valid_orf_rate": float(np.mean(orf)) if orf else None,
                "plddt_mean": float(np.mean(plddt)) if plddt else None,
                "n_seeds": len(ds),
            }

    # double dissociation at the top shared dose
    top = max(alphas) if alphas else None
    dd = {}
    if top is not None:
        def delta(kind, metric):
            hi = table.get(kind, {}).get(top, {}).get(metric)
            lo = table.get(kind, {}).get(0, {}).get(metric)
            return (hi - lo) if (hi is not None and lo is not None) else None
        dd = {
            "dose": top,
            "helix_delta_S": delta("alpha_helix_S", "helix_mean"),
            "helix_delta_matched": delta("matched_control", "helix_mean"),
            "helix_delta_beta": delta("beta_sheet_offtarget", "helix_mean"),
            "sheet_delta_S": delta("alpha_helix_S", "sheet_mean"),
            "sheet_delta_beta": delta("beta_sheet_offtarget", "sheet_mean"),
        }
        # significance: S helix rise vs matched-control helix rise (Mann-Whitney on pooled samples at top dose)
        try:
            s_hi = pooled[("alpha_helix_S", top)]["helix"]; m_hi = pooled[("matched_control", top)]["helix"]
            dd["S_vs_matched_helix_p"] = float(stats.mannwhitneyu(s_hi, m_hi, alternative="greater")[1]) if s_hi and m_hi else None
        except Exception:
            dd["S_vs_matched_helix_p"] = None
        try:
            b_hi = pooled[("beta_sheet_offtarget", top)]["sheet"]; s_sh = pooled[("alpha_helix_S", top)]["sheet"]
            dd["beta_vs_S_sheet_p"] = float(stats.mannwhitneyu(b_hi, s_sh, alternative="greater")[1]) if b_hi and s_sh else None
        except Exception:
            dd["beta_vs_S_sheet_p"] = None

    result = {"alphas": alphas, "table": table, "double_dissociation": dd}
    json.dump(result, open(os.path.join(RES, "m3_specificity_summary.json"), "w"), indent=2)
    print("=== M3 specificity ===")
    for kind in KINDS:
        print(f"  {kind}:")
        for a in alphas:
            t = table.get(kind, {}).get(a)
            if t:
                print(f"    a={a:>3}: helix={t['helix_mean']} sheet={t['sheet_mean']} orf={t['valid_orf_rate']} plddt={t['plddt_mean']}")
    print("double dissociation:", json.dumps(dd, indent=2))


if __name__ == "__main__":
    main()
