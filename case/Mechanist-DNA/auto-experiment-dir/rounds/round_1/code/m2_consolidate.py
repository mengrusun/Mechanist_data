"""Consolidate the M2 dose-response sweep: per-dose helix mean (pooled over seeds), Spearman trend,
optimal alpha*, per-dose Mann-Whitney vs alpha=0, quality metrics. -> results/m2_dose_response_curve.json"""
import os, sys, json, glob
import numpy as np
from scipy import stats

RES = "/data/wanghaoxiong/intergene_mechanist_v6/results"


def main():
    runs = {}
    for f in glob.glob(os.path.join(RES, "m2_a*_s*.json")):
        d = json.load(open(f)); c = d["config"]
        runs.setdefault(c["alpha"], []).append(d)
    alphas = sorted(runs.keys())
    per_dose = {}
    pooled_samples = {}
    for a in alphas:
        helix_means, valid_orf, gated_pass, plddt, sheet_means = [], [], [], [], []
        samples = []
        for d in runs[a]:
            h = d["helix_hgi"]
            if h.get("helix_hgi_mean") is not None:
                helix_means.append(h["helix_hgi_mean"])
            valid_orf.append(h["valid_orf_rate"]); gated_pass.append(h["gated_pass_rate"])
            if h.get("plddt_mean"): plddt.append(h["plddt_mean"])
            s = d["sheet"]
            if s.get("sheet_mean") is not None: sheet_means.append(s["sheet_mean"])
            samples += [v for v in d.get("per_sample_helix_hgi", []) if v is not None]
        per_dose[a] = {
            "helix_hgi_mean": float(np.mean(helix_means)) if helix_means else None,
            "helix_hgi_sem": float(np.std(helix_means) / np.sqrt(len(helix_means))) if len(helix_means) > 1 else None,
            "n_seeds": len(runs[a]), "n_folded_total": len(samples),
            "valid_orf_rate": float(np.mean(valid_orf)), "gated_pass_rate": float(np.mean(gated_pass)),
            "plddt_mean": float(np.mean(plddt)) if plddt else None,
            "sheet_mean": float(np.mean(sheet_means)) if sheet_means else None,
        }
        pooled_samples[a] = samples

    # Spearman trend over the swept range (alpha vs mean helix), using per-seed points
    xs, ys = [], []
    for a in alphas:
        for d in runs[a]:
            h = d["helix_hgi"].get("helix_hgi_mean")
            if h is not None:
                xs.append(a); ys.append(h)
    spearman = stats.spearmanr(xs, ys) if len(set(xs)) > 1 else (None, None)

    # optimal alpha*: max helix among doses whose quality (valid_orf, plddt) stays within tolerance of alpha=0
    base = per_dose.get(0, {})
    base_orf = base.get("valid_orf_rate", 1.0)
    def quality_ok(a):
        pd = per_dose[a]
        return pd["valid_orf_rate"] >= 0.5 * base_orf and (pd["plddt_mean"] is None or pd["plddt_mean"] >= 45)
    pos_alphas = [a for a in alphas if a > 0 and per_dose[a]["helix_hgi_mean"] is not None]
    alpha_star = None
    if pos_alphas:
        ok_alphas = [a for a in pos_alphas if quality_ok(a)]
        pool = ok_alphas or pos_alphas
        alpha_star = max(pool, key=lambda a: per_dose[a]["helix_hgi_mean"])

    # per-dose Mann-Whitney vs alpha=0 (pooled samples)
    mw = {}
    if 0 in pooled_samples and pooled_samples[0]:
        for a in alphas:
            if a == 0 or not pooled_samples[a]:
                continue
            try:
                u, p = stats.mannwhitneyu(pooled_samples[a], pooled_samples[0], alternative="two-sided")
                mw[a] = {"p": float(p), "delta_vs_0": float(np.mean(pooled_samples[a]) - np.mean(pooled_samples[0]))}
            except Exception:
                pass

    result = {
        "alphas": alphas, "per_dose": per_dose,
        "spearman_rho": float(spearman[0]) if spearman[0] is not None else None,
        "spearman_p": float(spearman[1]) if spearman[1] is not None else None,
        "alpha_star": alpha_star,
        "helix_at_alpha_star": per_dose[alpha_star]["helix_hgi_mean"] if alpha_star is not None else None,
        "helix_at_0": per_dose.get(0, {}).get("helix_hgi_mean"),
        "effect_size_delta": (per_dose[alpha_star]["helix_hgi_mean"] - per_dose[0]["helix_hgi_mean"])
                             if (alpha_star is not None and per_dose.get(0, {}).get("helix_hgi_mean") is not None) else None,
        "mannwhitney_vs_0": mw,
        "n_configs": sum(len(v) for v in runs.values()),
    }
    json.dump(result, open(os.path.join(RES, "m2_dose_response_curve.json"), "w"), indent=2)
    print("=== M2 dose-response ===")
    for a in alphas:
        pd = per_dose[a]
        print(f"  alpha={a:>4}: helix={pd['helix_hgi_mean']} sem={pd['helix_hgi_sem']} "
              f"valid_orf={pd['valid_orf_rate']:.2f} plddt={pd['plddt_mean']} n={pd['n_folded_total']}")
    print(f"Spearman rho={result['spearman_rho']} p={result['spearman_p']}")
    print(f"alpha*={alpha_star} helix@a*={result['helix_at_alpha_star']} helix@0={result['helix_at_0']} "
          f"effect={result['effect_size_delta']}")


if __name__ == "__main__":
    main()
