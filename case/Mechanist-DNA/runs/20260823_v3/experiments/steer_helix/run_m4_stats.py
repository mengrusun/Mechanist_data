"""M4 stats (CPU, no GPU): formal C1 / C2 verdicts from M3 sweep + M4 winning-setting conditions.

C1 = dose-response trend (permutation Spearman on coef, primary all-generations helix endpoint,
     TEST split, s28 CAA) + effect size (Cliff's delta, Hedges g) + bootstrap CI on helix gain +
     BH-FDR across sites x coefs + specificity (matched_control & sham show no gain).
C2 = validity non-inferiority at the winning setting (one-sided 95% bootstrap LB of validity delta
     vs -margin) + off-target deltas (sheet, GC, Evo2 NLL, length, aa-composition).
"""
import os, sys, json, argparse
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(__file__))
import common as C

RNG = np.random.default_rng(12345)


def spearman(x, y):
    return stats.spearmanr(x, y).statistic


def perm_trend_test(coefs, helix, n_perm=20000):
    """One-sided permutation test for a positive dose-response (Spearman rho of helix on coef)."""
    x = np.asarray(coefs, float); y = np.asarray(helix, float)
    obs = spearman(x, y)
    ge = 1
    for _ in range(n_perm):
        if spearman(x, RNG.permutation(y)) >= obs:
            ge += 1
    return float(obs), float(ge / (n_perm + 1))


def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    gt = sum((a[:, None] > b[None, :]).sum(1)) if a.size * b.size < 4_000_000 else None
    if gt is None:  # chunked fallback
        gt = lt = 0
        for v in a:
            gt += (v > b).sum(); lt += (v < b).sum()
        return float((gt - lt) / (a.size * b.size))
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (a.size * b.size))


def hedges_g(a, b):
    a = np.asarray(a); b = np.asarray(b)
    na, nb = a.size, b.size
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = (a.mean() - b.mean()) / (sp + 1e-12)
    J = 1 - 3 / (4 * (na + nb) - 9)
    return float(d * J)


def boot_ci_mean_diff(a, b, n_boot=20000, alpha=0.05):
    a = np.asarray(a); b = np.asarray(b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = a[RNG.integers(0, a.size, a.size)].mean() - b[RNG.integers(0, b.size, b.size)].mean()
    return float(np.percentile(diffs, 100 * alpha / 2)), float(np.percentile(diffs, 100 * (1 - alpha / 2)))


def boot_ni_validity(valid_win, valid_base, margin, n_boot=20000):
    """One-sided 95% percentile-bootstrap lower bound of (p_win - p_base). NI iff LB > -margin."""
    vw = np.asarray(valid_win); vb = np.asarray(valid_base)
    d = np.empty(n_boot)
    for i in range(n_boot):
        d[i] = vw[RNG.integers(0, vw.size, vw.size)].mean() - vb[RNG.integers(0, vb.size, vb.size)].mean()
    lb = float(np.percentile(d, 5.0))
    return dict(delta=float(vw.mean() - vb.mean()), lb_one_sided95=lb,
                ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                margin=margin, non_inferior=bool(lb > -margin))


def bh_fdr(pvals):
    p = np.asarray(pvals); n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        adj[i] = prev
    return adj.tolist()


def sweep_arrays(rows, site):
    """Return (coef_per_gen, helix_per_gen) for one site's CAA sweep (all coefs, all seeds)."""
    cc, hh = [], []
    for r in rows:
        for h in r["per_gen_helix"]:
            cc.append(r["coef"]); hh.append(h)
    return cc, hh


def per_coef_helix(rows, coef):
    out = []
    for r in rows:
        if r["coef"] == coef:
            out.extend(r["per_gen_helix"])
    return out


def valid_array_from_m3(rows, coef):
    arr = []
    for r in rows:
        if r["coef"] == coef:
            k = int(round(r["validity_rate"] * r["n"]))
            arr.extend([1] * k + [0] * (r["n"] - k))
    return np.array(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=int, default=28)
    ap.add_argument("--baseline_coef", type=float, default=0.0)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--m3_dir", default=os.path.join(C.PROJECT, "results/m3"))
    ap.add_argument("--m4_dir", default=os.path.join(C.PROJECT, "results/m4"))
    args = ap.parse_args()

    win = json.load(open(os.path.join(args.m3_dir, "winning.json")))
    win_coef = float(win["coef"])
    s28 = json.load(open(os.path.join(args.m3_dir, f"s{args.site}_caa.json")))["rows"]

    # ---- C1: dose-response trend on s28 CAA, TEST split ----
    cc, hh = sweep_arrays(s28, args.site)
    rho, p_trend = perm_trend_test(cc, hh)
    base_h = per_coef_helix(s28, args.baseline_coef)
    win_h = per_coef_helix(s28, win_coef)
    c1 = dict(site=args.site, winning_coef=win_coef,
              trend_spearman_rho=rho, trend_perm_p=p_trend,
              cliffs_delta_win_vs_base=cliffs_delta(win_h, base_h),
              hedges_g_win_vs_base=hedges_g(win_h, base_h),
              helix_baseline=float(np.mean(base_h)), helix_winning=float(np.mean(win_h)),
              helix_gain=float(np.mean(win_h) - np.mean(base_h)),
              helix_gain_ci95=boot_ci_mean_diff(win_h, base_h))

    # ---- BH-FDR across sites x nonzero coefs (Mann-Whitney one-sided greater vs baseline) ----
    tests = []
    for site, fname in [(28, "s28_caa.json"), (30, "s30_caa.json")]:
        rows = json.load(open(os.path.join(args.m3_dir, fname)))["rows"]
        bh = per_coef_helix(rows, args.baseline_coef)
        for coef in sorted({r["coef"] for r in rows if r["coef"] != args.baseline_coef}):
            ch = per_coef_helix(rows, coef)
            p = stats.mannwhitneyu(ch, bh, alternative="greater").pvalue
            tests.append(dict(site=site, coef=coef, mwu_p=float(p)))
    adj = bh_fdr([t["mwu_p"] for t in tests])
    for t, a in zip(tests, adj):
        t["bh_fdr_q"] = float(a)
    win_test = next(t for t in tests if t["site"] == args.site and t["coef"] == win_coef)

    # ---- specificity: matched_control & sham vs M4 baseline ----
    def load_raw(cond):
        return json.load(open(os.path.join(args.m4_dir, f"raw_{cond}.json")))["per_gen"]
    raw = {c: load_raw(c) for c in ["baseline", "caa_win", "matched_control", "sham"]}
    def helix_of(cond): return [g["helix_frac"] for g in raw[cond]]
    def valid_of(cond): return np.array([1 if g["valid"] else 0 for g in raw[cond]])
    b_h = helix_of("baseline")
    spec = {}
    for cond in ["caa_win", "matched_control", "sham"]:
        ch = helix_of(cond)
        spec[cond] = dict(mean_helix=float(np.mean(ch)),
                          gain_vs_baseline=float(np.mean(ch) - np.mean(b_h)),
                          mwu_p_greater=float(stats.mannwhitneyu(ch, b_h, alternative="greater").pvalue),
                          cliffs_delta=cliffs_delta(ch, b_h))

    # ---- C2: validity non-inferiority at winning (use M4 baseline vs caa_win per-gen valid) ----
    c2 = boot_ni_validity(valid_of("caa_win"), valid_of("baseline"), args.margin)
    # cross-check against M3 reconstructed validity (robustness)
    c2["m3_crosscheck"] = boot_ni_validity(valid_array_from_m3(s28, win_coef),
                                           valid_array_from_m3(s28, args.baseline_coef), args.margin)
    # off-target deltas (winning caa_win vs baseline), M4 per-gen
    def mean_field(cond, f): return float(np.mean([g[f] for g in raw[cond]]))
    offt = {}
    for f in ["sheet_frac", "gc", "mean_nll", "prot_len", "aa_helixfav", "plddt"]:
        offt[f] = dict(baseline=mean_field("baseline", f), caa_win=mean_field("caa_win", f),
                       delta=mean_field("caa_win", f) - mean_field("baseline", f))

    # ---- frontier (from M3 s28) ----
    by = {}
    for r in s28:
        by.setdefault(r["coef"], []).append(r)
    frontier = []
    for coef in sorted(by):
        frontier.append(dict(coef=coef,
                             helix_all=float(np.mean([r["helix_all"] for r in by[coef]])),
                             validity_rate=float(np.mean([r["validity_rate"] for r in by[coef]])),
                             helix_gain=float(np.mean([r["helix_all"] for r in by[coef]]) - np.mean(base_h))))
    import pandas as pd
    pd.DataFrame(frontier).to_parquet(os.path.join(args.m4_dir, "validity_frontier.parquet"))

    c1_supported = bool(p_trend < 0.05 and c1["cliffs_delta_win_vs_base"] > 0
                        and win_test["bh_fdr_q"] < 0.05
                        and spec["matched_control"]["mwu_p_greater"] > 0.05
                        and spec["sham"]["mwu_p_greater"] > 0.05)
    out = dict(site=args.site, winning_coef=win_coef, margin=args.margin,
               C1=dict(supported=c1_supported, **c1, bh_fdr_at_winning=win_test),
               specificity=spec,
               C2=dict(supported=bool(c2["non_inferior"]), **c2, off_target=offt),
               joint_C1_and_C2=bool(c1_supported and c2["non_inferior"]),
               bh_fdr_table=tests, validity_frontier=frontier,
               winning_selection=win.get("eligibility"))
    C.save_json(out, os.path.join(args.m4_dir, "specificity.json"))

    print("=== M4 SUMMARY ===")
    print(json.dumps(dict(winning_coef=win_coef, C1_supported=c1_supported,
                          trend_rho=round(rho, 4), trend_p=p_trend,
                          cliffs_delta=round(c1["cliffs_delta_win_vs_base"], 4),
                          hedges_g=round(c1["hedges_g_win_vs_base"], 4),
                          helix_baseline=round(c1["helix_baseline"], 4),
                          helix_winning=round(c1["helix_winning"], 4),
                          helix_gain_ci=c1["helix_gain_ci95"],
                          bh_fdr_q_winning=round(win_test["bh_fdr_q"], 5),
                          matched_gain=round(spec["matched_control"]["gain_vs_baseline"], 4),
                          matched_p=round(spec["matched_control"]["mwu_p_greater"], 4),
                          sham_gain=round(spec["sham"]["gain_vs_baseline"], 4),
                          sham_p=round(spec["sham"]["mwu_p_greater"], 4),
                          C2_delta=round(c2["delta"], 4), C2_lb=round(c2["lb_one_sided95"], 4),
                          C2_non_inferior=c2["non_inferior"],
                          joint=bool(c1_supported and c2["non_inferior"])), indent=2))
    print("=== off-target ===")
    print(json.dumps({k: round(v["delta"], 4) for k, v in offt.items()}, indent=2))


if __name__ == "__main__":
    main()
