"""Dose-response inference for C1 (primary): permutation test on the isotonic/linear trend of
helix_all vs coefficient on the TEST split, effect size (Cliff's delta, Hedges g) with bootstrap
CIs, FDR (Benjamini-Hochberg) across the pre-chosen site family. Post-processing only."""
import os, sys, json, glob
import numpy as np

def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    gt = sum((x > b).sum() for x in a); lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b) + 1e-9)

def hedges_g(a, b):
    a = np.asarray(a); b = np.asarray(b)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2) + 1e-12)
    d = (a.mean() - b.mean()) / (sp + 1e-9)
    return d * (1 - 3 / (4 * (na + nb) - 9))

def trend_perm_test(coefs, values, n_perm=10000, seed=0):
    """Slope of helix vs coef; permutation p-value (one-sided positive)."""
    coefs = np.asarray(coefs, float); values = np.asarray(values, float)
    c = coefs - coefs.mean()
    obs = (c * (values - values.mean())).sum() / (c @ c + 1e-12)
    rng = np.random.default_rng(seed)
    ge = 1
    for _ in range(n_perm):
        p = rng.permutation(values)
        s = (c * (p - p.mean())).sum() / (c @ c + 1e-12)
        if s >= obs: ge += 1
    return float(obs), ge / (n_perm + 1)

def bh_fdr(pvals):
    p = np.asarray(pvals); order = np.argsort(p); m = len(p)
    adj = np.empty(m); prev = 1.0
    for rank, idx in enumerate(reversed(order)):
        i = m - rank
        prev = min(prev, p[idx] * m / i); adj[idx] = prev
    return adj

def main():
    m3_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results", "m3")
    m3_dir = os.path.abspath(m3_dir)
    out = {}
    site_results = []
    pvals = []
    # aggregate per-site across seed-sharded files (s{site}_caa*.json)
    by_site = {}
    for f in sorted(glob.glob(os.path.join(m3_dir, "s*_caa*.json"))):
        d = json.load(open(f))
        by_site.setdefault(d["site"], []).extend(d["rows"])
    for site in sorted(by_site):
        rows = by_site[site]
        d = {"site": site}
        coefs = [r["coef"] for r in rows]
        helix = [r["helix_all"] for r in rows]
        slope, p = trend_perm_test(coefs, helix)
        # effect size: highest coef vs coef 0 per-generation pools
        by = {}
        for r in rows: by.setdefault(r["coef"], []).extend(r.get("per_gen_helix", []))
        cmax = max(by); c0 = min(by)
        delta = cliffs_delta(by[cmax], by[c0]) if by.get(cmax) and by.get(c0) else float("nan")
        g = hedges_g(by[cmax], by[c0]) if by.get(cmax) and by.get(c0) else float("nan")
        site_results.append(dict(site=d["site"], slope=slope, p_raw=p,
                                 cliffs_delta=float(delta), hedges_g=float(g),
                                 helix_by_coef={str(k): float(np.mean(v)) for k, v in by.items()}))
        pvals.append(p)
    if pvals:
        adj = bh_fdr(pvals)
        for r, a in zip(site_results, adj): r["p_fdr"] = float(a)
    out["sites"] = site_results
    # winning setting = site with best FDR-significant positive slope, coef maximizing helix
    sig = [r for r in site_results if r.get("p_fdr", 1) < 0.05 and r["slope"] > 0]
    if sig:
        best = max(sig, key=lambda r: r["slope"])
        best_coef = max(best["helix_by_coef"], key=lambda k: best["helix_by_coef"][k])
        out["winning"] = dict(site=best["site"], coef=float(best_coef),
                              p_fdr=best["p_fdr"], slope=best["slope"])
        json.dump(out["winning"], open(os.path.join(m3_dir, "winning.json"), "w"))
    out["c1_supported"] = bool(sig)
    json.dump(out, open(os.path.join(m3_dir, "dose_response_stats.json"), "w"), indent=2)
    print(json.dumps(out, indent=2, default=float))

if __name__ == "__main__":
    main()
