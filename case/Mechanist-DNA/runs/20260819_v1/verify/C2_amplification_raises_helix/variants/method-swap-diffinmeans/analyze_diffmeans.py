"""Analyze the diff-in-means method-swap arms.

Stage A (dev sweep, block D): compute the dose-response, pick alpha*_dm by argmax
conditional predicted-%-helix subject to the SAME validity floor as the main experiment
(valid_rate >= baseline-0.10 AND ppl_median <= baseline 95th pct).
Stage B (high-power confirm, block H): if confirm arms exist, two-sample bootstrap of
Delta predicted-%-helix (alpha*_dm vs alpha=0) with p-value.

Writes result.json (consumed by /result-to-claim + Phase 9 audit).
"""
import os, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name):
    p = os.path.join(HERE, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def conds(arm):
    """per-seq conditional helix_all (QC-valid, folded)."""
    return [p["helix_all"] for p in arm["per_seq"]
            if p.get("qc_ok") and ("helix_all" in p) and p["helix_all"] == p["helix_all"]]


def itts(arm):
    out = []
    for p in arm["per_seq"]:
        if p.get("qc_ok") and ("helix_all" in p) and p["helix_all"] == p["helix_all"]:
            out.append(p["helix_all"])
        else:
            out.append(0.0)
    return out


N_MIN_VALID = 150   # minimum QC-valid count for an arm to be an alpha* candidate


def diff_stats(a, b, n_boot=5000, n_perm=10000, seed=1):
    """Two-sample bootstrap CI + two-sample permutation p (two-sided, +1 correction).
    Unpaired two-sample matches the main experiment's stated conditional test
    (seed pairing gives ~no variance reduction once steering diverges the sequences)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    delta = float(b.mean() - a.mean())
    d = np.empty(n_boot)
    for i in range(n_boot):
        d[i] = rng.choice(b, len(b), replace=True).mean() - rng.choice(a, len(a), replace=True).mean()
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    obs = abs(delta); pool = np.concatenate([a, b]); na = len(a); n = len(pool); ge = 0
    for _ in range(n_perm):
        idx = rng.permutation(n)
        if abs(pool[idx[na:]].mean() - pool[idx[:na]].mean()) >= obs:
            ge += 1
    p = (1 + ge) / (1 + n_perm)
    return delta, lo, hi, p


def main():
    vdm = np.load(os.path.join(HERE, "vdm.npz"))
    dev_alphas = [0, 0.5, 1, 2, 4]
    rows = []
    base = load("dm_dev_a0")
    base_s = base["summary"]
    base_valid = base_s["valid_rate"]; base_ppl95 = base_s["ppl_p95"]
    base_cond = base_s["helix_cond_mean"]
    for a in dev_alphas:
        tag = "dm_dev_a" + (str(a) if a != 0 else "0")
        arm = load(tag)
        if arm is None:
            continue
        s = arm["summary"]
        hc = s["helix_cond_mean"]
        passes_floor = ((s["valid_rate"] >= base_valid - 0.10) and (s["ppl_median"] <= base_ppl95)
                        and (s["n_valid"] >= N_MIN_VALID) and (hc == hc))   # finite + N_min
        rows.append(dict(alpha=a, valid_rate=s["valid_rate"], helix_cond=hc,
                         helix_itt=s["helix_itt_mean"], ppl_median=s["ppl_median"],
                         n_valid=s["n_valid"], passes_validity_floor=bool(passes_floor),
                         d_cond=hc - base_cond))
    # alpha* = argmax conditional helix among alpha>0 passing the floor; deterministic
    # smallest-alpha tie-break; fallback alpha*=0 if no candidate passes (informative null).
    cand = [r for r in rows if r["alpha"] > 0 and r["passes_validity_floor"]]
    if cand:
        best = max(r["helix_cond"] for r in cand)
        alpha_star = min(r["alpha"] for r in cand if abs(r["helix_cond"] - best) < 1e-12)
    else:
        alpha_star = 0    # no positive alpha passes the validity floor -> null (informative)

    res = dict(variant_tag="method-swap-diffinmeans", dimension="method",
               direction_diagnostics=dict(
                   cosine_vdm_vS=float(vdm["cosine_vdm_vS"]),
                   norm=float(np.linalg.norm(vdm["v_dm"])),
                   norm_S=float(vdm["norm_S"]),
                   n_helix_codons=int(vdm["n_helix_codons"]),
                   n_nonhelix_codons=int(vdm["n_nonhelix_codons"]),
                   n_genes=int(vdm["n_genes"])),
               dev_sweep=rows, alpha_star_dm=alpha_star, base_valid=base_valid,
               base_ppl_p95=base_ppl95)

    # Stage B: high-power confirm at alpha*_dm vs alpha=0 (block H)
    hp0 = load("dm_hp_a0")
    hps = load("dm_hp_astar") if (alpha_star and alpha_star > 0) else None
    if hp0 is not None and hps is not None:
        c0, cs = conds(hp0), conds(hps)
        i0, is_ = itts(hp0), itts(hps)
        d_cond, lo_c, hi_c, p_c = diff_stats(c0, cs)
        d_itt, lo_i, hi_i, p_i = diff_stats(i0, is_)
        res["highpower"] = dict(
            alpha_star_dm=alpha_star,
            n_valid_baseline=len(c0), n_valid_steered=len(cs),
            baseline_helix_cond=float(np.mean(c0)), steered_helix_cond=float(np.mean(cs)),
            d_cond=d_cond, d_cond_ci=[lo_c, hi_c], p_cond=p_c,
            baseline_helix_itt=float(np.mean(i0)), steered_helix_itt=float(np.mean(is_)),
            d_itt=d_itt, d_itt_ci=[lo_i, hi_i], p_itt=p_i,
            base_valid_rate=hp0["summary"]["valid_rate"], steered_valid_rate=hps["summary"]["valid_rate"],
            base_ppl_median=hp0["summary"]["ppl_median"], steered_ppl_median=hps["summary"]["ppl_median"])
    json.dump(res, open(os.path.join(HERE, "result.json"), "w"), indent=2)
    print("[analyze] alpha*_dm =", alpha_star, " cosine(v_dm,v_S)=", res["direction_diagnostics"]["cosine_vdm_vS"])
    for r in rows:
        print(f"  a={r['alpha']}: helix_cond={r['helix_cond']:.3f} (d={r['d_cond']:+.3f}) "
              f"valid={r['valid_rate']:.2f} ppl={r['ppl_median']:.2f} floor={r['passes_validity_floor']}")
    if "highpower" in res:
        h = res["highpower"]
        print(f"[analyze] HIGH-POWER a*={alpha_star}: cond {h['baseline_helix_cond']:.3f}->{h['steered_helix_cond']:.3f} "
              f"d={h['d_cond']:+.4f} CI{[round(x,4) for x in h['d_cond_ci']]} p={h['p_cond']:.3f}; "
              f"itt d={h['d_itt']:+.4f} p={h['p_itt']:.3f}")


if __name__ == "__main__":
    main()
