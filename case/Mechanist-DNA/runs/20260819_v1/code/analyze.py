"""Analysis for M2 (dose-response + alpha* selection), M-CTRL (specificity),
and M3 (held-out confirmation). Statistics use the generated sequence as the
resampling unit; baseline and steered arms share seeds (paired)."""
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ITT_FAIL = 0.0

def load_arm(path):
    with open(path) as fh:
        return json.load(fh)

def seq_helix_map(arm, mode="cond"):
    """seed -> helix value. cond = valid-only; itt = all (fail->0)."""
    out = {}
    for p in arm["per_seq"]:
        h = p.get("helix_all")
        valid = p.get("qc_ok") and (h is not None) and (h == h)
        if mode == "cond":
            if valid:
                out[p["seed"]] = h
        else:  # itt
            out[p["seed"]] = h if valid else ITT_FAIL
    return out

def paired_diff_boot(base_arm, steer_arm, mode="cond", n_boot=2000, seed=0):
    """Paired bootstrap over shared seeds of mean(steer)-mean(base). Returns
    (delta, ci_lo, ci_hi, p_two_sided, n_pairs)."""
    rng = np.random.default_rng(seed)
    b = seq_helix_map(base_arm, mode); s = seq_helix_map(steer_arm, mode)
    common = sorted(set(b) & set(s))
    if len(common) < 5:
        return float("nan"), float("nan"), float("nan"), float("nan"), len(common)
    db = np.array([s[k] for k in common]) - np.array([b[k] for k in common])
    delta = float(db.mean())
    boots = np.array([db[rng.integers(0, len(db), len(db))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # two-sided p: fraction of bootstrap crossing 0
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    p = min(1.0, p)
    return delta, float(lo), float(hi), float(p), len(common)

def unpaired_diff_boot(base_arm, steer_arm, mode="cond", n_boot=4000, seed=0):
    """Two-sample (unpaired) bootstrap of mean(steer)-mean(base). Appropriate for
    the conditional-on-valid comparison: the two valid subsets are independent
    samples and RNG-seed pairing yields no variance reduction under steering.
    Returns (delta, ci_lo, ci_hi, p_two_sided, n_base, n_steer)."""
    rng = np.random.default_rng(seed)
    b = np.array(list(seq_helix_map(base_arm, mode).values()))
    s = np.array(list(seq_helix_map(steer_arm, mode).values()))
    if len(b) < 5 or len(s) < 5:
        return float("nan"), float("nan"), float("nan"), float("nan"), len(b), len(s)
    delta = float(s.mean() - b.mean())
    boots = np.empty(n_boot)
    for i in range(n_boot):
        bb = b[rng.integers(0, len(b), len(b))].mean()
        ss = s[rng.integers(0, len(s), len(s))].mean()
        boots[i] = ss - bb
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return delta, float(lo), float(hi), float(p), len(b), len(s)

def unpaired_mean_boot(arm, mode="cond", n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    m = seq_helix_map(arm, mode)
    vals = np.array(list(m.values()))
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.array([vals[rng.integers(0, len(vals), len(vals))].mean()
                      for _ in range(n_boot)])
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def analyze_dose(arm_paths, baseline_name):
    """arm_paths: dict alpha-> path. Returns dose table + alpha* selection."""
    arms = {a: load_arm(p) for a, p in arm_paths.items()}
    base = arms[0.0]
    base_valid_rate = base["summary"]["valid_rate"]
    base_ppl_p95 = base["summary"]["ppl_p95"]
    rows = []
    for a in sorted(arms):
        arm = arms[a]
        s = arm["summary"]
        dcond = paired_diff_boot(base, arm, "cond") if a != 0 else (0.0,0.0,0.0,1.0,s["n_valid"])
        ditt = paired_diff_boot(base, arm, "itt") if a != 0 else (0.0,0.0,0.0,1.0,s["n_gen"])
        ucond = unpaired_diff_boot(base, arm, "cond") if a != 0 else (0.0,0.0,0.0,1.0,0,0)
        uitt = unpaired_diff_boot(base, arm, "itt") if a != 0 else (0.0,0.0,0.0,1.0,0,0)
        rows.append(dict(alpha=a, valid_rate=s["valid_rate"],
                         helix_cond=s["helix_cond_mean"], helix_itt=s["helix_itt_mean"],
                         helix_hi=s["helix_hi_cond_mean"],
                         sheet_cond=s["sheet_cond_mean"], coil_cond=s["coil_cond_mean"],
                         ppl_median=s["ppl_median"], plddt=s["plddt_mean"],
                         gc=s["gc_mean"], prot_len=s["prot_len_mean"], n_valid=s["n_valid"],
                         d_helix_cond=dcond[0], d_cond_ci=[dcond[1], dcond[2]], p_cond=dcond[3],
                         d_helix_itt=ditt[0], d_itt_ci=[ditt[1], ditt[2]], p_itt=ditt[3],
                         d_cond_unpaired=ucond[0], p_cond_unpaired=ucond[3],
                         d_cond_unpaired_ci=[ucond[1], ucond[2]],
                         d_itt_unpaired=uitt[0], p_itt_unpaired=uitt[3]))
    # validity floor: ORF-valid rate >= baseline-10pp AND median ppl <= baseline p95
    def passes_floor(r):
        return (r["valid_rate"] >= base_valid_rate - 0.10) and \
               (r["ppl_median"] <= base_ppl_p95 if base_ppl_p95==base_ppl_p95 else True)
    cand = [r for r in rows if r["alpha"] > 0 and passes_floor(r)]
    # alpha* = argmax conditional helix among floor-passing
    astar = None
    if cand:
        astar = max(cand, key=lambda r: r["helix_cond"])["alpha"]
    return dict(rows=rows, base_valid_rate=base_valid_rate, base_ppl_p95=base_ppl_p95,
                alpha_star=astar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["m2", "m3", "mctrl"], required=True)
    ap.add_argument("--glob", required=True, help="glob of arm result JSONs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha_star", type=float, default=None)
    args = ap.parse_args()
    paths = glob.glob(args.glob)

    if args.mode in ("m2", "m3"):
        arm_paths = {}
        for p in paths:
            arm = load_arm(p)
            if "alpha" not in arm or "per_seq" not in arm:
                continue   # skip analysis/summary files caught by the glob
            arm_paths[float(arm["alpha"])] = p
        res = analyze_dose(arm_paths, baseline_name="a0")
        # verdicts
        rows = res["rows"]
        # C2 primary: conditional (UNPAIRED two-sample) AND ITT both raise helix at p<0.05,
        # without validity collapse. (Paired conditional is under-powered because RNG-seed
        # pairing yields ~no variance reduction under steering; reported alongside.)
        pos = [r for r in rows if r["alpha"] > 0 and r["d_cond_unpaired"] > 0 and
               r["p_cond_unpaired"] < 0.05 and r["d_helix_itt"] > 0 and r["p_itt"] < 0.05 and
               r["valid_rate"] >= res["base_valid_rate"] - 0.10]
        c2 = dict(pass_=bool(pos),
                  best_alpha=(max(pos, key=lambda r: r["d_cond_unpaired"])["alpha"] if pos else None),
                  criterion="unpaired-conditional & paired-ITT both p<0.05")
        # dose-response non-flat & single-peaked/saturating; alpha* identifiable
        helix_by_a = [(r["alpha"], r["helix_cond"]) for r in rows]
        nonflat = (max(r["helix_cond"] for r in rows) - min(r["helix_cond"] for r in rows)) > 0.02
        c3 = dict(pass_=bool(res["alpha_star"] is not None and nonflat),
                  alpha_star=res["alpha_star"], nonflat=bool(nonflat))
        res["c2"] = c2; res["c3"] = c3
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"[analyze:{args.mode}] alpha*={res['alpha_star']} C2={c2['pass_']} C3={c3['pass_']}")
        for r in rows:
            print(f"  a={r['alpha']:>5}: valid={r['valid_rate']:.2f} n={r['n_valid']} "
                  f"helix_cond={r['helix_cond']:.3f} dcond={r['d_cond_unpaired']:+.3f} "
                  f"p_uc={r['p_cond_unpaired']:.3f} (p_pair={r['p_cond']:.2f}) "
                  f"itt_d={r['d_helix_itt']:+.3f} p_itt={r['p_itt']:.3f} ppl={r['ppl_median']:.1f}")

    elif args.mode == "mctrl":
        # compare each control arm's helix gain vs the S arm at alpha*
        arms = {load_arm(p)["name"]: load_arm(p) for p in paths}
        out = {}
        for name, arm in arms.items():
            m, lo, hi = unpaired_mean_boot(arm, "cond")
            out[name] = dict(helix_cond=m, ci=[lo, hi],
                             sheet_cond=arm["summary"]["sheet_cond_mean"],
                             valid_rate=arm["summary"]["valid_rate"],
                             kind=arm["meta"].get("kind"))
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        for name, d in out.items():
            print(f"  {name} ({d['kind']}): helix={d['helix_cond']:.3f} "
                  f"ci={d['ci']} sheet={d['sheet_cond']:.3f}")

if __name__ == "__main__":
    main()
