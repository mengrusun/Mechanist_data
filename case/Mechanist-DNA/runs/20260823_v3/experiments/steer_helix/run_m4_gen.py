"""M4 generation (GPU): winning-setting conditions on the TEST split.

Conditions (selectable via --conditions so the queue can split across cards):
  - baseline        : no intervention (coef 0)                                  -> reference
  - caa_win         : CAA direction at winning site, winning coef               -> the lever
  - matched_control : random unit direction of equal norm at the winning site   -> expect no gain
  - sham            : permuted CAA direction of identical norm at same site      -> expect no gain

The winning coef is chosen by the pre-registered frontier rule: max helix_all among coefs whose
validity is NON-INFERIOR to baseline (one-sided 95% percentile-bootstrap lower bound of the
validity delta > -margin). Deterministic from M3, so every split process agrees. Raw per-generation
records are flushed per condition so a session/OOM death still leaves usable artifacts.
"""
import os, sys, json, argparse, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import common as C
from run_m3 import build_caa_vector, score_generations


def reconstruct_valid_count(rate, n):
    return int(round(rate * n))


def pick_winning_coef(m3_rows, baseline_coef, margin, n_boot=20000, seed=0):
    rng = np.random.default_rng(seed)
    by = {}
    for r in m3_rows:
        by.setdefault(r["coef"], []).append(r)

    def valid_array(coef):
        arr = []
        for r in by[coef]:
            k = reconstruct_valid_count(r["validity_rate"], r["n"])
            arr.extend([1] * k + [0] * (r["n"] - k))
        return np.array(arr)

    base_v = valid_array(baseline_coef)
    elig = []
    for coef in sorted(by):
        if coef == baseline_coef:
            continue
        cv = valid_array(coef)
        bi = rng.integers(0, len(cv), (n_boot, len(cv)))
        bj = rng.integers(0, len(base_v), (n_boot, len(base_v)))
        deltas = cv[bi].mean(1) - base_v[bj].mean(1)
        lo = float(np.percentile(deltas, 5.0))
        helix = float(np.mean([r["helix_all"] for r in by[coef]]))
        elig.append((coef, helix, bool(lo > -margin), lo))
    passing = [(c, h) for (c, h, ni, lo) in elig if ni]
    win = max(passing, key=lambda t: t[1])[0] if passing else min(elig, key=lambda t: t[0])[0]
    return float(win), elig


def gen_condition(evo, assay, site, vec, coef, seeds, n):
    recs = []
    for seed in seeds:
        gen = []
        for cc in range(int(np.ceil(n / len(C.DNA_PRIMERS)))):
            evo.clear_hooks()
            if vec is not None and coef != 0.0:
                evo.add_steering_hook(site, vec, coef)
            seqs = evo.generate(C.DNA_PRIMERS, seed=900000 + seed * 1000 + cc)  # TEST offset
            evo.clear_hooks()
            nlls = evo.sequence_nll(seqs)
            gen.extend(score_generations(evo, assay, seqs, nlls))
        recs.append(gen[:n])
    return recs


def flatten(recs):
    out = []
    for si, seed_gen in enumerate(recs):
        for g in seed_gen:
            out.append(dict(seed_idx=si, helix_frac=g["helix_frac"], sheet_frac=g["sheet_frac"],
                            valid=bool(g["valid"]), mean_nll=g["mean_nll"], gc=g["gc"],
                            prot_len=g["prot_len"], aa_helixfav=g["aa_helixfav"], plddt=g["plddt"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=int, default=28)
    ap.add_argument("--baseline_coef", type=float, default=0.0)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--conditions", default="baseline,caa_win,matched_control,sham")
    ap.add_argument("--m1_out", default=os.path.join(C.PROJECT, "results/m1"))
    ap.add_argument("--m3_dir", default=os.path.join(C.PROJECT, "results/m3"))
    ap.add_argument("--out", default=os.path.join(C.PROJECT, "results/m4"))
    ap.add_argument("--gpu", default=None)
    ap.add_argument("--plan_only", action="store_true",
                    help="write results/m3/winning.json (CPU) and exit; run this once before GPU jobs")
    args = ap.parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    want = [c.strip() for c in args.conditions.split(",") if c.strip()]

    win_path = os.path.join(args.m3_dir, "winning.json")
    if args.plan_only or not os.path.exists(win_path):
        m3 = json.load(open(os.path.join(args.m3_dir, f"s{args.site}_caa.json")))
        win_coef, elig = pick_winning_coef(m3["rows"], args.baseline_coef, args.margin)
        C.save_json(dict(site=args.site, coef=win_coef, margin=args.margin,
                         rule="max helix_all s.t. validity NI (one-sided 95% bootstrap LB > -margin)",
                         eligibility=[dict(coef=c, helix=h, ni_pass=ni, ni_lb=lo) for c, h, ni, lo in elig]),
                    win_path)
        if args.plan_only:
            print(f"[M4-gen] plan_only: winning site={args.site} coef={win_coef}", flush=True)
            print("[M4-gen] eligibility:", json.dumps(
                [dict(coef=c, helix=round(h, 4), ni_pass=ni, ni_lb=round(lo, 4)) for c, h, ni, lo in elig]), flush=True)
            return
    else:
        win_coef = float(json.load(open(win_path))["coef"])
    print(f"[M4-gen] winning site={args.site} coef={win_coef}; conditions={want}", flush=True)

    evo = C.Evo2Wrapper("evo2_7b")
    assay = C.HelixAssay()
    vec, _ = build_caa_vector(evo, args.m1_out, args.site)
    vnorm = float(np.linalg.norm(vec))
    rng = np.random.default_rng(0)
    rand = rng.standard_normal(vec.shape).astype(np.float32)
    rand = rand / (np.linalg.norm(rand) + 1e-9) * vnorm
    sham = vec[rng.permutation(vec.shape[0])].copy()
    sham = sham / (np.linalg.norm(sham) + 1e-9) * vnorm

    spec = {"baseline": (None, 0.0), "caa_win": (vec, win_coef),
            "matched_control": (rand, win_coef), "sham": (sham, win_coef)}
    t0 = time.time()
    for cond in want:
        v, cf = spec[cond]
        recs = gen_condition(evo, assay, args.site, v, cf, seeds, args.n)
        C.save_json(dict(site=args.site, coef=cf, vnorm=vnorm, condition=cond, per_gen=flatten(recs)),
                    os.path.join(args.out, f"raw_{cond}.json"))
        print(f"[M4-gen] {cond} done @ {time.time()-t0:.0f}s", flush=True)
    print("[M4-gen] COMPLETE", flush=True)


if __name__ == "__main__":
    main()
