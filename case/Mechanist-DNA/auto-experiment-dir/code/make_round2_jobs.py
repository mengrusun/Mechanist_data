"""
Emit dispatch2 jobs.json for a round-2 phase (m2 | m3 | null), deriving sigma_proj-unit doses from
M1's calibration so the grid is anchored to round-1's raw-alpha optimum.

M2 grid re-bind (documented in MECHANISM_ROUTING plan-reconciliation): the plan's nominal c_sigma
grid is re-bound to round-1's raw alphas [-2,0,1,2,4,8,12,16,24,32] mapped onto the c-axis via M1's
raw_alpha_to_c map (+ alpha 12,24 to refine around the round-1 optimum at alpha=8). This GUARANTEES
the interior optimum is bracketed with capability-degraded high doses (alpha 16-32) included so c*
is interior to the tested range (plan P7), not a grid edge.

Usage:
  python code/make_round2_jobs.py m2   --out jobs_m2.json
  python code/make_round2_jobs.py m3   --c_star <c*> --out jobs_m3.json
  python code/make_round2_jobs.py null --c_star <c*> --n_dirs 48 --chunk 12 --out jobs_null.json
"""
import os, sys, json, argparse

RES = "results"
CALIB = os.path.join(RES, "m1_calibration.json")
M2_RAW_ALPHAS = [-2, 0, 1, 2, 4, 8, 12, 16, 24, 32]
SEEDS = [42, 200, 201]
KINDS = ["alpha_helix_S", "matched_control", "beta_sheet_offtarget"]
PRED = "esmfold,omegafold"


def c_of_raw(calib, a):
    base_norm, sigma = calib["base_norm"], calib["sigma_proj"]
    return round(a * base_norm / sigma, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["m2", "m3", "null"])
    ap.add_argument("--c_star", type=float, default=None)
    ap.add_argument("--n_dirs", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--n_per_dose", type=int, default=300)
    ap.add_argument("--n_per_dir", type=int, default=120)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    calib = json.load(open(CALIB))
    jobs = []

    if args.phase == "m2":
        cs = sorted(set(c_of_raw(calib, a) for a in M2_RAW_ALPHAS))
        for c in cs:
            for s in SEEDS:
                rid = f"m2_alpha_helix_S_c{c}_s{s}"
                jobs.append({"run_id": rid, "log": f"{RES}/{rid}.log",
                             "cmd": ["code/m_steer_run.py", "--feature_kind", "alpha_helix_S",
                                     "--c_sigma", str(c), "--seed", str(s), "--n_per_dose",
                                     str(args.n_per_dose), "--predictors", PRED,
                                     "--out", f"{RES}/{rid}.json"]})

    elif args.phase == "m3":
        assert args.c_star is not None
        # bracket c* with grid neighbors from the M2 dose set
        cs_all = sorted(set(c_of_raw(calib, a) for a in M2_RAW_ALPHAS))
        below = [c for c in cs_all if c < args.c_star]
        above = [c for c in cs_all if c > args.c_star]
        c_low = max(below) if below else 0.0
        c_high = min(above) if above else args.c_star
        bracket = sorted(set([0.0, round(c_low, 4), round(args.c_star, 4), round(c_high, 4)]))
        for kind in KINDS:
            for c in bracket:
                for s in SEEDS:
                    rid = f"m3_{kind}_c{c}_s{s}"
                    jobs.append({"run_id": rid, "log": f"{RES}/{rid}.log",
                                 "cmd": ["code/m_steer_run.py", "--feature_kind", kind,
                                         "--c_sigma", str(c), "--seed", str(s), "--n_per_dose",
                                         str(args.n_per_dose), "--predictors", PRED,
                                         "--out", f"{RES}/{rid}.json"]})

    elif args.phase == "null":
        assert args.c_star is not None
        cstar = round(args.c_star, 4)
        start = 0
        while start < args.n_dirs:
            end = min(start + args.chunk - 1, args.n_dirs - 1)
            rid = f"m3_random_c{cstar}_d{start}-{end}"
            jobs.append({"run_id": rid, "log": f"{RES}/{rid}.log",
                         "cmd": ["code/m3_random2.py", "--dir_start", str(start), "--dir_end", str(end),
                                 "--c_sigma", str(cstar), "--n_per_dose", str(args.n_per_dir),
                                 "--predictors", PRED, "--out", f"{RES}/{rid}.json"]})
            start = end + 1

    json.dump(jobs, open(args.out, "w"), indent=2)
    print(f"[jobs] phase={args.phase} n_jobs={len(jobs)} -> {args.out}")
    if args.phase == "m2":
        print(f"[jobs] M2 doses (c_sigma): {sorted(set(c_of_raw(calib,a) for a in M2_RAW_ALPHAS))}")
        print(f"[jobs] round-1 alpha=8 -> c={c_of_raw(calib,8)}, alpha=16 -> c={c_of_raw(calib,16)}")


if __name__ == "__main__":
    main()
