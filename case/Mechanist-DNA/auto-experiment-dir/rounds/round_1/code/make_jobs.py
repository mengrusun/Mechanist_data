"""Generate M2 / M3 job manifests for the dispatcher (one process per (config))."""
import json, sys, os

N_PER_DOSE = int(sys.argv[2]) if len(sys.argv) > 2 else 300
TEMP = float(sys.argv[3]) if len(sys.argv) > 3 else 0.7
which = sys.argv[1] if len(sys.argv) > 1 else "all"
RUNS = "runs"


def m2_jobs():
    jobs = []
    for alpha in [-2, 0, 1, 2, 4, 8, 16, 32]:
        for seed in [42, 200, 201]:
            rid = f"M2_a{alpha}_s{seed}"
            jobs.append({"run_id": rid,
                         "cmd": ["code/m2_dose_response.py", "--alpha", str(alpha), "--seed", str(seed),
                                 "--n_per_dose", str(N_PER_DOSE), "--temperature", str(TEMP),
                                 "--out", f"results/{rid.lower()}.json"],
                         "log": f"{RUNS}/{rid}.log"})
    return jobs


def m3_jobs():
    jobs = []
    for kind in ["alpha_helix_S", "matched_control", "beta_sheet_offtarget"]:
        for alpha in [0, 4, 8, 16]:
            for seed in [42, 200, 201]:
                rid = f"M3_{kind}_a{alpha}_s{seed}"
                jobs.append({"run_id": rid,
                             "cmd": ["code/m3_specificity.py", "--feature_kind", kind, "--alpha", str(alpha),
                                     "--seed", str(seed), "--n_per_dose", str(N_PER_DOSE),
                                     "--temperature", str(TEMP), "--out", f"results/{rid.lower()}.json"],
                             "log": f"{RUNS}/{rid}.log"})
    return jobs


if which in ("m2", "all"):
    json.dump(m2_jobs(), open("runs/m2_jobs.json", "w"), indent=2)
    print("m2 jobs:", len(m2_jobs()))
if which in ("m3", "all"):
    json.dump(m3_jobs(), open("runs/m3_jobs.json", "w"), indent=2)
    print("m3 jobs:", len(m3_jobs()))
