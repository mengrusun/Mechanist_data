"""Workhorse: run a set of generation+eval arms on ONE GPU (models loaded once).

Consumes an arm-spec JSON list; each arm:
  {name, alpha, control(None|random_feature|beta_sheet|null_direction),
   seed_block(D|H), n(int) OR target_valid(int, with n_cap)}
Writes results/<name>.json per arm.

Seed blocks (disjoint, paired across alpha):
  D (dev)      : seeds 10_000 + i
  H (held-out) : seeds 900_000 + i
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import common as C
import gen_eval as GE

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED_BASE = {"D": 10_000, "H": 900_000}

def seeds_for(block, n, offset=0):
    b = SEED_BASE[block]
    return [b + offset + i for i in range(n)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True, help="arm-spec JSON file")
    ap.add_argument("--features", default=os.path.join(ROOT, "results/m1_features.json"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--gen_batch", type=int, default=32)
    ap.add_argument("--readout", default="esm2_probe",
                    choices=["esm2_probe", "esmfold"])
    args = ap.parse_args()
    dev = "cuda:0"
    os.makedirs(args.outdir, exist_ok=True)
    with open(args.arms) as fh:
        arms = json.load(fh)

    feat = GE.load_features(args.features)
    if not feat.get("S"):
        print("[run_arms] ERROR: empty feature set S in m1_features.json — C1 precondition unmet.")
        sys.exit(3)

    t0 = time.time()
    evo2 = C.load_evo2()
    sae = C.TiedTopKSAE(device=dev, relu_before_topk=True)
    if args.readout == "esmfold":
        import folding as FD
        folder = FD.ESMFolder(device=dev)
    else:
        import ss_predictor as SP
        folder = SP.SSPredictor(device=dev)
    print(f"[run_arms] models loaded ({args.readout}) in {time.time()-t0:.0f}s; "
          f"{len(arms)} arms", flush=True)

    for a in arms:
        name = a["name"]
        outp = os.path.join(args.outdir, f"{name}.json")
        if os.path.exists(outp) and a.get("skip_if_done", True):
            print(f"[run_arms] skip {name} (exists)", flush=True)
            continue
        alpha = float(a["alpha"])
        control = a.get("control")
        block = a.get("seed_block", "D")
        vec, meta = GE.build_steer_vector(feat, sae, dev, control=control,
                                          seed=a.get("ctrl_seed", 0))
        meta["seed_block"] = block

        seed_off0 = int(a.get("seed_offset", 0))   # base offset for fresh/disjoint seeds
        if "target_valid" in a:
            # generate in waves until target #valid QC-passing folded proteins
            target = int(a["target_valid"]); n_cap = int(a.get("n_cap", 4*target))
            got, raw, all_arm = 0, 0, None   # raw = #generated this run (excl. seed_off0)
            wave = int(a.get("wave", 128))
            merged = None
            while got < target and raw < n_cap:
                nb = min(wave, n_cap - raw)
                sd = seeds_for(block, nb, seed_off0 + raw)
                arm = GE.eval_arm(evo2, folder, vec, alpha, sd, dev, dict(meta),
                                  gen_batch=args.gen_batch, log=print)
                if merged is None:
                    merged = arm
                else:
                    merged["per_seq"].extend(arm["per_seq"])
                    merged["n_gen"] += arm["n_gen"]; merged["n_valid"] += arm["n_valid"]
                raw += nb
                summ = GE.arm_summary(merged)
                got = summ["n_valid"]
                print(f"[run_arms] {name}: valid={got}/{target} raw={raw}", flush=True)
            arm = merged
        else:
            n = int(a["n"])
            sd = seeds_for(block, n)
            arm = GE.eval_arm(evo2, folder, vec, alpha, sd, dev, dict(meta),
                              gen_batch=args.gen_batch, log=print)

        arm["summary"] = GE.arm_summary(arm)
        arm["name"] = name
        with open(outp, "w") as fh:
            json.dump(arm, fh)
        s = arm["summary"]
        print(f"[run_arms] DONE {name}: valid_rate={s['valid_rate']:.2f} "
              f"helix_cond={s['helix_cond_mean']:.3f} helix_itt={s['helix_itt_mean']:.3f} "
              f"ppl_med={s['ppl_median']:.2f}", flush=True)

    print(f"[run_arms] all arms done in {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
