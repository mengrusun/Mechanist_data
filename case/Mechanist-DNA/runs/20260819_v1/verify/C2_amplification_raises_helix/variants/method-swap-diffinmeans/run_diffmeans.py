"""VERIFY C2 — METHOD swap: generate + evaluate arms steering along the DIFF-IN-MEANS
direction v_dm (loaded from vdm.npz), reusing the main experiment's exact generation,
QC, and ESM-2-probe readout machinery (gen_eval / ss_predictor) — MINIMUM DIFF: only the
steering VECTOR is swapped (SAE decoder direction -> diff-in-means direction).

Consumes an arm-spec JSON (same shape as code/run_arms.py arms): each arm
  {name, alpha, seed_block(D|H), n(int) OR target_valid(int, n_cap), seed_offset}
Writes <outdir>/<name>.json per arm. Readout held FIXED = esm2_probe (only method swaps).
"""
import os, sys, json, time, argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../../.."))
sys.path.insert(0, os.path.join(ROOT, "code"))
import common as C
import gen_eval as GE

SEED_BASE = {"D": 10_000, "H": 900_000}   # same disjoint blocks as code/run_arms.py


def seeds_for(block, n, offset=0):
    return [SEED_BASE[block] + offset + i for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--outdir", default=HERE)
    ap.add_argument("--gen_batch", type=int, default=32)
    args = ap.parse_args()
    dev = "cuda:0"
    os.makedirs(args.outdir, exist_ok=True)

    vdm = np.load(os.path.join(HERE, "vdm.npz"))
    vec = torch.tensor(vdm["v_dm"], device=dev, dtype=torch.float32)   # norm-matched
    meta_dir = dict(kind="diffinmeans", norm=float(np.linalg.norm(vdm["v_dm"])),
                    cosine_vdm_vS=float(vdm["cosine_vdm_vS"]),
                    n_helix_codons=int(vdm["n_helix_codons"]))
    print(f"[diffmeans] v_dm loaded: ||v||={meta_dir['norm']:.3f} "
          f"cos(v_dm,v_S)={meta_dir['cosine_vdm_vS']:.3f}", flush=True)

    with open(args.arms) as fh:
        arms = json.load(fh)

    t0 = time.time()
    evo2 = C.load_evo2()
    import ss_predictor as SP
    folder = SP.SSPredictor(device=dev)      # SAME readout as main experiment (fixed)
    print(f"[diffmeans] models loaded in {time.time()-t0:.0f}s; {len(arms)} arms", flush=True)

    for a in arms:
        name = a["name"]
        outp = os.path.join(args.outdir, f"{name}.json")
        if os.path.exists(outp) and a.get("skip_if_done", True):
            print(f"[diffmeans] skip {name} (exists)", flush=True); continue
        alpha = float(a["alpha"]); block = a.get("seed_block", "D")
        seed_off0 = int(a.get("seed_offset", 0))
        m = dict(meta_dir); m["seed_block"] = block; m["alpha"] = alpha
        if "target_valid" in a:
            target = int(a["target_valid"]); n_cap = int(a.get("n_cap", 3 * target))
            wave = int(a.get("wave", 256)); raw = 0; merged = None
            while (merged is None or GE.arm_summary(merged)["n_valid"] < target) and raw < n_cap:
                nb = min(wave, n_cap - raw)
                sd = seeds_for(block, nb, seed_off0 + raw)
                arm = GE.eval_arm(evo2, folder, vec, alpha, sd, dev, dict(m),
                                  gen_batch=args.gen_batch, log=print)
                if merged is None:
                    merged = arm
                else:
                    merged["per_seq"].extend(arm["per_seq"])
                    merged["n_gen"] += arm["n_gen"]; merged["n_valid"] += arm["n_valid"]
                raw += nb
                print(f"[diffmeans] {name}: valid={GE.arm_summary(merged)['n_valid']}/{target} raw={raw}",
                      flush=True)
            arm = merged
        else:
            n = int(a["n"]); sd = seeds_for(block, n, seed_off0)
            arm = GE.eval_arm(evo2, folder, vec, alpha, sd, dev, dict(m),
                              gen_batch=args.gen_batch, log=print)
        arm["summary"] = GE.arm_summary(arm); arm["name"] = name
        json.dump(arm, open(outp, "w"))
        s = arm["summary"]
        print(f"[diffmeans] DONE {name}: valid_rate={s['valid_rate']:.2f} "
              f"helix_cond={s['helix_cond_mean']:.3f} helix_itt={s['helix_itt_mean']:.3f} "
              f"ppl_med={s['ppl_median']:.2f}", flush=True)

    print(f"[diffmeans] all arms done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
