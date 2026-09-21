"""
Drive the 12-config M0 scoring grid (organism x helix_def x seed) from cached activations,
writing results/m0_<org>_<helix>_s<seed>.json and runs/<id>/cost.json for each.
Scoring is CPU; the GPU cost was the shared per-organism activation caching (recorded separately).
Run: python code/run_m0_grid.py --organisms prokaryote eukaryote
"""
import os, sys, json, time, argparse, subprocess
sys.path.insert(0, "code")
import m0_data as D

RUNS = "/data/wanghaoxiong/intergene_mechanist_v6/runs"
RES = "/data/wanghaoxiong/intergene_mechanist_v6/results"


def write_cost(run_id, gpu_ids, seconds, note):
    d = os.path.join(RUNS, run_id)
    os.makedirs(d, exist_ok=True)
    json.dump({"run_id": run_id, "gpu_ids": gpu_ids, "seconds": seconds, "note": note,
               "cuda_visible_devices": "2,3,4,5"}, open(os.path.join(d, "cost.json"), "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--organisms", nargs="+", default=["prokaryote", "eukaryote"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 200, 201])
    ap.add_argument("--helix_defs", nargs="+", default=["HGI", "H_only"])
    args = ap.parse_args()
    done, skipped = [], []
    for org in args.organisms:
        if not os.path.exists(os.path.join(D.DATA_DIR, f"m0_acts_{org}.npz")):
            print(f"[grid] SKIP {org}: no cached activations", flush=True)
            skipped.append(org); continue
        for hd in args.helix_defs:
            for seed in args.seeds:
                out = os.path.join(RES, f"m0_{org}_{hd}_s{seed}.json")
                run_id = f"M0_{org}_{hd}_s{seed}_score"
                t0 = time.time()
                r = subprocess.run([sys.executable, "code/m0_feature_selectivity.py",
                                    "--organism", org, "--helix_def", hd, "--split_seed", str(seed),
                                    "--tau_auc", "0.75", "--tau_f1", "0.3", "--fdr", "bh", "--out", out],
                                   capture_output=True, text=True)
                dt = time.time() - t0
                write_cost(run_id, [], dt, "M0 scoring (CPU; activation caching GPU cost recorded per-organism)")
                if r.returncode == 0 and os.path.exists(out):
                    v = json.load(open(out))
                    done.append((org, hd, seed, v["verdict"], v.get("best_test_auroc"),
                                 v.get("combined_set_test_auroc"), v.get("relaxed_S_size")))
                    print(f"[grid] {org} {hd} s{seed}: {v['verdict']} best_auroc={v.get('best_test_auroc'):.3f} "
                          f"combined={v.get('combined_set_test_auroc')} relaxed|S|={v.get('relaxed_S_size')} ({dt:.0f}s)", flush=True)
                else:
                    print(f"[grid] FAIL {org} {hd} s{seed}: {r.stderr[-300:]}", flush=True)
    print(f"\n[grid] completed {len(done)}/{len(done)+len(skipped)*6} configs", flush=True)
    for row in done:
        print("   ", row, flush=True)


if __name__ == "__main__":
    main()
