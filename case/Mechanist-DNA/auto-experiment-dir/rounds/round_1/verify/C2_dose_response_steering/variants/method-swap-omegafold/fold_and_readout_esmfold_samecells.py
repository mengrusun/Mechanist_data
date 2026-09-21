"""
Verify variant (C2, dimension=method) -- companion cross-check (reviewer-requested strengthening):
fold the SAME saved sequences (from gen_sequences.py's JSONL) with ESMFold, so OmegaFold and ESMFold
are compared on IDENTICAL proteins per (alpha, seed) rather than on independently-generated samples.
Runs in the `scientist` conda env (same as the main experiment's ESMFold path in code/mechanism.py).

Run (scientist env): CUDA_VISIBLE_DEVICES=<gpu> python fold_and_readout_esmfold_samecells.py \
    --in seqs_a8_s42.jsonl --out results_esmfold_samecells_a8_s42.json --plddt_min 50
"""
import os, sys, json, argparse, time
sys.path.insert(0, "/data/wanghaoxiong/intergene_mechanist_v6/code")
import numpy as np
import mechanism as M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plddt_min", type=float, default=50.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    t0 = time.time()

    recs = [json.loads(l) for l in open(args.inp)]
    valid = [r for r in recs if r.get("valid_orf") and r.get("prot")]
    n_total = len(recs); n_valid = len(valid)
    print(f"[esmfold-samecells] {args.inp}: n_total={n_total} n_valid_orf={n_valid}", flush=True)

    per_sample = []
    conf_vals, helix_vals, sheet_vals = [], [], []
    for r in valid:
        entry = {"idx": r["idx"], "gated": True}
        try:
            pdb, plddt = M.esmfold_pdb(r["prot"], device=args.device)
            entry["plddt"] = plddt
            if pdb is not None and plddt is not None and plddt >= args.plddt_min:
                fr = M.dssp_fractions(pdb)
                if fr:
                    entry.update({"helix_hgi": fr["helix_hgi"], "sheet": fr["sheet"],
                                  "n_resolved": fr["n_resolved"], "gated": False})
                    helix_vals.append(fr["helix_hgi"]); sheet_vals.append(fr["sheet"])
            if plddt is not None:
                conf_vals.append(plddt)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {str(e)[:150]}"
        per_sample.append(entry)

    agg = {
        "n": n_total, "n_valid_orf": n_valid, "n_folded": len(conf_vals),
        "n_gated_pass": len(helix_vals),
        "valid_orf_rate": n_valid / max(n_total, 1),
        "gated_pass_rate": len(helix_vals) / max(n_total, 1),
        "plddt_mean": float(np.mean(conf_vals)) if conf_vals else None,
        "helix_hgi_mean": float(np.mean(helix_vals)) if helix_vals else None,
        "helix_hgi_sem": float(np.std(helix_vals) / np.sqrt(len(helix_vals))) if len(helix_vals) > 1 else None,
        "sheet_mean": float(np.mean(sheet_vals)) if sheet_vals else None,
    }
    result = {"config": {"plddt_min": args.plddt_min, "n_total": n_total, "predictor": "ESMFold_samecells"},
              "aggregate": agg, "per_sample": per_sample, "elapsed_s": time.time() - t0}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"[esmfold-samecells] wrote {args.out}: helix_mean={agg['helix_hgi_mean']} "
          f"sheet_mean={agg['sheet_mean']} plddt_mean={agg['plddt_mean']} "
          f"gated_pass_rate={agg['gated_pass_rate']:.2f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
    import os as _os; sys.stdout.flush(); _os._exit(0)
