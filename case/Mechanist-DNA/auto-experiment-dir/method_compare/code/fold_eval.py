"""Structural evaluation of delivered sequences: ESMFold + DSSP.

ESMFold is NEVER used inside the search (the scorer ensemble is sequence-level only), so it is a
held-out endpoint here and the comparison is not circular. Endpoints match the paper:
  helix_hgi    unweighted alpha-helix fraction (DSSP H/G/I)  -- the 43.8% -> 56.6% endpoint
  helix_hgi_w  pLDDT-weighted alpha-helix fraction           -- the ledger's primary endpoint
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
import mc_env; mc_env.patch()
from fold import fold_and_read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--predictor", default="esmfold")
    ap.add_argument("--max_len", type=int, default=400)
    args = ap.parse_args()

    d = json.load(open(args.inp))
    key = f"struct_{args.predictor}"
    todo = [r for r in d["records"] if r.get("prot") and key not in r]
    print(f"[fold] {os.path.basename(args.inp)}: {len(todo)} to fold "
          f"({sum(1 for r in d['records'] if r.get('prot'))} valid ORFs)", flush=True)
    t0 = time.time()
    for n, r in enumerate(todo):
        try:
            r[key] = fold_and_read(r["prot"], predictor=args.predictor, device="cuda:0",
                                   max_len=args.max_len)
        except Exception as e:
            r[key] = None
            r[f"{key}_error"] = f"{type(e).__name__}: {str(e)[:80]}"
        if (n + 1) % 25 == 0:
            print(f"[fold] {n+1}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    d.setdefault("fold_cost", {})[args.predictor] = {
        "n_folded": sum(1 for r in d["records"] if r.get(key)),
        "seconds": time.time() - t0,
    }
    json.dump(d, open(args.inp, "w"), indent=2)
    print(f"[fold] WROTE {args.inp} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush(); os._exit(0)
