"""Emit per-GPU arm-spec JSON files for a milestone, sharding arms across GPUs.

Usage:
  python make_specs.py m2                       # dev alpha grid (block D, n=300)
  python make_specs.py mctrl --alpha_star A     # control arms at alpha* (block D)
  python make_specs.py m3 --alpha_star A        # held-out confirm (block H, target_valid)
Writes results/specs/<milestone>_gpu{k}.json for k in 0..ngpu-1.
"""
import os, sys, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPECDIR = os.path.join(ROOT, "results/specs")

def shard(arms, ngpu):
    groups = [[] for _ in range(ngpu)]
    # place baseline first on gpu0; distribute the rest round-robin balancing load
    for i, a in enumerate(arms):
        groups[i % ngpu].append(a)
    return groups

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("milestone", choices=["m2", "mctrl", "m3"])
    ap.add_argument("--alpha_star", type=float, default=None)
    ap.add_argument("--ngpu", type=int, default=4)
    ap.add_argument("--n_dev", type=int, default=300)
    ap.add_argument("--target_valid", type=int, default=500)
    ap.add_argument("--n_cap", type=int, default=1000)
    args = ap.parse_args()
    os.makedirs(SPECDIR, exist_ok=True)

    if args.milestone == "m2":
        grid = [0, 0.5, 1, 2, 4, 8, 16]
        arms = [dict(name=f"m2_a{a}_D", alpha=a, seed_block="D", n=args.n_dev)
                for a in grid]
    elif args.milestone == "mctrl":
        astar = args.alpha_star
        arms = [dict(name=f"mctrl_{c}", alpha=astar, control=c, seed_block="D",
                     n=args.n_dev)
                for c in ["random_feature", "beta_sheet", "null_direction"]]
        # also an S arm at alpha* on block D for a same-seed reference
        arms.append(dict(name="mctrl_S_ref", alpha=astar, seed_block="D", n=args.n_dev))
    else:  # m3
        astar = args.alpha_star
        alphas = [0.0, astar/2.0, astar, astar*2.0]
        names = ["m3_a0", "m3_ahalf", "m3_astar", "m3_a2x"]
        arms = []
        for a, nm in zip(alphas, names):
            arm = dict(name=nm, alpha=a, seed_block="H",
                       target_valid=args.target_valid, n_cap=args.n_cap)
            if a == 0.0:  # baseline also needs enough valid
                arm["target_valid"] = args.target_valid
            arms.append(arm)

    groups = shard(arms, args.ngpu)
    paths = []
    for k, g in enumerate(groups):
        p = os.path.join(SPECDIR, f"{args.milestone}_gpu{k}.json")
        with open(p, "w") as fh:
            json.dump(g, fh, indent=2)
        paths.append(p)
        print(f"[specs] gpu{k}: {[a['name'] for a in g]}")
    print("SPEC_FILES " + " ".join(paths))

if __name__ == "__main__":
    main()
