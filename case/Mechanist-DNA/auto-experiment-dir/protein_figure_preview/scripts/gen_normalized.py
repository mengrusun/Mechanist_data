"""Normalized generation: for every DNA in all_dnas.json, orig-19 + Encoder-Clamp-Decoder (delta),
doses {0,2,4,8} x 5 seeds. Writes the normalized layout:
    protein_figure_preview/<DNA_name>/seed_<seed>/dose_<dose>.pdb
Readouts go to a shard json (rendering + montage are separate steps). Sharded for parallel GPUs.
Run: CUDA_VISIBLE_DEVICES=3 python protein_figure_preview/scripts/gen_normalized.py --shard 0 --nshards 5
"""
import os, sys, json, time, argparse
import numpy as np, torch
ROOT = "/data/wanghaoxiong/intergene_mechanist_v6"
sys.path.insert(0, os.path.join(ROOT, "code"))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evo2_sae import load_evo2, BatchTopKSAE
import harness2 as H
from fold import esmfold_pdb, structural_readout
from steering import make_steerer

PREVIEW = os.path.join(ROOT, "protein_figure_preview")
FEATURESET = os.path.join(ROOT, "results/m0_feature_set.json")   # original 19-feature set
DOSES = [0, 2, 4, 8]
SEEDS = [101, 200, 300, 111, 222]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--dnas", default="", help="comma list to restrict to (default: all in all_dnas.json)")
    ap.add_argument("--seeds", default="", help="comma list of seeds (default: standard 5)")
    args = ap.parse_args()
    t0 = time.time()
    dnas = json.load(open(os.path.join(HERE, "all_dnas.json")))
    if args.dnas:
        dnas = {k: dnas[k] for k in args.dnas.split(",")}
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else SEEDS
    fs = json.load(open(FEATURESET))
    evo2 = load_evo2("cuda:0")
    sae = BatchTopKSAE(device="cuda:0")
    steerer = make_steerer("clampdecode", evo2, sae, fs["helix_features"], fs["s_f"],
                           device="cuda:0", clamp_mode="delta")

    combos = [(name, s) for name in dnas for s in seeds]
    mine = [c for i, c in enumerate(combos) if i % args.nshards == args.shard]
    print(f"[norm] shard {args.shard}: {len(mine)} (dna,seed) combos", flush=True)

    entries = []
    for (name, seed) in mine:
        dna = dnas[name]; prompt = dna["cds"][:45]; table = dna["table"]
        seed_dir = os.path.join(PREVIEW, name, f"seed_{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        for dose in DOSES:
            rec = H.generate_proteins(evo2, steerer, [prompt], float(dose), 300, 0.7, 4,
                                      seed_base=seed, table=table)[0]
            entries.append({"dna": name, "seed": seed, "dose": dose,
                            "valid_orf": rec.get("valid_orf", False), "prot": rec.get("prot")})
        print(f"[norm] gen {name} s{seed} ({time.time()-t0:.0f}s)", flush=True)
    del evo2
    torch.cuda.empty_cache()

    for e in entries:
        seed_dir = os.path.join(PREVIEW, e["dna"], f"seed_{e['seed']}")
        if not e.get("valid_orf") or not e.get("prot"):
            e["struct"] = None; e.pop("prot", None); continue
        pdb, mp = esmfold_pdb(e["prot"], device="cuda:0")
        if pdb is None:
            e["struct"] = None; e.pop("prot", None); continue
        rd = structural_readout(pdb)
        if rd is not None:
            fp = os.path.join(seed_dir, f"dose_{e['dose']}.pdb")
            open(fp, "w").write(pdb); e["pdb_path"] = fp; rd["struct_mean_plddt"] = mp
        e["struct"] = rd; e.pop("prot", None)

    os.makedirs(os.path.join(PREVIEW, "_shards"), exist_ok=True)
    json.dump({"config": {"features": "orig19", "method": "clampdecode/delta",
                          "doses": DOSES, "seeds": SEEDS}, "entries": entries},
              open(os.path.join(PREVIEW, "_shards", f"shard{args.shard}.json"), "w"), indent=2)
    print(f"[norm] shard {args.shard} DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
    import os as _os; sys.stdout.flush(); _os._exit(0)
