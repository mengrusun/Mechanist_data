"""Run ESMFold baseline on curated chains and select a working test set.

For each candidate: predict structure, run DSSP on prediction, and check
whether the ground-truth hairpin is also reproduced in the prediction.
Also build a "control" (hairpin-broken) sequence for each, run DSSP.

Saves: outputs/baseline.jsonl   with per-chain flags and metadata.
Also saves: outputs/per_block/{pdb}_{chain}/native_s_z.pt (per-block s,z for hairpin region).
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import numpy as np

from common import (load_esmfold, esmfold_infer, output_to_pdb_str,
                    run_dssp, dssp_ss_string, find_hairpins_in_ss, write_pdb,
                    DATA_DIR, OUT_DIR)


def break_hairpin(seq, hp, mode="poly_g"):
    s1s, s1e, ls, le, s2s, s2e = hp
    span = list(range(s1s, s2e + 1))
    arr = list(seq)
    if mode == "poly_g":
        for i in span:
            arr[i] = "G"
    elif mode == "poly_p":
        for i in span:
            arr[i] = "P"
    return "".join(arr)


def check_hairpin(ss, hp, tol_shift=2, min_e=2):
    """Return True if the SS in the hairpin region contains two E-strand stretches near the expected positions."""
    s1s, s1e, ls, le, s2s, s2e = hp
    e_in_s1 = sum(1 for c in ss[max(0, s1s - tol_shift): s1e + 1 + tol_shift] if c == 'E')
    e_in_s2 = sum(1 for c in ss[max(0, s2s - tol_shift): s2e + 1 + tol_shift] if c == 'E')
    return (e_in_s1 >= min_e) and (e_in_s2 >= min_e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DATA_DIR / "hairpin_chains.jsonl"))
    ap.add_argument("--out", default=str(OUT_DIR / "baseline.jsonl"))
    ap.add_argument("--pdb_out_dir", default=str(OUT_DIR / "pred_pdb"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_chains", type=int, default=300)
    ap.add_argument("--min_len", type=int, default=40)
    ap.add_argument("--max_len", type=int, default=90)
    ap.add_argument("--num_recycles", type=int, default=1)
    args = ap.parse_args()

    Path(args.pdb_out_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    entries = []
    with open(args.input) as f:
        for ln in f:
            e = json.loads(ln)
            if args.min_len <= e["length"] <= args.max_len:
                entries.append(e)
    if args.max_chains:
        entries = entries[: args.max_chains]
    print(f"Loading model on {args.device} ...")
    tok, model = load_esmfold(device=args.device, dtype=torch.float32)

    fo = open(args.out, "w")
    n_native_hp = 0
    n_broken_lost = 0
    t0 = time.time()
    for i, e in enumerate(entries):
        seq = e["seq"]
        hp = e["hairpins"][0]  # first hairpin
        # Native prediction
        try:
            out = esmfold_infer(model, tok, seq, num_recycles=args.num_recycles, device=args.device)
        except Exception as ex:
            print(f"skip {e['pdb']}_{e['chain']}: {ex}")
            continue
        pdb_native = output_to_pdb_str(model, out)
        pdb_p = Path(args.pdb_out_dir) / f"{e['pdb']}_{e['chain']}_native.pdb"
        write_pdb(pdb_native, pdb_p)
        try:
            dssp_n = run_dssp(str(pdb_p))
            ss_n = dssp_ss_string(dssp_n)
        except Exception as ex:
            print(f"dssp fail {e['pdb']}_{e['chain']}: {ex}")
            continue

        # Broken prediction (poly-G in hairpin span)
        seq_b = break_hairpin(seq, hp, mode="poly_g")
        try:
            out_b = esmfold_infer(model, tok, seq_b, num_recycles=args.num_recycles, device=args.device)
        except Exception as ex:
            print(f"skip broken {e['pdb']}_{e['chain']}: {ex}")
            continue
        pdb_broken = output_to_pdb_str(model, out_b)
        pdb_bp = Path(args.pdb_out_dir) / f"{e['pdb']}_{e['chain']}_broken.pdb"
        write_pdb(pdb_broken, pdb_bp)
        try:
            dssp_b = run_dssp(str(pdb_bp))
            ss_b = dssp_ss_string(dssp_b)
        except Exception as ex:
            print(f"dssp fail broken {e['pdb']}_{e['chain']}: {ex}")
            continue

        native_hp_ok = check_hairpin(ss_n, hp)
        broken_hp_lost = not check_hairpin(ss_b, hp, tol_shift=3, min_e=3)
        n_native_hp += int(native_hp_ok)
        n_broken_lost += int(broken_hp_lost)

        entry = {
            "pdb": e["pdb"], "chain": e["chain"], "length": len(seq),
            "seq_native": seq, "seq_broken": seq_b, "hairpin": hp,
            "ss_native": ss_n, "ss_broken": ss_b,
            "native_hp_ok": bool(native_hp_ok),
            "broken_hp_lost": bool(broken_hp_lost),
            "plddt_native": float(out["plddt"].mean().item()),
            "plddt_broken": float(out_b["plddt"].mean().item()),
        }
        fo.write(json.dumps(entry) + "\n")
        fo.flush()
        if (i + 1) % 5 == 0:
            print(f"[{i+1}/{len(entries)}] native_hp_ok={n_native_hp} broken_lost={n_broken_lost} elapsed={time.time()-t0:.1f}s")

    fo.close()
    print(f"done. native_hp_ok={n_native_hp} broken_lost={n_broken_lost}")


if __name__ == "__main__":
    main()
