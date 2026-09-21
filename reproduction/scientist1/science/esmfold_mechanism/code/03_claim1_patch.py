"""Claim 1: Localize β-hairpin decision by layer-wise patching of s vs z.

For each (native, broken) pair:
  1. Full forward on `native` — cache per-block (s_k, z_k).
  2. Full forward on `broken` — cache per-block (s_k, z_k).
  3. Sweep block k in `layers`:
       (a) patch-s(k): forward on `broken` sequence but overwrite output s of
           block k with `native`'s cached s at hairpin positions.
       (b) patch-z(k): similarly overwrite z (at hairpin×hairpin positions).
       Run DSSP on the resulting predicted structure and check if the target
       β-hairpin is now present.
  4. Reverse direction (native → patched with broken s or z) as a
     validation of "removing" the decision.

Writes: outputs/claim1_patch.jsonl
"""
import argparse
import json
import os
import time
from pathlib import Path
import gc

import torch
import numpy as np

from common import (load_esmfold, esmfold_infer, output_to_pdb_str,
                    run_dssp, dssp_ss_string, find_hairpins_in_ss, write_pdb,
                    DATA_DIR, OUT_DIR)
from intervene import capture_block_outputs, patch_s_at_block, patch_z_at_block


def check_hairpin(ss, hp, tol_shift=2, min_e=2):
    s1s, s1e, ls, le, s2s, s2e = hp
    e_in_s1 = sum(1 for c in ss[max(0, s1s - tol_shift): s1e + 1 + tol_shift] if c == 'E')
    e_in_s2 = sum(1 for c in ss[max(0, s2s - tol_shift): s2e + 1 + tol_shift] if c == 'E')
    return int((e_in_s1 >= min_e) and (e_in_s2 >= min_e))


def hairpin_positions(hp, extend_flank=0, seq_len=None):
    s1s, s1e, ls, le, s2s, s2e = hp
    pos = list(range(max(0, s1s - extend_flank), min(seq_len, s2e + 1 + extend_flank)))
    return pos


@torch.no_grad()
def infer_and_dssp(model, tok, seq, num_recycles, device, tmp_pdb):
    out = esmfold_infer(model, tok, seq, num_recycles=num_recycles, device=device)
    pdb = output_to_pdb_str(model, out)
    write_pdb(pdb, tmp_pdb)
    dssp = run_dssp(tmp_pdb)
    return out, pdb, dssp_ss_string(dssp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(OUT_DIR / "baseline.jsonl"))
    ap.add_argument("--out", default=str(OUT_DIR / "claim1_patch.jsonl"))
    ap.add_argument("--tmp_dir", default="/tmp/claim1_pdb")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_recycles", type=int, default=1)
    ap.add_argument("--max_chains", type=int, default=25)
    ap.add_argument("--layers", type=str, default="0,2,4,6,8,10,12,16,20,24,32,40,47")
    ap.add_argument("--extend", type=int, default=2, help="extend hairpin positions by this many flanking residues on each side")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    entries = []
    with open(args.baseline) as f:
        for ln in f:
            e = json.loads(ln)
            if e.get("native_hp_ok") and e.get("broken_hp_lost"):
                entries.append(e)
    print(f"good baseline pairs: {len(entries)}")
    entries = entries[: args.max_chains]

    print(f"loading model on {args.device} ...")
    tok, model = load_esmfold(device=args.device, dtype=torch.float32)

    fo = open(args.out, "w")
    t_start = time.time()
    for idx, e in enumerate(entries):
        seq_n = e["seq_native"]
        seq_b = e["seq_broken"]
        hp = tuple(e["hairpin"])
        pos = hairpin_positions(hp, extend_flank=args.extend, seq_len=len(seq_n))

        # Cache native s,z
        cache_n = {}
        with capture_block_outputs(model, cache_n):
            out_n = esmfold_infer(model, tok, seq_n, num_recycles=args.num_recycles, device=args.device)
        # Cache broken s,z
        cache_b = {}
        with capture_block_outputs(model, cache_b):
            out_b = esmfold_infer(model, tok, seq_b, num_recycles=args.num_recycles, device=args.device)

        # Also do baseline SS
        pdb_n = output_to_pdb_str(model, out_n)
        pdb_b = output_to_pdb_str(model, out_b)
        pn = Path(args.tmp_dir) / f"{e['pdb']}_{e['chain']}_n.pdb"
        pb = Path(args.tmp_dir) / f"{e['pdb']}_{e['chain']}_b.pdb"
        write_pdb(pdb_n, pn); write_pdb(pdb_b, pb)
        ss_n = dssp_ss_string(run_dssp(str(pn)))
        ss_b = dssp_ss_string(run_dssp(str(pb)))
        base_native = check_hairpin(ss_n, hp)
        base_broken = check_hairpin(ss_b, hp)

        # sweep
        result_layers = {}
        pos_t = torch.tensor(pos, device=args.device, dtype=torch.long)
        for k in layers:
            # patch-s: run broken but overwrite s at block k with native's s at hairpin positions
            src_s = cache_n[k][0][:, pos, :]  # (1, |pos|, D)
            with patch_s_at_block(model, k, src_s, positions=pos):
                out_ps = esmfold_infer(model, tok, seq_b, num_recycles=args.num_recycles, device=args.device)
            pdb_ps = output_to_pdb_str(model, out_ps)
            p_ps = Path(args.tmp_dir) / f"{e['pdb']}_{e['chain']}_ps_{k}.pdb"
            write_pdb(pdb_ps, p_ps)
            try:
                ss_ps = dssp_ss_string(run_dssp(str(p_ps)))
                hp_after_s = check_hairpin(ss_ps, hp)
            except Exception:
                ss_ps = ""; hp_after_s = -1

            # patch-z at hairpin×hairpin block
            src_z = cache_n[k][1][:, pos, :, :][:, :, pos, :]  # (1,|pos|,|pos|,C)
            with patch_z_at_block(model, k, src_z, pair_positions=pos):
                out_pz = esmfold_infer(model, tok, seq_b, num_recycles=args.num_recycles, device=args.device)
            pdb_pz = output_to_pdb_str(model, out_pz)
            p_pz = Path(args.tmp_dir) / f"{e['pdb']}_{e['chain']}_pz_{k}.pdb"
            write_pdb(pdb_pz, p_pz)
            try:
                ss_pz = dssp_ss_string(run_dssp(str(p_pz)))
                hp_after_z = check_hairpin(ss_pz, hp)
            except Exception:
                ss_pz = ""; hp_after_z = -1

            # Also reverse: run native but overwrite with broken's s/z
            src_s_bn = cache_b[k][0][:, pos, :]
            with patch_s_at_block(model, k, src_s_bn, positions=pos):
                out_rps = esmfold_infer(model, tok, seq_n, num_recycles=args.num_recycles, device=args.device)
            pdb_rps = output_to_pdb_str(model, out_rps)
            p_rps = Path(args.tmp_dir) / f"{e['pdb']}_{e['chain']}_rps_{k}.pdb"
            write_pdb(pdb_rps, p_rps)
            try:
                ss_rps = dssp_ss_string(run_dssp(str(p_rps)))
                rev_after_s = check_hairpin(ss_rps, hp)
            except Exception:
                ss_rps = ""; rev_after_s = -1

            src_z_bn = cache_b[k][1][:, pos, :, :][:, :, pos, :]
            with patch_z_at_block(model, k, src_z_bn, pair_positions=pos):
                out_rpz = esmfold_infer(model, tok, seq_n, num_recycles=args.num_recycles, device=args.device)
            pdb_rpz = output_to_pdb_str(model, out_rpz)
            p_rpz = Path(args.tmp_dir) / f"{e['pdb']}_{e['chain']}_rpz_{k}.pdb"
            write_pdb(pdb_rpz, p_rpz)
            try:
                ss_rpz = dssp_ss_string(run_dssp(str(p_rpz)))
                rev_after_z = check_hairpin(ss_rpz, hp)
            except Exception:
                ss_rpz = ""; rev_after_z = -1

            result_layers[k] = {
                "hp_after_s_patch_b2n": hp_after_s,
                "hp_after_z_patch_b2n": hp_after_z,
                "hp_after_s_patch_n2b_reverse": rev_after_s,
                "hp_after_z_patch_n2b_reverse": rev_after_z,
                "ss_ps": ss_ps, "ss_pz": ss_pz, "ss_rps": ss_rps, "ss_rpz": ss_rpz,
            }
        # release caches
        del cache_n, cache_b
        gc.collect(); torch.cuda.empty_cache()

        entry = {
            "pdb": e["pdb"], "chain": e["chain"], "hairpin": hp,
            "positions": pos,
            "seq_native": seq_n, "seq_broken": seq_b,
            "base_native": base_native, "base_broken": base_broken,
            "ss_native_pred": ss_n, "ss_broken_pred": ss_b,
            "layers": result_layers,
        }
        fo.write(json.dumps(entry) + "\n")
        fo.flush()
        print(f"[{idx+1}/{len(entries)}] {e['pdb']}_{e['chain']} done | elapsed={time.time()-t_start:.1f}s")

    fo.close()


if __name__ == "__main__":
    main()
