"""Claim 3 (v2): Positive-control mutation study + amplified steering.

Two experiments:

(A) Mutation study — for each β-hairpin chain, pick the first `n_pairs` (up
    to 3) cross-strand residue pairs (i on strand1, j on strand2, paired
    from the loop outward). For each pair generate four variants by mutating
    (i, j) to combinations of Lys(K, +1) and Glu(E, −1):
        (K,E) opposite,   (E,K) opposite (reversed),
        (K,K) same-positive,   (E,E) same-negative.
    Predict each, run DSSP + Cα cross-strand distance. Physical prediction:
    opposite-charge configurations have smaller cross-strand distance and
    more reliable β-hairpin than same-charge.

(B) Amplified steering — at *early* blocks (0..7), normalize the learned
    charge direction to unit norm and inject at the paired positions with
    scale swept from 0 to a value comparable with ‖s‖. Use broken-hairpin
    sequences as the substrate and test whether flipping the sign of the
    injected direction (opposite vs same) changes β-hairpin recovery.

Outputs:
    outputs/claim3_mutation.jsonl  — mutation study rows
    outputs/claim3_steer_v2.jsonl  — amplified steering rows
"""
import argparse
import json
import time
import gc
from pathlib import Path

import numpy as np
import torch

from common import (load_esmfold, esmfold_infer, output_to_pdb_str,
                    run_dssp, dssp_ss_string, write_pdb, OUT_DIR)
from intervene import add_direction_to_s


def check_hairpin_local(ss, hp, tol_shift=2, min_e=2):
    s1s, s1e, ls, le, s2s, s2e = hp
    e_in_s1 = sum(1 for c in ss[max(0, s1s - tol_shift): s1e + 1 + tol_shift] if c == 'E')
    e_in_s2 = sum(1 for c in ss[max(0, s2s - tol_shift): s2e + 1 + tol_shift] if c == 'E')
    return int((e_in_s1 >= min_e) and (e_in_s2 >= min_e))


def cross_strand_dist_local(pdb_path, hp):
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("x", pdb_path)
    model = st[0]
    ca = {}
    idx = 0
    for chain in model:
        for res in chain:
            if res.id[0] != " ":
                continue
            if "CA" in res:
                ca[idx] = np.array(res["CA"].coord)
                idx += 1
    s1s, s1e, ls, le, s2s, s2e = hp
    strand1 = list(reversed(range(s1s, s1e + 1)))
    strand2 = list(range(s2s, s2e + 1))
    pairs = list(zip(strand1, strand2))
    ds = []
    for i, j in pairs:
        if i in ca and j in ca:
            ds.append(np.linalg.norm(ca[i] - ca[j]))
    if not ds:
        return None, []
    return float(np.mean(ds)), ds


def mutate(seq, positions_aa):
    arr = list(seq)
    for pos, aa in positions_aa:
        arr[pos] = aa
    return "".join(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(OUT_DIR / "baseline.jsonl"))
    ap.add_argument("--out_mut", default=str(OUT_DIR / "claim3_mutation.jsonl"))
    ap.add_argument("--out_steer", default=str(OUT_DIR / "claim3_steer_v2.jsonl"))
    ap.add_argument("--direction_pt", default=str(OUT_DIR / "claim3_direction.pt"))
    ap.add_argument("--tmp_dir", default="/tmp/claim3v2_pdb")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_recycles", type=int, default=1)
    ap.add_argument("--n_chains", type=int, default=25)
    ap.add_argument("--n_pairs", type=int, default=2, help="paired residues per hairpin")
    ap.add_argument("--steer_blocks", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--steer_scales", default="0,5,10,15,20,30")
    ap.add_argument("--skip_mutation", action="store_true")
    args = ap.parse_args()

    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_mut).parent.mkdir(parents=True, exist_ok=True)

    entries = []
    with open(args.baseline) as f:
        for ln in f:
            e = json.loads(ln)
            if e.get("native_hp_ok"):
                entries.append(e)
    entries = entries[: args.n_chains]
    print(f"chains: {len(entries)}")

    tok, model = load_esmfold(device=args.device, dtype=torch.float32)

    # ---------- (A) mutation study (skipped when --skip_mutation) ----------
    if args.skip_mutation:
        print("Skipping mutation study (already ran).")
    else:
        fmut = open(args.out_mut, "w")
        t0 = time.time()
        for idx, e in enumerate(entries):
            seq = e["seq_native"]
            hp = tuple(e["hairpin"])
            s1s, s1e, ls, le, s2s, s2e = hp
            strand1 = list(reversed(range(s1s, s1e + 1)))
            strand2 = list(range(s2s, s2e + 1))
            pairs = list(zip(strand1, strand2))[: args.n_pairs]
            outb = esmfold_infer(model, tok, seq, num_recycles=args.num_recycles, device=args.device)
            pdbb = output_to_pdb_str(model, outb); pb = Path(args.tmp_dir) / f"{e['pdb']}_{e['chain']}_base.pdb"; write_pdb(pdbb, pb)
            ss_b = dssp_ss_string(run_dssp(str(pb)))
            base_d, _ = cross_strand_dist_local(str(pb), hp)
            base_hp = check_hairpin_local(ss_b, hp)

            variants = {"KE_opp": [("K", "E")], "EK_opp": [("E", "K")],
                        "KK_same_pos": [("K", "K")], "EE_same_neg": [("E", "E")]}
            results = {}
            for name, mapping in variants.items():
                aa_i, aa_j = mapping[0]
                positions = [(i, aa_i) for (i, j) in pairs] + [(j, aa_j) for (i, j) in pairs]
                seq_m = mutate(seq, positions)
                outm = esmfold_infer(model, tok, seq_m, num_recycles=args.num_recycles, device=args.device)
                pdbm = output_to_pdb_str(model, outm)
                pm = Path(args.tmp_dir) / f"{e['pdb']}_{e['chain']}_mut_{name}.pdb"
                write_pdb(pdbm, pm)
                try:
                    ss_m = dssp_ss_string(run_dssp(str(pm)))
                    d_m, _ = cross_strand_dist_local(str(pm), hp)
                    hp_m = check_hairpin_local(ss_m, hp)
                except Exception:
                    ss_m = ""; d_m = None; hp_m = -1
                results[name] = {"seq": seq_m, "ss": ss_m, "d": d_m, "hp": hp_m,
                                 "plddt": float(outm["plddt"].mean().item())}
            entry = {
                "pdb": e["pdb"], "chain": e["chain"], "hairpin": hp,
                "pairs": pairs, "base_seq": seq, "base_ss": ss_b, "base_d": base_d, "base_hp": base_hp,
                "base_plddt": float(outb["plddt"].mean().item()),
                "variants": results,
            }
            fmut.write(json.dumps(entry) + "\n")
            fmut.flush()
            print(f"[MUT {idx+1}/{len(entries)}] {e['pdb']}_{e['chain']} base_d={base_d:.2f} "
                  f"KE={results['KE_opp']['d']} KK={results['KK_same_pos']['d']} "
                  f"EE={results['EE_same_neg']['d']} EK={results['EK_opp']['d']} "
                  f"elapsed={time.time()-t0:.1f}s")
            gc.collect(); torch.cuda.empty_cache()
        fmut.close()

    # ---------- (B) amplified steering ----------
    directions = torch.load(args.direction_pt, weights_only=False)
    requested_blocks = [int(x) for x in args.steer_blocks.split(",")]
    steer_blocks = [k for k in requested_blocks if k in directions]
    if not steer_blocks:
        raise ValueError(f"None of requested steer_blocks {requested_blocks} are in directions {list(directions.keys())}")
    print(f"steering at blocks {steer_blocks}")
    steer_scales = [float(x) for x in args.steer_scales.split(",")]
    # unit-normalize the direction at each block; take average across blocks as
    # a single global charge direction so we can inject into multiple blocks
    # with consistent sign.
    dir_to_neg = {}
    dir_to_pos = {}
    for k in steer_blocks:
        vn = directions[k]["to_neg"].float()
        vp = directions[k]["to_pos"].float()
        dir_to_neg[k] = vn / (vn.norm() + 1e-9)
        dir_to_pos[k] = vp / (vp.norm() + 1e-9)

    fst = open(args.out_steer, "w")
    t0 = time.time()
    for idx, e in enumerate(entries):
        # use broken sequence as substrate → hairpin lost at baseline
        seq_b = e.get("seq_broken")
        if not seq_b:
            continue
        hp = tuple(e["hairpin"])
        s1s, s1e, ls, le, s2s, s2e = hp
        strand1 = list(reversed(range(s1s, s1e + 1)))
        strand2 = list(range(s2s, s2e + 1))
        pairs = list(zip(strand1, strand2))[: args.n_pairs]
        pos_i = [p[0] for p in pairs]; pos_j = [p[1] for p in pairs]

        # baseline broken
        outb = esmfold_infer(model, tok, seq_b, num_recycles=args.num_recycles, device=args.device)
        pdbb = output_to_pdb_str(model, outb); pb = Path(args.tmp_dir) / f"{e['pdb']}_{e['chain']}_bkn.pdb"; write_pdb(pdbb, pb)
        ss_b = dssp_ss_string(run_dssp(str(pb)))
        base_d, _ = cross_strand_dist_local(str(pb), hp)
        base_hp = check_hairpin_local(ss_b, hp)

        # steering — apply direction at multiple blocks
        results = {}
        for scale in steer_scales:
            # opposite-charge steering: i toward +, j toward -
            hooks_a = []
            # opposite
            with torch.no_grad():
                from intervene import add_direction_to_s
                # for each block k, add the unit direction at that block
                ctxs = []
                # need to nest context managers. Use ExitStack.
                from contextlib import ExitStack
                with ExitStack() as stack:
                    for k in steer_blocks:
                        stack.enter_context(add_direction_to_s(model, [k], dir_to_pos[k], pos_i, scale=scale))
                        stack.enter_context(add_direction_to_s(model, [k], dir_to_neg[k], pos_j, scale=scale))
                    out_op = esmfold_infer(model, tok, seq_b, num_recycles=args.num_recycles, device=args.device)
                pdb_op = output_to_pdb_str(model, out_op)
                p_op = Path(args.tmp_dir) / f"{e['pdb']}_{e['chain']}_op_{scale}.pdb"; write_pdb(pdb_op, p_op)
                try:
                    ss_op = dssp_ss_string(run_dssp(str(p_op)))
                    d_op, _ = cross_strand_dist_local(str(p_op), hp)
                    hp_op = check_hairpin_local(ss_op, hp)
                except Exception:
                    ss_op = ""; d_op = None; hp_op = -1

                # same-charge steering: both toward +
                with ExitStack() as stack:
                    for k in steer_blocks:
                        stack.enter_context(add_direction_to_s(model, [k], dir_to_pos[k], pos_i + pos_j, scale=scale))
                    out_sm = esmfold_infer(model, tok, seq_b, num_recycles=args.num_recycles, device=args.device)
                pdb_sm = output_to_pdb_str(model, out_sm)
                p_sm = Path(args.tmp_dir) / f"{e['pdb']}_{e['chain']}_sm_{scale}.pdb"; write_pdb(pdb_sm, p_sm)
                try:
                    ss_sm = dssp_ss_string(run_dssp(str(p_sm)))
                    d_sm, _ = cross_strand_dist_local(str(p_sm), hp)
                    hp_sm = check_hairpin_local(ss_sm, hp)
                except Exception:
                    ss_sm = ""; d_sm = None; hp_sm = -1

            results[str(scale)] = {"opp_d": d_op, "opp_hp": hp_op, "opp_plddt": float(out_op["plddt"].mean().item()),
                                    "same_d": d_sm, "same_hp": hp_sm, "same_plddt": float(out_sm["plddt"].mean().item()),
                                    "opp_ss": ss_op, "same_ss": ss_sm}
        entry = {
            "pdb": e["pdb"], "chain": e["chain"], "hairpin": hp,
            "pairs": pairs, "seq_broken": seq_b,
            "base_d": base_d, "base_hp": base_hp, "base_plddt": float(outb["plddt"].mean().item()),
            "steer_blocks": steer_blocks,
            "results": results,
        }
        fst.write(json.dumps(entry) + "\n")
        fst.flush()
        print(f"[STEER {idx+1}/{len(entries)}] {e['pdb']}_{e['chain']} base_d={base_d} base_hp={base_hp} elapsed={time.time()-t0:.1f}s")
        gc.collect(); torch.cuda.empty_cache()
    fst.close()


if __name__ == "__main__":
    main()
