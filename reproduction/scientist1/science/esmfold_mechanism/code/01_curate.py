"""Curate dataset:
- Parse PISCES cull list; keep length in [40, 100] and PDB present in cache.
- Run DSSP on ground-truth PDBs; extract sequence + SS for the chain.
- Detect β-hairpin motifs (2 anti-parallel E-strands separated by 2-6 loop residues).
- Save a JSONL with one entry per chain containing hairpin windows.
"""
import os
import json
import time
import argparse
from pathlib import Path

from common import (parse_pisces, pisces_pdb_path, read_pdb_sequence_and_positions,
                    run_dssp, dssp_ss_string, find_hairpins_in_ss,
                    PISCES_LIST, DATA_DIR, AA3TO1)


def has_pair_charge(seq, hp):
    """Return count of oppositely-charged pairs across the two strands when aligned outward from the loop.
    hp = (s1s, s1e, ls, le, s2s, s2e)"""
    from common import aa_charge
    s1s, s1e, ls, le, s2s, s2e = hp
    strand1 = list(range(s1s, s1e + 1))
    strand2 = list(range(s2s, s2e + 1))
    # Align from loop outward: positions closest to loop pair up
    # strand1: from s1e (closest to loop) backwards
    # strand2: from s2s (closest to loop) forwards
    strand1_ordered = list(reversed(strand1))
    strand2_ordered = list(strand2)
    pairs = list(zip(strand1_ordered, strand2_ordered))
    charged_pairs = 0
    opp_pairs = 0
    same_pairs = 0
    for i, j in pairs:
        c1 = aa_charge(seq[i]); c2 = aa_charge(seq[j])
        if c1 != 0 and c2 != 0:
            charged_pairs += 1
            if c1 * c2 < 0:
                opp_pairs += 1
            else:
                same_pairs += 1
    return len(pairs), charged_pairs, opp_pairs, same_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_len", type=int, default=40)
    ap.add_argument("--max_len", type=int, default=100)
    ap.add_argument("--max_chains", type=int, default=1500,
                    help="Cap number of chains examined")
    ap.add_argument("--out", default=str(DATA_DIR / "hairpin_chains.jsonl"))
    args = ap.parse_args()

    rows = parse_pisces(PISCES_LIST)
    # length filter
    rows = [r for r in rows if args.min_len <= r["length"] <= args.max_len]
    # pdb-cache filter
    rows = [r for r in rows if os.path.exists(pisces_pdb_path(r["pdb"]))]
    print(f"After filters: {len(rows)} candidate chains")

    if args.max_chains and len(rows) > args.max_chains:
        rows = rows[: args.max_chains]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_hairpin = 0
    n_ok = 0
    with open(args.out, "w") as fo:
        t0 = time.time()
        for k, r in enumerate(rows):
            pdb_path = pisces_pdb_path(r["pdb"])
            seq, resids = read_pdb_sequence_and_positions(pdb_path, r["chain"])
            if seq is None or len(seq) < args.min_len:
                continue
            # Truncate to chain length limit for prediction feasibility
            try:
                dssp_out = run_dssp(pdb_path)
            except Exception as e:
                continue
            # DSSP walks all chains; we need per-chain slice. Re-run DSSP just to get keys per chain.
            # Simpler: get SS from DSSP records restricted to the target chain via key lookup.
            # But our run_dssp returns idx over all residues. Redo with chain filter.
            try:
                from Bio.PDB import PDBParser
                from Bio.PDB.DSSP import DSSP
                parser = PDBParser(QUIET=True)
                st = parser.get_structure("x", pdb_path)
                mod = st[0]
                dssp = DSSP(mod, pdb_path, dssp="mkdssp")
                ss_chars = []
                aa_chars = []
                for res in mod[r["chain"]]:
                    if res.id[0] != " ":
                        continue
                    rn = res.get_resname()
                    if rn not in AA3TO1:
                        continue
                    if "CA" not in res:
                        continue
                    key = (r["chain"], res.id)
                    if key in dssp.keys():
                        aa_chars.append(dssp[key][1])
                        ss_chars.append(dssp[key][2])
                    else:
                        aa_chars.append(AA3TO1[rn])
                        ss_chars.append("-")
                ss = "".join(ss_chars)
                seq2 = "".join(aa_chars)
            except Exception as e:
                continue

            # Use seq2 (from DSSP-aligned residues) as canonical
            if len(seq2) != len(seq):
                # keep whichever matches
                seq = seq2
            hairpins = find_hairpins_in_ss(ss)
            if not hairpins:
                continue
            # Filter hairpins to those inside sequence range
            hairpins_ok = [hp for hp in hairpins if hp[5] < len(seq)]
            if not hairpins_ok:
                continue

            entry = {
                "pdb": r["pdb"], "chain": r["chain"], "length": len(seq),
                "seq": seq, "ss": ss, "hairpins": hairpins_ok,
            }
            fo.write(json.dumps(entry) + "\n")
            n_hairpin += 1
            n_ok += 1
            if (k + 1) % 50 == 0:
                print(f"[{k+1}/{len(rows)}] hairpin_chains={n_hairpin} elapsed={time.time()-t0:.1f}s")
    print(f"Done. Chains with hairpin: {n_hairpin}. Written to {args.out}")


if __name__ == "__main__":
    main()
