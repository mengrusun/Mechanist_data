"""Claim 2: seq2pair vs pair2seq ablation by layer range.

For each (native) chain that folds to a β-hairpin at baseline:
  Ablate seq2pair (zero out its contribution) in early blocks [0, K_early]
  vs late blocks [K_late, 47]. Do the same for pair2seq. Compare with baseline
  and full ablation. Measure DSSP β-hairpin outcome and pLDDT.

Writes: outputs/claim2_pathway.jsonl
"""
import argparse
import json
import os
import time
from pathlib import Path
import gc

import torch

from common import (load_esmfold, esmfold_infer, output_to_pdb_str,
                    run_dssp, dssp_ss_string, write_pdb, OUT_DIR)
from intervene import ablate_seq2pair, ablate_pair2seq


def check_hairpin(ss, hp, tol_shift=2, min_e=2):
    s1s, s1e, ls, le, s2s, s2e = hp
    e_in_s1 = sum(1 for c in ss[max(0, s1s - tol_shift): s1e + 1 + tol_shift] if c == 'E')
    e_in_s2 = sum(1 for c in ss[max(0, s2s - tol_shift): s2e + 1 + tol_shift] if c == 'E')
    return int((e_in_s1 >= min_e) and (e_in_s2 >= min_e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(OUT_DIR / "baseline.jsonl"))
    ap.add_argument("--out", default=str(OUT_DIR / "claim2_pathway.jsonl"))
    ap.add_argument("--tmp_dir", default="/tmp/claim2_pdb")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--num_recycles", type=int, default=1)
    ap.add_argument("--max_chains", type=int, default=25)
    # windows to ablate (start,end) inclusive
    ap.add_argument("--windows", default="0-7,8-15,16-23,24-31,32-39,40-47,0-11,12-23,24-35,36-47,0-47")
    args = ap.parse_args()

    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    entries = []
    with open(args.baseline) as f:
        for ln in f:
            e = json.loads(ln)
            if e.get("native_hp_ok"):
                entries.append(e)
    entries = entries[: args.max_chains]
    print(f"chains to test: {len(entries)}")

    def parse_windows(s):
        out = []
        for tok in s.split(","):
            a, b = tok.split("-")
            out.append((int(a), int(b)))
        return out
    wins = parse_windows(args.windows)

    tok, model = load_esmfold(device=args.device, dtype=torch.float32)

    fo = open(args.out, "w")
    t0 = time.time()
    for idx, e in enumerate(entries):
        seq = e["seq_native"]
        hp = tuple(e["hairpin"])
        # baseline
        out_base = esmfold_infer(model, tok, seq, num_recycles=args.num_recycles, device=args.device)
        pdb_base = output_to_pdb_str(model, out_base)
        pb = Path(args.tmp_dir) / f"{e['pdb']}_{e['chain']}_base.pdb"
        write_pdb(pdb_base, pb)
        ss_base = dssp_ss_string(run_dssp(str(pb)))
        base_hp = check_hairpin(ss_base, hp)

        results = {}
        for (a, b) in wins:
            wname = f"{a}-{b}"
            rng = list(range(a, b + 1))
            # seq2pair ablation
            with ablate_seq2pair(model, rng):
                out_s2p = esmfold_infer(model, tok, seq, num_recycles=args.num_recycles, device=args.device)
            pdb_s2p = output_to_pdb_str(model, out_s2p)
            p_s2p = Path(args.tmp_dir) / f"{e['pdb']}_{e['chain']}_s2p_{wname}.pdb"
            write_pdb(pdb_s2p, p_s2p)
            try:
                ss_s2p = dssp_ss_string(run_dssp(str(p_s2p)))
                hp_s2p = check_hairpin(ss_s2p, hp)
            except Exception:
                ss_s2p = ""; hp_s2p = -1
            # pair2seq ablation
            with ablate_pair2seq(model, rng):
                out_p2s = esmfold_infer(model, tok, seq, num_recycles=args.num_recycles, device=args.device)
            pdb_p2s = output_to_pdb_str(model, out_p2s)
            p_p2s = Path(args.tmp_dir) / f"{e['pdb']}_{e['chain']}_p2s_{wname}.pdb"
            write_pdb(pdb_p2s, p_p2s)
            try:
                ss_p2s = dssp_ss_string(run_dssp(str(p_p2s)))
                hp_p2s = check_hairpin(ss_p2s, hp)
            except Exception:
                ss_p2s = ""; hp_p2s = -1

            results[wname] = {
                "s2p_hp": hp_s2p, "s2p_plddt": float(out_s2p["plddt"].mean().item()), "s2p_ss": ss_s2p,
                "p2s_hp": hp_p2s, "p2s_plddt": float(out_p2s["plddt"].mean().item()), "p2s_ss": ss_p2s,
            }
        entry = {
            "pdb": e["pdb"], "chain": e["chain"], "hairpin": hp,
            "seq": seq, "base_hp": base_hp, "base_plddt": float(out_base["plddt"].mean().item()),
            "ss_base": ss_base, "windows": results,
        }
        fo.write(json.dumps(entry) + "\n")
        fo.flush()
        print(f"[{idx+1}/{len(entries)}] {e['pdb']}_{e['chain']} elapsed={time.time()-t0:.1f}s")
        gc.collect(); torch.cuda.empty_cache()

    fo.close()


if __name__ == "__main__":
    main()
