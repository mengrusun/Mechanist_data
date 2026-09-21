"""Sanity check: load ESMFold, predict PGB1 β-hairpin, run DSSP."""
import os, time, torch
from common import (load_esmfold, esmfold_infer, output_to_pdb_str,
                    run_dssp, dssp_ss_string, find_hairpins_in_ss, write_pdb)

def main():
    t0 = time.time()
    tok, model = load_esmfold(device="cuda:0", dtype=torch.float32)
    print(f"loaded {time.time()-t0:.1f}s")
    seq = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"
    t0 = time.time()
    out = esmfold_infer(model, tok, seq, num_recycles=1)
    print(f"infer {time.time()-t0:.1f}s")
    print("plddt mean:", out['plddt'].mean().item())
    pdb = output_to_pdb_str(model, out)
    write_pdb(pdb, "/tmp/pgb1.pdb")
    dssp = run_dssp("/tmp/pgb1.pdb")
    ss = dssp_ss_string(dssp)
    print("SS:", ss)
    print("hairpins:", find_hairpins_in_ss(ss))

if __name__ == "__main__":
    main()
