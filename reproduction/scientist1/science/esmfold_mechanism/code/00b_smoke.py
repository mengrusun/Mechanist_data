"""Smoke test: pick one small hairpin chain, run baseline + all intervention types.
Prints diagnostic output to verify hooks work correctly.
"""
import json, os, time, torch
from pathlib import Path

from common import (load_esmfold, esmfold_infer, output_to_pdb_str,
                    run_dssp, dssp_ss_string, find_hairpins_in_ss, write_pdb)
from intervene import (capture_block_outputs, patch_s_at_block,
                       patch_z_at_block, ablate_seq2pair, ablate_pair2seq,
                       add_direction_to_s)

DATA = "/data/zhenqian/Reproduction1/cc/science/esmfold_mechanism/data/hairpin_chains.jsonl"


def check_hairpin(ss, hp, tol_shift=2, min_e=2):
    s1s, s1e, ls, le, s2s, s2e = hp
    e_in_s1 = sum(1 for c in ss[max(0, s1s - tol_shift): s1e + 1 + tol_shift] if c == 'E')
    e_in_s2 = sum(1 for c in ss[max(0, s2s - tol_shift): s2e + 1 + tol_shift] if c == 'E')
    return int((e_in_s1 >= min_e) and (e_in_s2 >= min_e))


def main():
    # load a small chain
    entries = []
    with open(DATA) as f:
        for ln in f:
            e = json.loads(ln)
            if 40 <= e["length"] <= 65:
                entries.append(e)
    e = entries[0]
    seq = e["seq"]
    hp = e["hairpins"][0]
    print("PDB:", e["pdb"], e["chain"], "len:", len(seq))
    print("hairpin:", hp)
    print("SS gt:", e["ss"])

    tok, model = load_esmfold(device="cuda:0")

    # baseline
    t0 = time.time()
    out = esmfold_infer(model, tok, seq, num_recycles=1)
    pdb = output_to_pdb_str(model, out); write_pdb(pdb, "/tmp/s_base.pdb")
    ss = dssp_ss_string(run_dssp("/tmp/s_base.pdb"))
    print(f"baseline infer {time.time()-t0:.1f}s; SS pred: {ss}; hp_ok={check_hairpin(ss, hp)}")

    # broken sequence
    seq_b = list(seq)
    for i in range(hp[0], hp[5] + 1):
        seq_b[i] = "G"
    seq_b = "".join(seq_b)
    out_b = esmfold_infer(model, tok, seq_b, num_recycles=1)
    pdbb = output_to_pdb_str(model, out_b); write_pdb(pdbb, "/tmp/s_broken.pdb")
    ssb = dssp_ss_string(run_dssp("/tmp/s_broken.pdb"))
    print(f"broken SS: {ssb}; hp_ok={check_hairpin(ssb, hp)}")

    # capture native s,z
    cache_n = {}
    with capture_block_outputs(model, cache_n):
        _ = esmfold_infer(model, tok, seq, num_recycles=1)
    print(f"captured {len(cache_n)} blocks; s shape {cache_n[0][0].shape}; z shape {cache_n[0][1].shape}")

    positions = list(range(hp[0], hp[5] + 1))
    # patch s at block 4 on broken
    src_s = cache_n[4][0][:, positions, :]
    with patch_s_at_block(model, 4, src_s, positions=positions):
        out_ps = esmfold_infer(model, tok, seq_b, num_recycles=1)
    pdb_ps = output_to_pdb_str(model, out_ps); write_pdb(pdb_ps, "/tmp/s_ps.pdb")
    ss_ps = dssp_ss_string(run_dssp("/tmp/s_ps.pdb"))
    print(f"patch-s(block=4) SS: {ss_ps}; hp_ok={check_hairpin(ss_ps, hp)}")

    src_z = cache_n[4][1][:, positions, :, :][:, :, positions, :]
    with patch_z_at_block(model, 4, src_z, pair_positions=positions):
        out_pz = esmfold_infer(model, tok, seq_b, num_recycles=1)
    pdb_pz = output_to_pdb_str(model, out_pz); write_pdb(pdb_pz, "/tmp/s_pz.pdb")
    ss_pz = dssp_ss_string(run_dssp("/tmp/s_pz.pdb"))
    print(f"patch-z(block=4) SS: {ss_pz}; hp_ok={check_hairpin(ss_pz, hp)}")

    # ablate seq2pair in early blocks 0..7 for native
    with ablate_seq2pair(model, range(0, 8)):
        out_a = esmfold_infer(model, tok, seq, num_recycles=1)
    pdb_a = output_to_pdb_str(model, out_a); write_pdb(pdb_a, "/tmp/s_ablate_s2p_early.pdb")
    ss_a = dssp_ss_string(run_dssp("/tmp/s_ablate_s2p_early.pdb"))
    print(f"ablate seq2pair blocks 0-7: SS {ss_a}; hp_ok={check_hairpin(ss_a, hp)}")

    with ablate_seq2pair(model, range(40, 48)):
        out_l = esmfold_infer(model, tok, seq, num_recycles=1)
    pdb_l = output_to_pdb_str(model, out_l); write_pdb(pdb_l, "/tmp/s_ablate_s2p_late.pdb")
    ss_l = dssp_ss_string(run_dssp("/tmp/s_ablate_s2p_late.pdb"))
    print(f"ablate seq2pair blocks 40-47: SS {ss_l}; hp_ok={check_hairpin(ss_l, hp)}")

    with ablate_pair2seq(model, range(0, 8)):
        out_p = esmfold_infer(model, tok, seq, num_recycles=1)
    pdb_p = output_to_pdb_str(model, out_p); write_pdb(pdb_p, "/tmp/s_ablate_p2s_early.pdb")
    ss_p = dssp_ss_string(run_dssp("/tmp/s_ablate_p2s_early.pdb"))
    print(f"ablate pair2seq blocks 0-7: SS {ss_p}; hp_ok={check_hairpin(ss_p, hp)}")

if __name__ == "__main__":
    main()
