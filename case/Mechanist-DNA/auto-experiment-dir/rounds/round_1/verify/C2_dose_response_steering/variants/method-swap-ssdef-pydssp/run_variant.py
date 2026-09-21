"""
Verify variant (C2, dimension=method) -- FINAL, EXECUTED design (pivoted from an initial
ESMFold->OmegaFold structure-predictor swap; see PIVOT_NOTE.md for why that swap was abandoned).

Swap: the SECONDARY-STRUCTURE ASSIGNMENT ALGORITHM, not the structure predictor. Everything upstream
of DSSP is held IDENTICAL to the main experiment (same Evo2-7B + frozen feature set S steering hook,
same prompts, same translation/ORF filter, same ESMFold structure prediction, same pLDDT gate). For
each ESMFold-predicted structure, this script computes secondary structure with BOTH:
  (a) mkdssp (the main experiment's tool; full 8-state DSSP, HGI = alpha-helix/3-10-helix/pi-helix), and
  (b) pydssp (github.com/ShintaroMinami/PyDSSP, already installed in the `scientist` env per
      results/setup_report.json -- a from-scratch, differentiable, hydrogen-bond-map reimplementation
      of the DSSP algorithm that outputs a SIMPLIFIED 3-state alphabet {-, H, E} directly from backbone
      coordinates (N, CA, C, O), not by re-deriving mkdssp's own 8-state output).
ON THE SAME PREDICTED STRUCTURE -- this isolates the SS-DEFINITION axis from any structure-prediction
variance, directly testing whether C2's dose-response conclusion is an artifact of DSSP's specific
8-state HGI grouping rule versus a genuinely different (geometric hydrogen-bond-based, 3-state)
secondary-structure assignment algorithm.

No new package installs required (pydssp + mkdssp + ESMFold are all already in the `scientist` env),
so this runs in ONE process/environment, unlike the abandoned OmegaFold path.

Run: CUDA_VISIBLE_DEVICES=<gpu> python run_variant.py --alpha 8 --seed 42 --n 150 \
    --out results_a8_s42.json --seq_out seqs_a8_s42.jsonl
"""
import os, sys, json, argparse, time, subprocess, tempfile
# MUST be set before any transformers/huggingface_hub import (bug caught in sanity pilot: without
# this, ESMFold re-downloads its ~2.7GB checkpoint into ~/.cache/huggingface instead of reusing the
# main experiment's already-cached copy -- matches code/dispatch.py's ENV_BASE["HF_HOME"] setting).
os.environ.setdefault("HF_HOME", "/data/wanghaoxiong/intergene_mechanist_v6/.hf_cache")
sys.path.insert(0, "/data/wanghaoxiong/intergene_mechanist_v6/code")
import numpy as np
import torch
import m0_data as D
from evo2_sae import load_evo2, BatchTopKSAE
import mechanism as M
from m1_harness_calibrate import natural_prompts

import pydssp

MKDSSP = "/data/wanghaoxiong/miniconda3/envs/scientist/bin/mkdssp"
BACKBONE_ATOMS = ["N", "CA", "C", "O"]


def parse_backbone_coords(pdb_string):
    """Extract (L,4,3) N/CA/C/O coords for the first chain from a PDB string, biopython-free
    (avoids chain/model-selection edge cases) -- simple fixed-column PDB ATOM parser."""
    coords_by_res = {}
    first_chain = None
    for line in pdb_string.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        chain = line[21]
        if first_chain is None:
            first_chain = chain
        if chain != first_chain or atom_name not in BACKBONE_ATOMS:
            continue
        resnum = int(line[22:26])
        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        coords_by_res.setdefault(resnum, {})[atom_name] = (x, y, z)
    resnums = sorted(r for r, d in coords_by_res.items() if all(a in d for a in BACKBONE_ATOMS))
    if len(resnums) < 3:
        return None
    coords = np.array([[coords_by_res[r][a] for a in BACKBONE_ATOMS] for r in resnums], dtype=np.float32)
    return coords  # (L, 4, 3)


def dssp_fractions_mkdssp(pdb_string):
    """Same as mechanism.dssp_fractions but takes a string directly (reuse main experiment's own fn)."""
    return M.dssp_fractions(pdb_string)


def dssp_fractions_pydssp(pdb_string):
    coords = parse_backbone_coords(pdb_string)
    if coords is None:
        return None
    try:
        ss = pydssp.assign(coords, out_type="c3")  # array of '-'/'H'/'E' per residue
    except Exception:
        return None
    n = len(ss)
    if n == 0:
        return None
    helix = sum(c == "H" for c in ss) / n   # pydssp's 3-state H (no separate G/I -- see PIVOT_NOTE.md)
    sheet = sum(c == "E" for c in ss) / n
    return {"helix_pydssp": helix, "sheet_pydssp": sheet, "n_resolved": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_set", default="/data/wanghaoxiong/intergene_mechanist_v6/results/m0_feature_set.json")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--n_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--min_aa", type=int, default=30)
    ap.add_argument("--plddt_min", type=float, default=50.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq_out", required=True, help="also save raw sequences (main experiment did not)")
    args = ap.parse_args()
    t0 = time.time()

    fs = json.load(open(args.feature_set))
    org = fs.get("steer_organism", fs.get("organism_primary", "prokaryote"))
    table = D.ORGANISMS[org]["transl_table"]

    evo2 = load_evo2("cuda:0")
    sae = BatchTopKSAE(device="cuda:0")
    steerer = M.Steerer(evo2, sae, fs["helix_features"], fs["s_f"], device="cuda:0")

    prompts, _, _ = natural_prompts(org, args.n)
    prompts = (prompts * (args.n // max(len(prompts), 1) + 1))[:args.n]
    print(f"[variant] alpha={args.alpha} seed={args.seed} n={len(prompts)}", flush=True)

    # PHASE 1: generate + translate (Evo2 only), matching mechanism.generate_and_readout's own
    # two-phase separation (avoids the vortex/flash-attn + ESMFold co-residency kernel-dispatch issue
    # noted in mechanism.py).
    ctx = steerer.steer(args.alpha) if args.alpha != 0.0 else __import__("contextlib").nullcontext()
    gen_recs = []
    with ctx, torch.no_grad():
        for i, prompt in enumerate(prompts):
            rec = {"idx": i, "valid_orf": False, "prot": None}
            try:
                full = M.generate_dna(evo2, prompt, args.n_tokens, args.temperature, args.top_k,
                                      seed=args.seed * 100000 + i)
                gen = full[len(prompt):] if full.startswith(prompt) else full
                prot, valid, meta = M.translate_orf(prompt + gen, min_aa=args.min_aa, table=table)
                rec = {"idx": i, "valid_orf": valid, "len_aa": meta.get("len_aa", 0), "prot": prot if valid else None}
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:150]}"
            gen_recs.append(rec)
    n_valid = sum(r["valid_orf"] for r in gen_recs)
    print(f"[variant] generation done: n_valid_orf={n_valid}/{len(gen_recs)}", flush=True)

    # Save raw sequences (closes the main experiment's sequence-inspectability gap for this subset).
    with open(args.seq_out, "w") as f:
        for r in gen_recs:
            f.write(json.dumps({"idx": r["idx"], "alpha": args.alpha, "seed": args.seed,
                                "valid_orf": r["valid_orf"], "prot": r.get("prot")}) + "\n")

    # PHASE 2: fold with ESMFold (SAME predictor as the main experiment) + run BOTH mkdssp and
    # pydssp on the SAME predicted structure.
    per_sample = []
    for r in gen_recs:
        entry = {"idx": r["idx"], "valid_orf": r["valid_orf"], "gated": True}
        if not r.get("prot"):
            per_sample.append(entry); continue
        try:
            pdb, plddt = M.esmfold_pdb(r["prot"], device="cuda:0")
            entry["plddt"] = plddt
            if pdb is not None and plddt is not None and plddt >= args.plddt_min:
                fr_dssp = dssp_fractions_mkdssp(pdb)
                fr_pyd = dssp_fractions_pydssp(pdb)
                if fr_dssp and fr_pyd:
                    entry.update({
                        "helix_hgi_mkdssp": fr_dssp["helix_hgi"], "sheet_mkdssp": fr_dssp["sheet"],
                        "helix_pydssp": fr_pyd["helix_pydssp"], "sheet_pydssp": fr_pyd["sheet_pydssp"],
                        "n_resolved": fr_dssp["n_resolved"], "gated": False,
                    })
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {str(e)[:150]}"
            torch.cuda.empty_cache()
        per_sample.append(entry)

    gated_pass = [e for e in per_sample if not e["gated"]]

    def agg(key):
        vals = [e[key] for e in gated_pass if key in e]
        return {"mean": float(np.mean(vals)) if vals else None,
                "sem": float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else None,
                "n": len(vals)}

    result = {
        "config": {"alpha": args.alpha, "seed": args.seed, "n": args.n, "plddt_min": args.plddt_min,
                   "predictor": "ESMFold (same as main experiment)",
                   "ss_methods_compared": ["mkdssp (main experiment's tool, 8-state HGI)",
                                           "pydssp (3-state -HE, hydrogen-bond-map reimplementation)"]},
        "n_total": len(gen_recs), "n_valid_orf": n_valid, "n_gated_pass": len(gated_pass),
        "valid_orf_rate": n_valid / max(len(gen_recs), 1),
        "gated_pass_rate": len(gated_pass) / max(len(gen_recs), 1),
        "helix_hgi_mkdssp": agg("helix_hgi_mkdssp"), "sheet_mkdssp": agg("sheet_mkdssp"),
        "helix_pydssp": agg("helix_pydssp"), "sheet_pydssp": agg("sheet_pydssp"),
        "per_sample": per_sample, "elapsed_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"[variant] wrote {args.out}: helix_mkdssp={result['helix_hgi_mkdssp']['mean']} "
          f"helix_pydssp={result['helix_pydssp']['mean']} gated_pass_rate={result['gated_pass_rate']:.2f} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
    import os as _os; sys.stdout.flush(); _os._exit(0)  # force GPU release
