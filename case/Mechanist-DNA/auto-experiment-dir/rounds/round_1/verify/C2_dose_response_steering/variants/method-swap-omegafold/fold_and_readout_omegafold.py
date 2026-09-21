"""
Verify variant (C2, dimension=method): swap the structure predictor ESMFold -> OmegaFold.
STAGE 2/2 (runs in the `verify_omegafold` conda env -- OmegaFold is PyTorch-based but installed in
an isolated env to avoid touching the shared `scientist` env's pinned numpy<2/pandas/biopython ABI).

Reads a JSONL of {idx, alpha, seed, valid_orf, prot} records written by gen_sequences.py (run in the
`scientist` env), folds each valid protein with OmegaFold (one pdb per sequence, confidence stored in
the B-factor column, mirroring mechanism.py's ESMFold+pLDDT readout), runs mkdssp (binary called by
full path -- no conda env needed for the compiled tool) for per-residue secondary structure, and
computes the SAME aggregate readout as code/mechanism.py's aggregate(): valid_orf_rate, gated_pass_rate
(confidence >= conf_min gate), helix_hgi_mean/sem, sheet_mean, mean confidence.

Run (verify_omegafold env): python fold_and_readout_omegafold.py --in seqs_a8_s42.jsonl \
    --out results_omegafold_a8_s42.json --conf_min <calibrated threshold>
"""
import os, sys, json, argparse, time, subprocess, tempfile, glob, re, shutil
import numpy as np

MKDSSP = "/data/wanghaoxiong/miniconda3/envs/scientist/bin/mkdssp"
HELIX_HGI = set("HGI")
SHEET = set("E")
SEQNAME_RE = re.compile(r"^seq(\d+)$")  # strict: matches ONLY "seq<idx>", not "seq<idx>_anything"


def parse_confidence_bfactor(pdb_path):
    """Mean of the B-factor column (OmegaFold's per-residue confidence), CA atoms only."""
    vals = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    vals.append(float(line[60:66]))
                except ValueError:
                    pass
    return float(np.mean(vals)) if vals else None


def dssp_fractions(pdb_path):
    out = pdb_path + ".dssp"
    r = subprocess.run([MKDSSP, pdb_path, out], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        r = subprocess.run([MKDSSP, "--output-format", "dssp", pdb_path, out], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        return None
    from Bio.PDB.DSSP import make_dssp_dict
    try:
        d, keys = make_dssp_dict(out)
    except Exception:
        return None
    ss = [d[k][1] for k in keys]
    n = len(ss)
    if n == 0:
        return None
    return {"helix_hgi": sum(c in HELIX_HGI for c in ss) / n,
            "sheet": sum(c in SHEET for c in ss) / n, "n_resolved": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf_min", type=float, required=True,
                    help="OmegaFold confidence gate -- MUST be calibrated empirically against a pilot "
                         "run (see calibrate_conf_gate.py) to match the main experiment's ESMFold "
                         "pLDDT-gate ACCEPTANCE RATE at alpha=0, not guessed (code-review fix: no "
                         "default, since OmegaFold's confidence scale is not known a priori to match "
                         "ESMFold's 0-100 pLDDT scale).")
    ap.add_argument("--subbatch_size", type=int, default=256)  # compute/memory-only, no effect on output
    ap.add_argument("--max_len", type=int, default=400,
                    help="matches mechanism.py's esmfold_pdb() max_len=400 exactly -- same truncation "
                         "policy as the main experiment, not an independent choice for this variant.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    t0 = time.time()

    recs = [json.loads(l) for l in open(args.inp)]
    valid = [r for r in recs if r.get("valid_orf") and r.get("prot")]
    n_total = len(recs); n_valid = len(valid)
    print(f"[fold] {args.inp}: n_total={n_total} n_valid_orf={n_valid}", flush=True)

    tmpdir = tempfile.mkdtemp(prefix="omegafold_")
    try:
        fasta_path = os.path.join(tmpdir, "in.fasta")
        with open(fasta_path, "w") as f:
            for r in valid:
                prot = r["prot"][: args.max_len]
                f.write(f">seq{r['idx']}\n{prot}\n")

        outdir = os.path.join(tmpdir, "out")
        os.makedirs(outdir, exist_ok=True)
        cmd = ["omegafold", fasta_path, outdir, "--subbatch_size", str(args.subbatch_size),
               "--device", args.device]
        print(f"[fold] running: {' '.join(cmd)}", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("[fold] OMEGAFOLD STDERR:", r.stderr[-3000:], flush=True)

        # Track EXPECTED (every valid-ORF input) vs OBSERVED (every pdb OmegaFold actually produced)
        # so a silent OmegaFold-side failure/skip doesn't just vanish from the denominator (code-review fix).
        expected = {r["idx"]: {"idx": r["idx"], "conf": None, "fold_ok": False, "gated": True} for r in valid}
        n_pdb_found = 0
        for pdb_path in sorted(glob.glob(os.path.join(outdir, "*.pdb"))):
            name = os.path.splitext(os.path.basename(pdb_path))[0]
            m = SEQNAME_RE.match(name)
            if not m:
                print(f"[fold] WARNING: unrecognized output filename, skipping: {pdb_path}", flush=True)
                continue
            idx = int(m.group(1))
            if idx not in expected:
                print(f"[fold] WARNING: pdb idx={idx} has no matching input record, skipping", flush=True)
                continue
            n_pdb_found += 1
            conf = parse_confidence_bfactor(pdb_path)
            expected[idx]["conf"] = conf
            expected[idx]["fold_ok"] = conf is not None
            if conf is not None and conf >= args.conf_min:
                fr = dssp_fractions(pdb_path)
                if fr:
                    expected[idx].update({"helix_hgi": fr["helix_hgi"], "sheet": fr["sheet"],
                                          "n_resolved": fr["n_resolved"], "gated": False})

        per_sample = [expected[idx] for idx in sorted(expected)]
        n_folded = sum(1 for e in per_sample if e["fold_ok"])
        n_missing_pdb = n_valid - n_pdb_found
        conf_vals = [e["conf"] for e in per_sample if e["conf"] is not None]
        helix_vals = [e["helix_hgi"] for e in per_sample if "helix_hgi" in e]
        sheet_vals = [e["sheet"] for e in per_sample if "sheet" in e]
        if n_missing_pdb > 0:
            print(f"[fold] WARNING: {n_missing_pdb}/{n_valid} valid-ORF inputs never produced a pdb "
                  f"(OmegaFold skip/crash) -- counted as fold failures, NOT silently dropped.", flush=True)

        agg = {
            "n": n_total, "n_valid_orf": n_valid, "n_folded": n_folded, "n_missing_pdb": n_missing_pdb,
            "n_gated_pass": len(helix_vals),
            "valid_orf_rate": n_valid / max(n_total, 1),
            "gated_pass_rate": len(helix_vals) / max(n_total, 1),
            "conf_mean": float(np.mean(conf_vals)) if conf_vals else None,
            "helix_hgi_mean": float(np.mean(helix_vals)) if helix_vals else None,
            "helix_hgi_sem": float(np.std(helix_vals) / np.sqrt(len(helix_vals))) if len(helix_vals) > 1 else None,
            "sheet_mean": float(np.mean(sheet_vals)) if sheet_vals else None,
        }
        result = {"config": {"conf_min": args.conf_min, "n_total": n_total}, "aggregate": agg,
                  "per_sample": per_sample, "elapsed_s": time.time() - t0}
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)
        print(f"[fold] wrote {args.out}: helix_mean={agg['helix_hgi_mean']} sheet_mean={agg['sheet_mean']} "
              f"conf_mean={agg['conf_mean']} gated_pass_rate={agg['gated_pass_rate']:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
