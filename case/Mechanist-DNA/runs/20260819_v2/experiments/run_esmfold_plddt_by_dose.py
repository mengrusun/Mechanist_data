"""
ESMFold pLDDT-by-dose (structural validation the CDN previously blocked).

For each steering dose alpha:
  - pool M2 steered generations (raw DNA) across seeds
  - translate longest ORF (SAME machinery the project used: evo2lib.find_longest_orf)
  - fold with local ESMFold (/mnt/quarkfs/share_model/esmfold_v1)
  - record mean pLDDT (CA B-factor), and (bonus) structural DSSP %H / %E on the folded model

Writes runs/followup_esmfold/plddt_by_dose.json and prints a table.
"""
import os, sys, json, glob, subprocess, tempfile
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import evo2lib as E
import scorer as S

ESMFOLD_PATH = "/mnt/quarkfs/share_model/esmfold_v1"
N_PER_ALPHA = 20          # folds per dose (tractable, spread across seeds)
MIN_AA = 20               # skip too-short ORFs
MAX_AA = 400              # ESMFold memory cap (same as scorer)
OUT_DIR = "runs/followup_esmfold"
os.makedirs(OUT_DIR, exist_ok=True)

def ca_plddt_from_pdb(pdb_str):
    """Mean CA B-factor (=pLDDT, 0-100) over the folded model."""
    vals = []
    for line in pdb_str.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                vals.append(float(line[60:66]))
            except Exception:
                pass
    return float(np.mean(vals)) if vals else float("nan")

def dssp_from_pdb_str(pdb_str):
    """structural %H / %E via mkdssp on the folded model."""
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
            f.write(pdb_str); path = f.name
        r = S.dssp_ss_from_pdb(path)
        os.unlink(path)
        if r is None: return float("nan"), float("nan")
        aas, ss8 = r
        return S.pct_helix_from_ss8(ss8), S.pct_sheet_from_ss8(ss8)
    except Exception:
        return float("nan"), float("nan")

def collect_proteins():
    gens = json.load(open("results/M2_generations.json"))
    # alpha -> list of proteins (translated), spread across seeds
    by_alpha = {}
    # discover alphas from keys aX_sY
    alphas = sorted({float(k.split("_")[0][1:]) for k in gens}, key=float)
    for a in alphas:
        prots = []
        # interleave seeds so the sample spans all 3
        cells = [k for k in gens if abs(float(k.split("_")[0][1:]) - a) < 1e-9]
        pools = [gens[c] for c in sorted(cells)]
        maxlen = max((len(p) for p in pools), default=0)
        for i in range(maxlen):
            for p in pools:
                if i < len(p):
                    dna = p[i]
                    prot, frame, start, orf = E.find_longest_orf(dna, min_aa=MIN_AA)
                    if prot and MIN_AA <= len(prot):
                        prots.append(prot[:MAX_AA])
                if len(prots) >= N_PER_ALPHA:
                    break
            if len(prots) >= N_PER_ALPHA:
                break
        by_alpha[a] = prots
    return alphas, by_alpha

def main():
    from transformers import AutoTokenizer, EsmForProteinFolding
    print("loading ESMFold from", ESMFOLD_PATH, flush=True)
    tok = AutoTokenizer.from_pretrained(ESMFOLD_PATH)
    model = EsmForProteinFolding.from_pretrained(ESMFOLD_PATH, low_cpu_mem_usage=True)
    model = model.cuda().eval()
    model.esm = model.esm.half()
    try:
        model.trunk.set_chunk_size(64)
    except Exception:
        pass

    alphas, by_alpha = collect_proteins()
    print("alphas:", alphas, "| counts:", {a: len(by_alpha[a]) for a in alphas}, flush=True)

    rows = {}
    per_seq = {}
    for a in alphas:
        plddts, hs, es, lens = [], [], [], []
        recs = []
        for j, seq in enumerate(by_alpha[a]):
            try:
                with torch.no_grad():
                    pdb = model.infer_pdb(seq)
                pl = ca_plddt_from_pdb(pdb)
                h, e = dssp_from_pdb_str(pdb)
            except Exception as ex:
                print(f"  [a={a} #{j}] fold fail: {ex}", flush=True)
                continue
            plddts.append(pl); hs.append(h); es.append(e); lens.append(len(seq))
            recs.append({"len": len(seq), "plddt": pl, "struct_pctH": h, "struct_pctE": e})
        def m(x):
            x = [v for v in x if v == v]
            return float(np.mean(x)) if x else float("nan")
        def sd(x):
            x = [v for v in x if v == v]
            return float(np.std(x)) if x else float("nan")
        rows[a] = {
            "n": len(plddts),
            "mean_plddt": m(plddts), "std_plddt": sd(plddts),
            "mean_struct_pctH": m(hs), "mean_struct_pctE": m(es),
            "mean_len": m(lens),
        }
        per_seq[a] = recs
        print(f"alpha={a:>5}: n={rows[a]['n']:>2}  pLDDT={rows[a]['mean_plddt']:.1f}"
              f"  struct%H={rows[a]['mean_struct_pctH']:.1f}  struct%E={rows[a]['mean_struct_pctE']:.1f}"
              f"  len={rows[a]['mean_len']:.0f}", flush=True)

    out = {
        "milestone": "followup_esmfold_plddt_by_dose",
        "esmfold_path": ESMFOLD_PATH,
        "n_per_alpha_target": N_PER_ALPHA, "min_aa": MIN_AA, "max_aa": MAX_AA,
        "by_alpha": {str(a): rows[a] for a in alphas},
        "per_seq": {str(a): per_seq[a] for a in alphas},
    }
    json.dump(out, open(os.path.join(OUT_DIR, "plddt_by_dose.json"), "w"), indent=1)
    print("WROTE", os.path.join(OUT_DIR, "plddt_by_dose.json"), flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
