"""
Round-2 dual-predictor structure readout: ESMFold (default) + OmegaFold (independent second
predictor), both feeding a per-residue pLDDT-aware DSSP readout.

Why two predictors (plan improvement 2 / P3): ESMFold and OmegaFold are trained differently
(ESMFold = ESM2 LM trunk + folding head; OmegaFold = its own OmegaPLM, no MSA). Agreement across
them is the robustness axis for C2/C3. Generation/ORF/c* are frozen WITHOUT the second predictor;
both predictors then run once on the SAME held-out proteins.

Primary endpoint (plan P1): pLDDT-WEIGHTED alpha-helix fraction
    helix_w = sum_res plddt_res * 1[ss_res in HGI] / sum_res plddt_res
Confidence folded in as a weight, not a filter. Hard-gated fraction + threshold sweep {50,60,70,80}
are SENSITIVITY analyses only (never the headline).

Both predictors write per-residue confidence into the PDB CA b-factor on a 0-100 scale
(ESMFold: output_to_pdb writes pLDDT; OmegaFold: save_pdb writes confidence*100), so a single
b-factor parse yields per-residue pLDDT aligned to DSSP by residue number.
"""
import os, sys, math, subprocess, tempfile, io
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

HELIX_HGI = set("HGI")
HELIX_H = set("H")
SHEET = set("E")
PLDDT_GRID = (50.0, 60.0, 70.0, 80.0)
DSSP_TMP = "/data/wanghaoxiong/intergene_mechanist_v6/data/dssp"
OMEGAFOLD_REPO = "/data/wanghaoxiong/intergene_mechanist_v6/third_party/OmegaFold"
OMEGAFOLD_WEIGHTS = os.path.expanduser("~/.cache/omegafold_ckpt/model.pt")  # release2 (model 2)

# ======================================================================================
# ESMFold
# ======================================================================================
_esmfold = {"model": None, "tok": None}


def load_esmfold(device="cuda:0"):
    if _esmfold["model"] is None:
        from transformers import EsmForProteinFolding, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").eval().to(device)
        model.esm = model.esm.half()
        model.trunk.set_chunk_size(64)
        _esmfold["model"] = model
        _esmfold["tok"] = tok
    return _esmfold["model"], _esmfold["tok"]


@torch.no_grad()
def esmfold_pdb(prot, device="cuda:0", max_len=400):
    """Return (pdb_string, mean_plddt_0_100) or (None, None)."""
    model, tok = load_esmfold(device)
    prot = prot[:max_len]
    if len(prot) < 20:
        return None, None
    ids = tok([prot], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    out = model(ids)
    plddt = float(out["plddt"][0, :, 1].mean().item()) if "plddt" in out else float(out.plddt.mean())
    if plddt <= 1.5:
        plddt *= 100.0
    pdb = model.output_to_pdb(out)[0]
    return pdb, plddt


# ======================================================================================
# OmegaFold (second, independent predictor) -- programmatic wrapper over the repo API.
# Bypasses the pip installer (its setup.py rejects py3.11); repo lives on PYTHONPATH.
# ======================================================================================
_omegafold = {"model": None, "fwd": None, "rc": None, "save_pdb": None, "recursive_to": None}


def load_omegafold(device="cuda:0", num_cycle=4):
    if _omegafold["model"] is None:
        if OMEGAFOLD_REPO not in sys.path:
            sys.path.insert(0, OMEGAFOLD_REPO)
        import argparse
        import omegafold as of
        from omegafold import pipeline
        from omegafold import utils as of_utils
        from omegafold.utils.protein_utils import residue_constants as rc
        sd = torch.load(OMEGAFOLD_WEIGHTS, map_location="cpu")
        sd = sd.pop("model", sd)
        model = of.OmegaFold(of.make_config(2))  # release2 == config 2
        model.load_state_dict(sd)
        model.eval().to(device)
        _omegafold["model"] = model
        _omegafold["fwd"] = argparse.Namespace(subbatch_size=None, num_recycle=num_cycle)
        _omegafold["rc"] = rc
        _omegafold["save_pdb"] = pipeline.save_pdb
        _omegafold["recursive_to"] = of_utils.recursive_to
        _omegafold["num_cycle"] = num_cycle
    return _omegafold["model"]


def _omegafold_make_input(prot, device, num_cycle=4, num_pseudo_msa=15, mask_rate=0.12):
    """Replicate pipeline.fasta2inputs for a single sequence (deterministic pseudo-MSA)."""
    rc = _omegafold["rc"]
    fas = prot.upper().replace("Z", "E").replace("B", "D").replace("U", "C")
    fas = "".join(c if c in rc.restypes_with_x else "X" for c in fas)
    aatype = torch.LongTensor(
        [rc.restypes_with_x.index(aa) if aa in rc.restypes_with_x else 20 for aa in fas]
    )
    mask = torch.ones_like(aatype).float()
    num_res = len(aatype)
    g = torch.Generator().manual_seed(num_res)
    data = []
    for _ in range(num_cycle):
        p_msa = aatype[None, :].repeat(num_pseudo_msa, 1)
        p_msa_mask = torch.rand([num_pseudo_msa, num_res], generator=g).gt(mask_rate)
        p_msa_mask = torch.cat((mask[None, :], p_msa_mask), dim=0)
        p_msa = torch.cat((aatype[None, :], p_msa), dim=0)
        p_msa[~p_msa_mask.bool()] = 21
        data.append({"p_msa": p_msa, "p_msa_mask": p_msa_mask})
    return _omegafold["recursive_to"](data, device=device)


@torch.no_grad()
def omegafold_pdb(prot, device="cuda:0", max_len=400, num_cycle=None):
    """Return (pdb_string, mean_plddt_0_100) or (None, None)."""
    load_omegafold(device, num_cycle=num_cycle or _omegafold.get("num_cycle", 4))
    prot = prot[:max_len]
    if len(prot) < 20:
        return None, None
    nc = num_cycle or _omegafold["num_cycle"]
    inp = _omegafold_make_input(prot, device, num_cycle=nc)
    try:
        out = _omegafold["model"](inp, predict_with_confidence=True, fwd_cfg=_omegafold["fwd"])
    except RuntimeError as e:
        torch.cuda.empty_cache()
        return None, None
    conf = out["confidence"]  # 0-1 per residue
    mean_plddt = float(conf.mean().item()) * 100.0
    # write PDB to a string buffer via a temp file (save_pdb wants a path)
    with tempfile.NamedTemporaryFile("r", suffix=".pdb", delete=False, dir=DSSP_TMP) as f:
        path = f.name
    try:
        _omegafold["save_pdb"](
            pos14=out["final_atom_positions"],
            b_factors=out["confidence"] * 100,
            sequence=inp[0]["p_msa"][0],
            mask=inp[0]["p_msa_mask"][0],
            save_path=path, model=0,
        )
        pdb = open(path).read()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
    del out
    torch.cuda.empty_cache()
    return pdb, mean_plddt


# ======================================================================================
# Per-residue readout: DSSP secondary structure + per-residue pLDDT from PDB b-factor
# ======================================================================================
def _plddt_by_resnum(pdb_string):
    """Parse per-residue pLDDT from CA-atom b-factors, keyed by (chain, resnum, icode)."""
    out = {}
    for line in pdb_string.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            try:
                chain = line[21]
                resnum = int(line[22:26])
                icode = line[26]
                out[(chain, resnum, icode)] = float(line[60:66])
            except Exception:
                continue
    return out


def _dssp_by_resnum(pdb_string):
    """Run mkdssp; return {(chain, resnum, icode): ss_char}."""
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False, dir=DSSP_TMP) as f:
        f.write(pdb_string)
        path = f.name
    outp = path.replace(".pdb", ".dssp")
    try:
        r = subprocess.run(["mkdssp", path, outp], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(outp):
            r = subprocess.run(["mkdssp", "--output-format", "dssp", path, outp],
                               capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(outp):
            return None
        from Bio.PDB.DSSP import make_dssp_dict
        d, keys = make_dssp_dict(outp)
        ss = {}
        for k in keys:
            chain, (het, resnum, icode) = k
            c = d[k][1]
            ic = icode if isinstance(icode, str) else " "
            ss[(chain, int(resnum), ic)] = "-" if c in ("-", " ") else c
        return ss
    finally:
        for p in (path, outp):
            try:
                os.remove(p)
            except Exception:
                pass


def structural_readout(pdb_string, plddt_grid=PLDDT_GRID):
    """Combine DSSP SS + per-residue pLDDT into the round-2 endpoint family.

    Returns dict with, per structure:
      helix_hgi, helix_h, sheet                 -- unweighted resolved fractions
      helix_hgi_w, sheet_w                       -- PRIMARY: pLDDT-weighted fractions
      helix_hgi_gate{T}                          -- hard-gated fraction over residues plddt>=T
      n_resolved, n_matched, mean_plddt
    Returns None if DSSP failed or the SS<->pLDDT join is empty (never fabricates weights --
    the pLDDT-weighted primary endpoint must use real per-residue confidence).
    """
    ss = _dssp_by_resnum(pdb_string)
    if not ss:
        return None
    pl = _plddt_by_resnum(pdb_string)
    reskeys = sorted((k for k in ss if k in pl), key=lambda t: (t[0], t[1], t[2]))
    if len(reskeys) == 0:
        return None
    ss_list = [ss[k] for k in reskeys]
    w = np.array([pl[k] for k in reskeys], dtype=float)
    if float(w.sum()) <= 0:
        return None
    resnums = reskeys
    h_hgi = np.array([1.0 if c in HELIX_HGI else 0.0 for c in ss_list])
    h_h = np.array([1.0 if c in HELIX_H else 0.0 for c in ss_list])
    sh = np.array([1.0 if c in SHEET else 0.0 for c in ss_list])
    n = len(resnums)
    wsum = float(w.sum()) if w.sum() > 0 else 1.0
    out = {
        "n_resolved": n,
        "n_matched": n,
        "mean_plddt": float(w.mean()),
        "helix_hgi": float(h_hgi.mean()),
        "helix_h": float(h_h.mean()),
        "sheet": float(sh.mean()),
        "helix_hgi_w": float((w * h_hgi).sum() / wsum),   # PRIMARY endpoint
        "helix_h_w": float((w * h_h).sum() / wsum),
        "sheet_w": float((w * sh).sum() / wsum),
    }
    for T in plddt_grid:
        keep = w >= T
        k = int(keep.sum())
        out[f"helix_hgi_gate{int(T)}"] = float(h_hgi[keep].mean()) if k > 0 else None
        out[f"sheet_gate{int(T)}"] = float(sh[keep].mean()) if k > 0 else None
        out[f"n_gate{int(T)}"] = k
    return out


PREDICTORS = {
    "esmfold": esmfold_pdb,
    "omegafold": omegafold_pdb,
}


def fold_and_read(prot, predictor="esmfold", device="cuda:0", max_len=400, num_cycle=None,
                  plddt_grid=PLDDT_GRID):
    """One protein -> one structural readout dict under the chosen predictor (adds 'predictor',
    'struct_mean_plddt'). Returns None on fold/DSSP failure."""
    fn = PREDICTORS[predictor]
    if predictor == "omegafold":
        pdb, mp = fn(prot, device=device, max_len=max_len, num_cycle=num_cycle)
    else:
        pdb, mp = fn(prot, device=device, max_len=max_len)
    if pdb is None:
        return None
    rd = structural_readout(pdb, plddt_grid=plddt_grid)
    if rd is None:
        return None
    rd["predictor"] = predictor
    rd["struct_mean_plddt"] = mp
    return rd
