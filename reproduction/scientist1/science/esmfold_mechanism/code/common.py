"""Common utilities: model loading, DSSP-based hairpin detection, and I/O."""
import os
import io
import sys
import json
import subprocess
import tempfile
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

MODEL_DIR = "/data/zhenqian/models/esmfold_v1"
PISCES_LIST = "/data/zhenqian/data/pisces_cull/cullpdb_pc25.0_res0.0-2.5_len40-10000_R0.3_Xray_d2026_05_14_chains12055"
PDB_CACHE = "/data/zhenqian/data/pdb_cache"

BASE = Path("/data/zhenqian/Reproduction1/cc/science/esmfold_mechanism")
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "outputs"

AA3TO1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

# Charge encoding: Arg/Lys = +1, Asp/Glu = -1, others = 0
CHARGE = {"R": +1, "K": +1, "H": 0,  # keep His neutral at physiological pH default
          "D": -1, "E": -1}
def aa_charge(aa: str) -> int:
    return CHARGE.get(aa, 0)


def load_esmfold(device="cuda:0", dtype=torch.float32):
    """Load ESMFold from local checkpoint."""
    from transformers import AutoTokenizer
    from transformers.models.esm.modeling_esmfold import EsmForProteinFolding
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = EsmForProteinFolding.from_pretrained(MODEL_DIR, torch_dtype=dtype)
    model = model.eval()
    model = model.to(device)
    # esm2 backbone can stay in fp32 or fp16 based on cfg
    try:
        # esm attribute may be nn.Module or class
        pass
    except Exception:
        pass
    # Optimize memory: use chunking for long sequences
    try:
        model.trunk.set_chunk_size(64)
    except Exception:
        pass
    return tokenizer, model


def tokenize(tokenizer, seqs):
    inp = tokenizer(seqs, return_tensors="pt", add_special_tokens=False, padding=True)
    return inp


@torch.no_grad()
def esmfold_infer(model, tokenizer, seq, num_recycles=1, device="cuda:0"):
    inp = tokenize(tokenizer, [seq]).to(device)
    out = model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"], num_recycles=num_recycles)
    return out


def output_to_pdb_str(model, output):
    # output_to_pdb expects a dict of tensors
    d = {}
    for k, v in output.items():
        if isinstance(v, torch.Tensor):
            d[k] = v.detach()
        else:
            d[k] = v
    pdbs = model.output_to_pdb(d)
    return pdbs[0]


CRYST1_LINE = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"

def write_pdb(pdb_str, path):
    """Write PDB, prepending CRYST1 line if missing (needed for DSSP)."""
    if not pdb_str.startswith("CRYST1"):
        # Strip PARENT header if present, prepend CRYST1
        lines = pdb_str.splitlines(keepends=True)
        clean = [ln for ln in lines if not ln.startswith(("PARENT",))]
        pdb_str = CRYST1_LINE + "".join(clean)
    Path(path).write_text(pdb_str)


def run_dssp(pdb_path: str):
    """Return list of (resid, aa, ss) tuples using mkdssp. Uses BioPython DSSP wrapper."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("x", pdb_path)
    model = structure[0]
    try:
        dssp = DSSP(model, pdb_path, dssp="mkdssp")
    except Exception:
        dssp = DSSP(model, pdb_path, dssp="dssp")
    out = []
    # DSSP keys: (chain, res_id_tuple) -> (index, aa, ss, ...)
    # We iterate over model residues in order to keep position order
    idx = 0
    for chain in model:
        for res in chain:
            if res.id[0] != " ":
                continue
            key = (chain.id, res.id)
            if key in dssp.keys():
                d = dssp[key]
                aa = d[1]
                ss = d[2]
            else:
                aa = AA3TO1.get(res.get_resname(), "X")
                ss = "-"
            out.append((idx, aa, ss))
            idx += 1
    return out


def dssp_ss_string(dssp_out):
    return "".join(r[2] for r in dssp_out)


def find_hairpins_in_ss(ss: str, min_strand=3, max_strand=10, min_loop=2, max_loop=6):
    """Find candidate β-hairpin motifs: two adjacent E-strands separated by 2..6 non-strand residues.

    Returns list of (s1_start, s1_end, loop_start, loop_end, s2_start, s2_end) using inclusive indices.
    """
    n = len(ss)
    # find contiguous stretches of 'E'
    strands = []
    i = 0
    while i < n:
        if ss[i] == 'E':
            j = i
            while j < n and ss[j] == 'E':
                j += 1
            strands.append((i, j - 1))
            i = j
        else:
            i += 1
    hairpins = []
    for a_idx in range(len(strands) - 1):
        s1 = strands[a_idx]
        s2 = strands[a_idx + 1]
        s1_len = s1[1] - s1[0] + 1
        s2_len = s2[1] - s2[0] + 1
        if not (min_strand <= s1_len <= max_strand):
            continue
        if not (min_strand <= s2_len <= max_strand):
            continue
        loop_len = s2[0] - s1[1] - 1
        if not (min_loop <= loop_len <= max_loop):
            continue
        # Between strands: no other 'E' present is guaranteed by construction.
        hairpins.append((s1[0], s1[1], s1[1] + 1, s2[0] - 1, s2[0], s2[1]))
    return hairpins


def parse_pisces(path: str):
    """Return list of dicts: {pdb, chain, len}."""
    rows = []
    with open(path) as f:
        header = None
        for line in f:
            parts = line.split()
            if header is None:
                header = parts
                continue
            if len(parts) < 2:
                continue
            pdb_chain = parts[0]
            pdb = pdb_chain[:4].upper()
            chain = pdb_chain[4:]
            try:
                length = int(parts[1])
            except ValueError:
                continue
            rows.append({"pdb": pdb, "chain": chain, "length": length})
    return rows


def read_pdb_sequence_and_positions(pdb_path: str, chain_id: str):
    """Return (sequence_str, residue_ids_list) for CA-containing residues in given chain."""
    parser = PDBParser(QUIET=True)
    st = parser.get_structure("x", pdb_path)
    model = st[0]
    if chain_id not in [c.id for c in model]:
        return None, None
    chain = model[chain_id]
    seq = []
    resids = []
    for res in chain:
        # skip HETATM & non-standard
        if res.id[0] != " ":
            continue
        rn = res.get_resname()
        if rn not in AA3TO1:
            continue
        if "CA" not in res:
            continue
        seq.append(AA3TO1[rn])
        resids.append(res.id[1])
    return "".join(seq), resids


def pisces_pdb_path(pdb_id: str) -> str:
    return os.path.join(PDB_CACHE, f"{pdb_id.upper()}.pdb")


def register_block_capture_hooks(model, save_dict, positions=None):
    """Attach forward hooks that capture s_out and z_out after each block."""
    hooks = []
    for i, blk in enumerate(model.trunk.blocks):
        def make(i):
            def hook(mod, inp, out):
                s, z = out
                # Save mean over positions of s at target region (or full), and mean z over hairpin pair positions
                entry = {}
                if positions is None:
                    entry["s_mean"] = s.detach().mean(dim=1).float().cpu()
                else:
                    p = positions
                    entry["s_target"] = s[:, p, :].detach().float().cpu()
                save_dict.setdefault(i, {}).update(entry)
            return hook
        hooks.append(blk.register_forward_hook(make(i)))
    return hooks


def clear_hooks(hooks):
    for h in hooks:
        h.remove()
