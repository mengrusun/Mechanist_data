"""Shared ESMFold + DSSP + intervention utilities for M1/M2/M3a/M3b.

Design principles
- Model is loaded ONCE per driver process; all interventions attach/detach hooks.
- Every intervention is implemented as a forward pre-hook on trunk.blocks[k] (for s/z
  patching or additive steering on the block INPUT) or a forward hook on
  block.sequence_to_pair / block.pair_to_sequence (for M2 pathway ablation on OUTPUT).
- DSSP is computed on the ESMFold-predicted PDB (task.md HARD constraint) via mkdssp.
- All per-chain results are stored as JSONL with pdb_id, chain_id, cath_label, effect_size,
  condition, etc., for verify-stage CATH-stratified swap without re-running ESMFold.

Do NOT modify without also updating results/EXPERIMENT_RESULTS.md schema.
"""

from __future__ import annotations
import os
import io
import json
import subprocess
import tempfile
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ESMFOLD_PATH = "/data/zhenqian/models/esmfold_v1"
PDB_CACHE = Path("/data/zhenqian/data/pdb_cache")
PISCES_LIST = Path(
    "/data/zhenqian/data/pisces_cull/"
    "cullpdb_pc25.0_res0.0-2.5_len40-10000_R0.3_Xray_d2026_05_14_chains12055"
)
PROJECT_ROOT = Path("/data/zhenqian/Reproduction1/mechanica/science/esmfold_mechanism")
PREPARED_DIR = PROJECT_ROOT / "data" / "prepared"
MANIFEST_PATH = PREPARED_DIR / "manifest.jsonl"

NUM_TRUNK_BLOCKS = 48
S_HIDDEN = 1024
Z_HIDDEN = 128

# 3-class charge lookup for probe/steering (biophysical convention)
CHARGE_NEG = set("DE")            # Asp / Glu
CHARGE_POS = set("KRH")           # Lys / Arg / His (H included: task-md spec)
# neutral = everything else (natural + unknown letters map to neutral)
CHARGE_CLASSES = ("neg", "pos", "neut")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# DSSP wrapper — pure function
# ---------------------------------------------------------------------------

# mkdssp v4 requires CRYST1 header; ESMFold PDB output omits it, so we prepend a
# dummy P 1 (no crystal packing) header if missing.
_CRYST1_DUMMY = "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"


def run_mkdssp(pdb_text: str) -> Optional[str]:
    """Run mkdssp on a PDB text; return the raw DSSP output text, or None on failure."""
    # mkdssp v4 requires a CRYST1 record. Native PDBs typically have one embedded in
    # the header (after HEADER / TITLE / …); ESMFold's output does NOT. Insert a dummy
    # CRYST1 only when the file has no CRYST1 line at all.
    if "\nCRYST1" not in pdb_text and not pdb_text.startswith("CRYST1"):
        pdb_text = _CRYST1_DUMMY + pdb_text
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as ftmp:
        ftmp.write(pdb_text)
        pdb_path = ftmp.name
    try:
        # mkdssp v4 CLI: `mkdssp INPUT_PDB OUTPUT_DSSP`. Force classic DSSP output
        # format (v4 defaults to mmCIF).
        proc = subprocess.run(
            ["mkdssp", "--output-format", "dssp", pdb_path, "/dev/stdout"],
            capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0:
            return None
        return proc.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    finally:
        try:
            os.unlink(pdb_path)
        except OSError:
            pass


def parse_dssp_ss(dssp_text: str, target_chain: Optional[str] = None) -> List[Tuple[str, int, str, str]]:
    """Parse an mkdssp text output.

    Returns list of (chain_id, resseq, aa, ss_letter). ss_letter uses raw DSSP codes:
    H/G/I helix, E/B strand, T turn, S bend, ' ' (space) → 'C' coil.
    """
    lines = dssp_text.splitlines()
    # Data begins after the line starting with "  #  RESIDUE"
    start = -1
    for i, line in enumerate(lines):
        if line.startswith("  #  RESIDUE"):
            start = i + 1
            break
    if start < 0:
        return []
    out = []
    for line in lines[start:]:
        if len(line) < 17:
            continue
        # chain col ~11, resnum col 6-10, aa col 13, ss col 16
        try:
            chain = line[11]
            resnum_str = line[5:10].strip()
            if not resnum_str:
                continue
            resnum = int(resnum_str)
            aa = line[13]
            ss = line[16]
        except (ValueError, IndexError):
            continue
        if ss == " ":
            ss = "C"
        if target_chain is not None and chain != target_chain:
            continue
        out.append((chain, resnum, aa, ss))
    return out


def ss_string_from_dssp(dssp_records: List[Tuple[str, int, str, str]]) -> str:
    """Concatenate DSSP records (already sorted by resseq inside a chain) into an SS string.

    For a chain returned by parse_dssp_ss the resseq column is monotonic within a chain
    for the majority of PDBs (missing residues → gaps). We just take ss letters in order.
    """
    return "".join(rec[3] for rec in dssp_records)


# ---------------------------------------------------------------------------
# β-hairpin detection
# ---------------------------------------------------------------------------

def find_hairpins(ss_string: str, aa_string: str,
                  min_strand: int = 3, max_turn: int = 5) -> List[Tuple[int, int, int, int]]:
    """Find candidate β-hairpins in a DSSP SS string.

    Returns list of (strand1_start, strand1_end, strand2_start, strand2_end) tuples
    (all inclusive, 0-indexed within the string).

    A β-hairpin here = two E runs of length ≥ min_strand separated by 1..max_turn
    non-E residues (turn / coil / bend). We do NOT check anti-parallel geometry here
    (would require 3D coords); the ≥ 0.7 baseline_hairpin_rate filter ensures ESMFold
    already produces the anti-parallel geometry for the retained chains. This function
    is only used for target-region *bookkeeping* — cross-strand pairs are recomputed
    from the ESMFold-predicted structure per chain.
    """
    N = len(ss_string)
    # Find contiguous E runs
    runs: List[Tuple[int, int]] = []
    i = 0
    while i < N:
        if ss_string[i] == "E":
            j = i
            while j < N and ss_string[j] == "E":
                j += 1
            if j - i >= min_strand:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1
    # For each adjacent pair of runs, check turn length
    hairpins = []
    for a, b in zip(runs, runs[1:]):
        s1_start, s1_end = a
        s2_start, s2_end = b
        turn_len = s2_start - s1_end - 1
        if 1 <= turn_len <= max_turn:
            hairpins.append((s1_start, s1_end, s2_start, s2_end))
    return hairpins


def is_hairpin_region(ss_string: str, tgt_start: int, tgt_end: int,
                      min_strand: int = 3, max_turn: int = 5) -> bool:
    """Boolean: is the region [tgt_start..tgt_end] a β-hairpin in this SS string?

    Interpretation: the region contains at least one hairpin whose two strands both
    lie inside the target region.
    """
    # Only look at the substring but return original indices via offset
    sub_ss = ss_string[tgt_start:tgt_end + 1]
    if len(sub_ss) < 2 * min_strand + 1:
        return False
    for h in find_hairpins(sub_ss, ""):
        # All four indices lie fully in [0, len(sub_ss)-1] by construction
        return True
    return False


# ---------------------------------------------------------------------------
# Cross-strand pair discovery from predicted 3-D structure
# ---------------------------------------------------------------------------

def find_cross_strand_pairs_from_coords(ca_coords: np.ndarray,
                                        strand1_range: Tuple[int, int],
                                        strand2_range: Tuple[int, int],
                                        distance_max: float = 6.5) -> List[Tuple[int, int]]:
    """Return list of (i, j) residue-pair indices (0-indexed in chain) whose Cα-Cα
    distance ≤ distance_max, i in strand1, j in strand2.

    Used to identify which residue pairs are "facing" across the hairpin — this is the
    seq2pair target-pair mask and the M3b steering pair set.
    """
    s1a, s1b = strand1_range
    s2a, s2b = strand2_range
    pairs = []
    for i in range(s1a, s1b + 1):
        for j in range(s2a, s2b + 1):
            d = np.linalg.norm(ca_coords[i] - ca_coords[j])
            if d <= distance_max:
                pairs.append((i, j))
    return pairs


# ---------------------------------------------------------------------------
# ESMFold model loader + intervention hooks
# ---------------------------------------------------------------------------

def load_esmfold(device: str = "cuda", dtype=torch.float32, no_recycles: int = 1):
    """Load ESMFold from local path. Returns (model, tokenizer, no_recycles)."""
    from transformers import AutoTokenizer, EsmForProteinFolding
    tok = AutoTokenizer.from_pretrained(ESMFOLD_PATH)
    model = EsmForProteinFolding.from_pretrained(
        ESMFOLD_PATH, low_cpu_mem_usage=True, torch_dtype=dtype
    )
    model.eval()
    model.trunk.set_chunk_size(64)
    model = model.to(device)
    return model, tok, no_recycles


@torch.no_grad()
def esmfold_forward(model, tok, seq: str, device: str = "cuda", no_recycles: int = 1):
    """Run one ESMFold forward on `seq`. Returns a dict with 'positions' (final 3-D atoms),
    'plddt' (per-residue confidence), and 'pdb' (a PDB text) so DSSP can be run.

    Uses model's built-in output_to_pdb for PDB text.
    """
    inputs = tok([seq], return_tensors="pt", add_special_tokens=False).to(device)
    # ESMFold's forward wants (input_ids, attention_mask), and takes num_recycles kwarg
    out = model(inputs["input_ids"], attention_mask=inputs["attention_mask"],
                num_recycles=no_recycles)
    pdb_list = model.output_to_pdb(out)  # list[str], one entry per batch item
    pdb_text = pdb_list[0]
    plddt = out["plddt"][0].cpu().numpy()  # (L, 37) or (L,) depending on version
    if plddt.ndim > 1:
        # per-atom → per-residue (mean over non-zero atoms of CA if available; use mean)
        plddt = plddt.mean(axis=-1)
    return {"pdb": pdb_text, "plddt": plddt, "raw": out}


def extract_ca_from_pdb(pdb_text: str, chain_id: str = "A") -> np.ndarray:
    """Return (L, 3) numpy array of Cα coordinates from a PDB text for the given chain.
    Missing CA → NaN.
    """
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("x", io.StringIO(pdb_text))
    coords = []
    for model in struct:
        for chain in model:
            if chain.id != chain_id:
                continue
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].coord)
                else:
                    coords.append(np.array([np.nan, np.nan, np.nan]))
        break
    return np.array(coords, dtype=float)


# ---------------------------------------------------------------------------
# Hook contexts — the core intervention primitives
# ---------------------------------------------------------------------------

@contextmanager
def patch_s_at_blocks(model, block_indices: Sequence[int], target_res: Sequence[int],
                      donor_s_by_block: Dict[int, torch.Tensor]):
    """Pre-hook context manager: for each block k in `block_indices`, replace
    `sequence_state[0, target_res, :]` on ENTRY with donor_s_by_block[k][target_res, :].

    donor_s_by_block: dict {block_idx: (L, S_HIDDEN)} — must contain an entry for
    every block in block_indices. Each entry represents the donor's s tensor
    entering that specific block (the true clean/corrupted paradigm requires
    block-specific donor activations).

    Applied to the block's forward pre-hook so the modification propagates through
    the block's internal computations.
    """
    handles = []
    # Pick device/dtype from first donor tensor
    _first = next(iter(donor_s_by_block.values()))
    device = _first.device
    tgt = torch.as_tensor(list(target_res), dtype=torch.long, device=device)

    def make_pre_hook(blk_idx):
        donor_s_local = donor_s_by_block[blk_idx]
        def pre_hook(module, args, kwargs):
            # args[0] = sequence_state (B, L, S_HIDDEN); args[1] = pairwise_state
            assert len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor), \
                f"unexpected block args signature for block {blk_idx}: types {[type(a).__name__ for a in args]}"
            seq_state = args[0]
            # In-place would be dangerous (shared with graph); clone then patch.
            new_seq = seq_state.clone()
            new_seq[0, tgt, :] = donor_s_local[tgt, :].to(new_seq.dtype).to(new_seq.device)
            new_args = (new_seq,) + args[1:]
            return (new_args, kwargs)
        return pre_hook

    try:
        for k in block_indices:
            if k not in donor_s_by_block:
                raise ValueError(f"donor_s_by_block missing block {k}")
            h = model.trunk.blocks[k].register_forward_pre_hook(make_pre_hook(k), with_kwargs=True)
            handles.append(h)
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def patch_z_at_blocks(model, block_indices: Sequence[int],
                      target_pairs: Sequence[Tuple[int, int]],
                      donor_z_by_block: Dict[int, torch.Tensor]):
    """Pre-hook: for each block k in block_indices, replace
    `pairwise_state[0, i, j, :]` on ENTRY for each (i, j) in target_pairs with
    donor_z_by_block[k][i, j, :].

    donor_z_by_block: dict {block_idx: (L, L, Z_HIDDEN)} — one entry per patched block.
    """
    handles = []
    _first = next(iter(donor_z_by_block.values()))
    device = _first.device
    pair_idx_i = torch.as_tensor([p[0] for p in target_pairs], dtype=torch.long, device=device)
    pair_idx_j = torch.as_tensor([p[1] for p in target_pairs], dtype=torch.long, device=device)

    def make_pre_hook(blk_idx):
        donor_z_local = donor_z_by_block[blk_idx]
        def pre_hook(module, args, kwargs):
            assert len(args) >= 2 and isinstance(args[0], torch.Tensor) and isinstance(args[1], torch.Tensor), \
                f"unexpected block args signature for block {blk_idx}"
            seq_state, pair_state = args[0], args[1]
            new_pair = pair_state.clone()
            # Symmetrize both orderings so z stays consistent along both directions.
            new_pair[0, pair_idx_i, pair_idx_j, :] = donor_z_local[pair_idx_i, pair_idx_j, :].to(new_pair.dtype)
            new_pair[0, pair_idx_j, pair_idx_i, :] = donor_z_local[pair_idx_j, pair_idx_i, :].to(new_pair.dtype)
            new_args = (seq_state, new_pair) + args[2:]
            return (new_args, kwargs)
        return pre_hook

    try:
        for k in block_indices:
            if k not in donor_z_by_block:
                raise ValueError(f"donor_z_by_block missing block {k}")
            h = model.trunk.blocks[k].register_forward_pre_hook(make_pre_hook(k), with_kwargs=True)
            handles.append(h)
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def patch_seq2pair_output(model, block_indices: Sequence[int],
                          target_pairs: Sequence[Tuple[int, int]],
                          donor_z_from_seq2pair: torch.Tensor,
                          zero_ablation: bool = False):
    """Post-hook on block.sequence_to_pair: replace the output tensor at target_pairs
    with donor values (or zero-ablate).

    sequence_to_pair returns a pairwise-shaped tensor (1, L, L, Z_HIDDEN).
    """
    handles = []
    pi = torch.as_tensor([p[0] for p in target_pairs], dtype=torch.long, device=donor_z_from_seq2pair.device)
    pj = torch.as_tensor([p[1] for p in target_pairs], dtype=torch.long, device=donor_z_from_seq2pair.device)

    def make_hook(blk_idx):
        def hook(module, inputs, output):
            new_out = output.clone()
            if zero_ablation:
                new_out[0, pi, pj, :] = 0
                new_out[0, pj, pi, :] = 0
            else:
                new_out[0, pi, pj, :] = donor_z_from_seq2pair[pi, pj, :].to(new_out.dtype)
                new_out[0, pj, pi, :] = donor_z_from_seq2pair[pj, pi, :].to(new_out.dtype)
            return new_out
        return hook

    try:
        for k in block_indices:
            h = model.trunk.blocks[k].sequence_to_pair.register_forward_hook(make_hook(k))
            handles.append(h)
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def patch_pair2seq_output(model, block_indices: Sequence[int],
                          target_res: Sequence[int],
                          donor_seq_update: torch.Tensor,
                          zero_ablation: bool = False):
    """Post-hook on block.pair_to_sequence: replace the output tensor at target_res
    with donor values (or zero-ablate).

    pair_to_sequence returns a sequence-shaped tensor (1, L, S_HIDDEN).
    """
    handles = []
    tgt = torch.as_tensor(list(target_res), dtype=torch.long, device=donor_seq_update.device)

    def make_hook(blk_idx):
        def hook(module, inputs, output):
            new_out = output.clone()
            if zero_ablation:
                new_out[0, tgt, :] = 0
            else:
                new_out[0, tgt, :] = donor_seq_update[tgt, :].to(new_out.dtype)
            return new_out
        return hook

    try:
        for k in block_indices:
            h = model.trunk.blocks[k].pair_to_sequence.register_forward_hook(make_hook(k))
            handles.append(h)
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def additive_steering_s(model, block_idx: int, positions: Sequence[int],
                        direction: torch.Tensor, coefficients: Sequence[float]):
    """Pre-hook: at `block_idx`, add coefficients[k] * direction to
    sequence_state[0, positions[k], :] for each k.

    positions: list of residue indices to steer
    coefficients: list of alpha values (same length as positions); different sign per position
      is what implements same/opposite charge configurations.
    direction: (S_HIDDEN,) unit vector.
    """
    handles = []
    tgt = torch.as_tensor(list(positions), dtype=torch.long, device=direction.device)
    coeffs = torch.as_tensor(list(coefficients), dtype=direction.dtype, device=direction.device)

    def pre_hook(module, args, kwargs):
        assert len(args) >= 2 and isinstance(args[0], torch.Tensor), \
            f"steering: unexpected block args, types {[type(a).__name__ for a in args]}"
        seq_state = args[0]
        new_seq = seq_state.clone()
        # scale = coeffs[:, None] * direction[None, :]  → (n_positions, S_HIDDEN)
        scale = coeffs.unsqueeze(-1) * direction.unsqueeze(0)
        new_seq[0, tgt, :] = new_seq[0, tgt, :] + scale.to(new_seq.dtype)
        new_args = (new_seq,) + args[1:]
        return (new_args, kwargs)

    try:
        h = model.trunk.blocks[block_idx].register_forward_pre_hook(pre_hook, with_kwargs=True)
        handles.append(h)
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def collect_s_z_at_block(model, block_idx: int) -> Dict[str, torch.Tensor]:
    """Pre-hook that captures (does not modify) the sequence_state and pairwise_state
    entering block `block_idx`. Access via the returned dict once the with-block runs
    a forward pass.

    Usage:
      with collect_s_z_at_block(model, block_idx=4) as store:
          out = esmfold_forward(model, tok, seq)
      s = store['s']  # (1, L, S_HIDDEN)
      z = store['z']  # (1, L, L, Z_HIDDEN)
    """
    handles = []
    store: Dict[str, torch.Tensor] = {}

    def pre_hook(module, args, kwargs):
        # args[0] = sequence_state (1, L, S), args[1] = pairwise_state (1, L, L, Z)
        store['s'] = args[0].detach().cpu()
        store['z'] = args[1].detach().cpu()
        return None

    try:
        h = model.trunk.blocks[block_idx].register_forward_pre_hook(pre_hook, with_kwargs=True)
        handles.append(h)
        yield store
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Full pipeline: predict → DSSP → hairpin verdict
# ---------------------------------------------------------------------------

def predict_and_judge_hairpin(model, tok, seq: str, target_start: int, target_end: int,
                              device: str = "cuda", no_recycles: int = 1,
                              return_extras: bool = False) -> Dict[str, Any]:
    """One forward pass: predict → run DSSP on predicted PDB → judge hairpin on target.

    ESMFold's PDB output uses sequential resseq 1..L with no insertion codes and no
    gaps, so DSSP records are in 1:1 correspondence with the input sequence. We
    still verify this alignment and fail cleanly if it breaks.

    Returns:
      {'is_hairpin': bool, 'ss': str, 'plddt_target_mean': float,
       'dssp_ok': bool, 'dssp_align_ok': bool,
       'ca_coords': np.ndarray (if return_extras), 'pdb': str (if return_extras)}
    """
    fwd = esmfold_forward(model, tok, seq, device=device, no_recycles=no_recycles)
    pdb_text = fwd["pdb"]
    plddt = fwd["plddt"]
    dssp_text = run_mkdssp(pdb_text)
    if dssp_text is None:
        return {"is_hairpin": None, "ss": None, "plddt_target_mean": float("nan"),
                "dssp_ok": False, "dssp_align_ok": False}
    recs = parse_dssp_ss(dssp_text)
    if not recs:
        return {"is_hairpin": None, "ss": None, "plddt_target_mean": float("nan"),
                "dssp_ok": False, "dssp_align_ok": False}
    chains = {r[0] for r in recs}
    align_ok = True
    if len(chains) > 1:
        align_ok = False
    # Filter to primary chain
    single_chain = next(iter(chains))
    recs_a = [r for r in recs if r[0] == single_chain]
    # ESMFold PDBs have monotonic resseq starting from 1 but DSSP may drop the
    # first residue (H-bond info missing for N-terminal). Reconstruct SS by
    # resseq-aligned assignment: SS[resseq-1] = ss_letter, missing → 'C'.
    ss_arr = ["C"] * len(seq)
    for chain, resseq, aa, ss_letter in recs_a:
        # resseq 1-based; seq positions 0-based
        pos = resseq - 1
        if 0 <= pos < len(seq):
            ss_arr[pos] = ss_letter
    # Compute a coverage stat: fraction of resseq-covered positions among expected
    # positions. If < 0.8, mark align_ok=False as a diagnostic (not a hard fail).
    n_covered = sum(1 for r in recs_a if 1 <= r[1] <= len(seq))
    if n_covered < 0.8 * len(seq):
        align_ok = False
    ss = "".join(ss_arr)
    is_h = is_hairpin_region(ss, target_start, target_end)
    plddt_tgt = float(np.nanmean(plddt[target_start:target_end + 1]))
    result = {"is_hairpin": bool(is_h), "ss": ss,
              "plddt_target_mean": plddt_tgt, "dssp_ok": True,
              "dssp_align_ok": align_ok}
    if return_extras:
        result["pdb"] = pdb_text
        try:
            result["ca_coords"] = extract_ca_from_pdb(pdb_text, chain_id=single_chain)
        except Exception:
            result["ca_coords"] = None
    return result


# ---------------------------------------------------------------------------
# Charge label helper
# ---------------------------------------------------------------------------

def charge_class_ids(seq: str) -> np.ndarray:
    """Map an amino-acid sequence to per-residue 3-class charge labels.
    0 = neg (D/E), 1 = pos (K/R/H), 2 = neut (rest).
    """
    out = np.full(len(seq), 2, dtype=np.int64)
    for i, aa in enumerate(seq):
        if aa in CHARGE_NEG:
            out[i] = 0
        elif aa in CHARGE_POS:
            out[i] = 1
    return out


# ---------------------------------------------------------------------------
# Manifest / persistence utilities
# ---------------------------------------------------------------------------

def load_manifest(path: Path = MANIFEST_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, default=lambda o: float(o) if hasattr(o, "item") else str(o)) + "\n")


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=lambda o: float(o) if hasattr(o, "item") else str(o)) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
