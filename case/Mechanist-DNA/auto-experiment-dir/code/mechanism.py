"""
Shared mechanism utilities for M1-M3: SAE feature steering during Evo2 autoregressive DNA
generation + DNA->protein->ESMFold->DSSP readout.

Steering: at the Layer-26 residual stream (`blocks.26` output, the SAE's native site), amplify
feature set S. In the SAE's normalized space the reconstruction delta of adding alpha*s_f to each
feature f in S is  d_dir = sum_{f in S} (alpha * s_f) * W[:, f]  (tied decoder = columns of W).
Mapped back to the raw residual (undo unit_sqrtd normalization) the per-token delta is
  delta = (||x|| / sqrt(d)) * d_dir.
This adds ONLY the steering contribution and preserves the rest of the residual stream.
"""
import os, sys, math, json, contextlib
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from evo2_sae import BatchTopKSAE, HIDDEN, SAE_DICT

STEER_SITE = "blocks.26.post_norm"  # SAE native site (M(-1): reproduces paper features f/28741, f/22326)
SAE_NORM = "none"                    # raw post_norm activations (matches discovery config)
SQRT_D = math.sqrt(HIDDEN)


class Steerer:
    """Registers a forward hook on blocks.26 that amplifies feature set S by coefficient alpha."""

    def __init__(self, model, sae: BatchTopKSAE, feats, s_f, device="cuda:0"):
        self.model = model
        self.sae = sae
        self.feats = torch.as_tensor(list(feats), dtype=torch.long, device=device)
        self.s_f = torch.as_tensor(list(s_f), dtype=torch.float32, device=device)  # per-feature scale
        self.device = device
        self.alpha = 0.0
        self._handle = None
        # decoder directions for S: (4096, |S|) then weighted by s_f -> base direction
        Wc = self.sae.W[:, self.feats].float()  # (4096, |S|)
        self.base_dir = (Wc * self.s_f.unsqueeze(0)).sum(1)  # (4096,) = sum_f s_f W[:,f]

    def _hook(self, module, inputs, output):
        if self.alpha == 0.0:
            return output
        x = output[0] if isinstance(output, tuple) else output  # (B,L,4096) post_norm output
        xf = x.float()
        if self.sae.normalize == "unit_sqrtd":
            # SAE reads unit_sqrtd-normalized activations -> map decoder delta back to raw scale
            r = xf.norm(dim=-1, keepdim=True)
            delta = (r / SQRT_D) * (self.alpha * self.base_dir)
        else:
            # raw-input SAE (post_norm|none): clamp-style add of the amplified decoder direction
            delta = self.alpha * self.base_dir
        xnew = (xf + delta).to(x.dtype)
        if isinstance(output, tuple):
            return (xnew,) + tuple(output[1:])
        return xnew

    @contextlib.contextmanager
    def steer(self, alpha):
        self.alpha = float(alpha)
        blk = self.model.model.get_submodule(STEER_SITE)
        self._handle = blk.register_forward_hook(self._hook)
        try:
            yield
        finally:
            if self._handle:
                self._handle.remove(); self._handle = None
            self.alpha = 0.0


def compute_feature_scales(org, feats):
    """s_f = mean active (nonzero) activation of feature f across the cached M0 codons (norm space)."""
    from scipy import sparse
    import m0_data as D
    X = sparse.load_npz(os.path.join(D.DATA_DIR, f"m0_acts_{org}.npz")).tocsc()
    scales = []
    for f in feats:
        col = X[:, int(f)]
        scales.append(float(col.data.mean()) if col.nnz > 0 else 1.0)
    return scales


# ---------- translation / ORF filter ----------
from Bio.Seq import Seq
STANDARD_TABLE = 1


def translate_orf(dna, min_aa=30, table=STANDARD_TABLE, require_start=False):
    """Translate DNA in frame 0 from first ATG (or position 0); stop at first stop codon.
    Returns (protein, valid, meta)."""
    dna = dna.upper().replace("U", "T")
    dna = "".join(c for c in dna if c in "ACGT")
    start = 0
    if require_start:
        idx = dna.find("ATG")
        if idx < 0:
            return "", False, {"reason": "no_start"}
        start = idx
    orf = dna[start:]
    orf = orf[: len(orf) - len(orf) % 3]
    if len(orf) < 3:
        return "", False, {"reason": "too_short"}
    prot = str(Seq(orf).translate(table=table))
    # cut at first stop
    stop_idx = prot.find("*")
    if stop_idx >= 0:
        prot_cut = prot[:stop_idx]
    else:
        prot_cut = prot.replace("*", "")
    premature = stop_idx >= 0 and stop_idx < len(prot) - 1
    valid = len(prot_cut) >= min_aa
    return prot_cut, valid, {"len_aa": len(prot_cut), "had_internal_stop": bool(premature),
                             "orf_nt": len(orf)}


# ---------- ESMFold structure prediction ----------
_esmfold = {"model": None, "tok": None}


def load_esmfold(device="cuda:0"):
    if _esmfold["model"] is None:
        from transformers import EsmForProteinFolding, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        model = model.eval().to(device)
        model.esm = model.esm.half()  # fp16 language-model trunk to save memory
        model.trunk.set_chunk_size(64)
        _esmfold["model"] = model; _esmfold["tok"] = tok
    return _esmfold["model"], _esmfold["tok"]


@torch.no_grad()
def esmfold_pdb(prot, device="cuda:0", max_len=400):
    """Predict structure; return (pdb_string, mean_plddt) or (None, None)."""
    model, tok = load_esmfold(device)
    prot = prot[:max_len]
    if len(prot) < 20:
        return None, None
    ids = tok([prot], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    out = model(ids)
    plddt = float(out["plddt"][0, :, 1].mean().item()) if "plddt" in out else float(out.plddt.mean())
    # HF ESMFold returns pLDDT on a 0-1 scale here; normalize to the conventional 0-100 scale.
    if plddt <= 1.5:
        plddt *= 100.0
    pdb = model.output_to_pdb(out)[0]
    return pdb, plddt


# ---------- DSSP readout on predicted structure ----------
import subprocess, tempfile


def dssp_fractions(pdb_string):
    """Run mkdssp on a PDB string; return dict with helix_hgi, helix_h, sheet fractions and n_resolved."""
    import m0_data as D
    with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False, dir=D.DSSP_DIR) as f:
        f.write(pdb_string); path = f.name
    out = path.replace(".pdb", ".dssp")
    try:
        r = subprocess.run(["mkdssp", path, out], capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            r = subprocess.run(["mkdssp", "--output-format", "dssp", path, out],
                               capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(out):
            return None
        from Bio.PDB.DSSP import make_dssp_dict
        d, keys = make_dssp_dict(out)
        ss = [d[k][1] for k in keys]
        n = len(ss)
        if n == 0:
            return None
        return {"helix_hgi": sum(c in "HGI" for c in ss) / n,
                "helix_h": sum(c == "H" for c in ss) / n,
                "sheet": sum(c in "E" for c in ss) / n, "n_resolved": n}
    finally:
        for p in (path, out):
            try: os.remove(p)
            except Exception: pass


@torch.no_grad()
def generate_dna(evo2, prompt, n_tokens, temperature=1.0, top_k=4, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    out = evo2.generate(prompt_seqs=[prompt], n_tokens=n_tokens, temperature=temperature,
                        top_k=top_k, verbose=0)
    # vortex GenerationOutput dataclass: .sequences is List[str] of generated continuations
    if hasattr(out, "sequences"):
        return out.sequences[0]
    if isinstance(out, (tuple, list)):
        return out[0][0] if isinstance(out[0], (list, tuple)) else out[0]
    return out


@torch.no_grad()
def generate_and_readout(evo2, steerer, prompts, alpha, n_tokens, temperature, top_k,
                         seed_base, plddt_min=60.0, device="cuda:0", table=STANDARD_TABLE,
                         min_aa=30, fold=True):
    """Generate one steered sequence per prompt at coefficient alpha, translate+ORF-filter,
    ESMFold (pLDDT-gated), DSSP. Returns per-sample records + aggregate."""
    # PHASE 1: generate ALL DNA + translate (Evo2 only; ESMFold NOT loaded yet).
    # Alternating Evo2 generation with ESMFold in one process corrupts vortex/flash-attn kernel
    # dispatch (schema_.has_value assert) -> keep the two model runtimes strictly separated.
    records = []
    ctx = steerer.steer(alpha) if (steerer is not None and alpha != 0.0) else contextlib.nullcontext()
    with ctx:
        for i, prompt in enumerate(prompts):
            rec = {"valid_orf": False, "gated": True, "prot": None}
            try:
                full = generate_dna(evo2, prompt, n_tokens, temperature, top_k, seed=seed_base + i)
                gen = full[len(prompt):] if full.startswith(prompt) else full
                prot, valid, meta = translate_orf(prompt + gen, min_aa=min_aa, table=table)
                rec = {"valid_orf": valid, "len_aa": meta.get("len_aa", 0),
                       "had_internal_stop": meta.get("had_internal_stop", False),
                       "gated": True, "prot": prot if valid else None}
            except Exception as e:
                rec["error"] = f"gen: {type(e).__name__}: {str(e)[:100]}"
            records.append(rec)

    # PHASE 2: fold all valid proteins with ESMFold (Evo2 generation done for this batch).
    if fold:
        for rec in records:
            if not rec.get("prot"):
                continue
            try:
                pdb, plddt = esmfold_pdb(rec["prot"], device=device)
                rec["plddt"] = plddt
                if pdb is not None and plddt is not None and plddt >= plddt_min:
                    fr = dssp_fractions(pdb)
                    if fr:
                        rec.update({"helix_hgi": fr["helix_hgi"], "helix_h": fr["helix_h"],
                                    "sheet": fr["sheet"], "n_resolved": fr["n_resolved"], "gated": False})
            except Exception as e:
                rec["error"] = f"fold: {type(e).__name__}: {str(e)[:100]}"
                torch.cuda.empty_cache()
    for rec in records:
        rec.pop("prot", None)
    return records


def aggregate(records, key="helix_hgi"):
    vals = [r[key] for r in records if (not r.get("gated", True)) and key in r]
    import numpy as np
    n_valid = sum(r.get("valid_orf", False) for r in records)
    n_gated_pass = len(vals)
    out = {"n": len(records), "n_valid_orf": n_valid, "n_folded_gated": n_gated_pass,
           "valid_orf_rate": n_valid / max(len(records), 1),
           "gated_pass_rate": n_gated_pass / max(len(records), 1)}
    if vals:
        a = np.array(vals)
        out.update({f"{key}_mean": float(a.mean()), f"{key}_std": float(a.std()),
                    f"{key}_sem": float(a.std() / np.sqrt(len(a))), f"{key}_median": float(np.median(a))})
        plddts = [r["plddt"] for r in records if r.get("plddt") is not None]
        if plddts:
            out["plddt_mean"] = float(np.mean(plddts))
    return out
