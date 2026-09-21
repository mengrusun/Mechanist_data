"""Shared library for the α-helix SAE-steering experiment on Evo2-7B.

All milestones (S0/M1/M2/M-CTRL/M3) import from here so the model, SAE,
hook point, tokenizer conventions, translation, ORF extraction, QC and DSSP
readout are defined exactly once.

resource_fidelity: strict — Evo2-7B (/mnt/quarkfs/share_models/evo2_7b_262k) and
the released Layer-26 Mixed SAE are used at full scale, never substituted.
"""
import os, sys, subprocess, tempfile, json, math, warnings
import numpy as np
import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------
# Fixed resources (strict)
# ----------------------------------------------------------------------------
EVO2_LOCAL = "/mnt/quarkfs/share_models/evo2_7b_262k/evo2_7b_262k.pt"
EVO2_NAME  = "evo2_7b_262k"
SAE_PATH   = "/mnt/quarkfs/share_model/Evo-2-Layer-26-Mixed/sae-layer26-mixed-expansion_8-k_64.pt"
LAYER_NAME = "blocks.26"          # layer-26 residual = output of block 26 (Evo2 public embedding API)
D_MODEL = 4096
D_SAE   = 32768
SAE_K   = 64

# Fixed decoding constants (identical for baseline and every steered arm).
DEC_TEMPERATURE = 1.0
DEC_TOP_P       = 1.0
DEC_TOP_K       = 4               # restrict to the nucleotide alphabet (ACGT); applied identically to all arms
GEN_MAX_NT      = 900
START_CONTEXT   = "ATG"           # neutral coding start; BOS handled by prepend_bos

# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
def load_evo2(verbose=True):
    from evo2 import Evo2
    if verbose:
        print(f"[common] loading Evo2 {EVO2_NAME} from {EVO2_LOCAL} ...", flush=True)
    m = Evo2(model_name=EVO2_NAME, local_path=EVO2_LOCAL)
    return m


def infer_layer_device(evo2_model):
    """Device where blocks.26 output lands (model may be single- or multi-GPU)."""
    try:
        blk = evo2_model.model.get_submodule(LAYER_NAME)
        p = next(blk.parameters())
        return p.device
    except Exception:
        return torch.device("cuda:0")


# ----------------------------------------------------------------------------
# Tied-weight TopK SAE
#   encoder:  pre = (x - b_dec) @ W + b_enc            # (.,32768)
#   latents:  z   = TopK_k( relu(pre) )                # keep k largest, rest 0
#   decoder:  x_hat = z @ W^T + b_dec
#   decoder direction of latent i = W[:, i]  (d_model vector)
# The exact ReLU/TopK order is validated by reconstruction error in the sanity check.
# ----------------------------------------------------------------------------
class TiedTopKSAE:
    def __init__(self, path=SAE_PATH, device="cuda:0", dtype=torch.float32, k=SAE_K,
                 relu_before_topk=True):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        W     = ck["_orig_mod.W"].to(device=device, dtype=dtype)      # (4096,32768)
        b_enc = ck["_orig_mod.b_enc"].to(device=device, dtype=dtype)  # (32768,)
        b_dec = ck["_orig_mod.b_dec"].to(device=device, dtype=dtype)  # (4096,)
        assert W.shape == (D_MODEL, D_SAE), W.shape
        self.W, self.b_enc, self.b_dec = W, b_enc, b_dec
        self.k = k
        self.device = torch.device(device)
        self.dtype = dtype
        self.relu_before_topk = relu_before_topk
        self.dnorm = W.norm(dim=0)                    # (32768,) decoder-column norms
        self.dhat  = W / (self.dnorm + 1e-8)          # (4096,32768) unit decoder directions

    def encode_pre(self, x):
        x = x.to(self.device, self.dtype)
        return (x - self.b_dec) @ self.W + self.b_enc         # (.,32768)

    def encode(self, x):
        """Return dense (.,32768) latent tensor after TopK."""
        pre = self.encode_pre(x)
        act = torch.relu(pre) if self.relu_before_topk else pre
        vals, idx = torch.topk(act, self.k, dim=-1)
        if not self.relu_before_topk:
            vals = torch.relu(vals)
        z = torch.zeros_like(pre)
        z.scatter_(-1, idx, vals)
        return z

    def decode(self, z):
        return z @ self.W.t() + self.b_dec

    def steer_vector(self, feature_idx, s_vals):
        """v = sum_i s_i * dhat_i  over the selected features (a fixed d_model vector).

        feature_idx: 1-D long tensor of latent indices (set S)
        s_vals:      1-D float tensor, s_i (median positive train activation) per index
        """
        idx = torch.as_tensor(feature_idx, device=self.device, dtype=torch.long)
        s   = torch.as_tensor(s_vals, device=self.device, dtype=self.dtype)
        v = (self.dhat[:, idx] * s.unsqueeze(0)).sum(dim=1)   # (4096,)
        return v


# ----------------------------------------------------------------------------
# Steering hook manager — residual-add on the layer-26 block output.
# x' = x + alpha * vec   at every forward (prefill + each cached decode step).
# ----------------------------------------------------------------------------
class ResidualSteerer:
    def __init__(self, evo2_model, vec, alpha=0.0, layer_name=LAYER_NAME):
        self.layer = evo2_model.model.get_submodule(layer_name)
        self.vec = vec                      # (4096,) tensor
        self.alpha = float(alpha)
        self.handle = None

    def _hook(self, mod, inp, out):
        if self.alpha == 0.0:
            return out                       # exact identity at alpha=0
        if isinstance(out, tuple):
            y = out[0]
            add = (self.alpha * self.vec).to(device=y.device, dtype=y.dtype)
            return (y + add,) + tuple(out[1:])
        else:
            add = (self.alpha * self.vec).to(device=out.device, dtype=out.dtype)
            return out + add

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ----------------------------------------------------------------------------
# Activation capture — blocks.26 output for a batch of token ids.
# Returns (B, L, 4096) float32 on CPU.
# ----------------------------------------------------------------------------
@torch.no_grad()
def capture_layer26(evo2_model, input_ids):
    logits, emb = evo2_model.forward(input_ids, return_embeddings=True,
                                     layer_names=[LAYER_NAME])
    act = emb[LAYER_NAME]                     # (B,L,4096)
    return act.float()


# ----------------------------------------------------------------------------
# Tokenizer helpers (CharLevelTokenizer: byte-level, 512 vocab)
# ----------------------------------------------------------------------------
def tokenize_seq(tokenizer, seq, device, prepend_bos=False):
    ids = tokenizer.tokenize(seq)             # list[int]
    if prepend_bos:
        ids = [tokenizer.eod_id if hasattr(tokenizer, "eod_id") else 0] + ids
    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)


# ----------------------------------------------------------------------------
# Genetic code / translation / ORF extraction
# ----------------------------------------------------------------------------
_CODON_TABLE = {  # standard genetic code (table 1); '*' = stop
 'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
 'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
 'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
 'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
 'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
 'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
 'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
 'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G'}
CANON_AA = set("ACDEFGHIKLMNPQRSTVWY")

def translate(dna, stop_at_stop=True):
    dna = dna.upper()
    aa = []
    for i in range(0, len(dna) - 2, 3):
        c = dna[i:i+3]
        r = _CODON_TABLE.get(c, 'X')
        if r == '*':
            if stop_at_stop:
                break
            aa.append('*')
        else:
            aa.append(r)
    return "".join(aa)

def clean_dna(seq):
    """Keep only ACGT characters (Evo2 may emit stray bytes)."""
    return "".join(ch for ch in seq.upper() if ch in "ACGT")

def longest_orf_protein(dna):
    """Longest ORF starting at ATG in any of the 3 forward frames, translated to AA.

    Returns (protein_aa, orf_nt_len, frame, start_nt). Empty if none.
    """
    dna = clean_dna(dna)
    best = ("", 0, -1, -1)
    for frame in range(3):
        i = frame
        n = len(dna)
        while i < n - 2:
            if dna[i:i+3] == "ATG":
                # walk to first stop
                j = i
                prot = []
                while j < n - 2:
                    c = dna[j:j+3]
                    r = _CODON_TABLE.get(c, 'X')
                    if r == '*':
                        break
                    prot.append(r)
                    j += 3
                orf_len = j - i
                if orf_len > best[1]:
                    best = ("".join(prot), orf_len, frame, i)
                i = j + 3
            else:
                i += 3
    return best


# ----------------------------------------------------------------------------
# Sequence validity QC (treatment-independent; identical for baseline & steered).
# Never filters on the DSSP helix outcome or on pLDDT.
# ----------------------------------------------------------------------------
def qc_valid(protein_aa, orf_len, min_aa=50, min_canon_frac=0.95):
    if len(protein_aa) < min_aa:
        return False, "too_short"
    canon = sum(1 for a in protein_aa if a in CANON_AA)
    if canon / max(1, len(protein_aa)) < min_canon_frac:
        return False, "noncanonical"
    # low-complexity guard: no single residue > 50% and >=8 distinct residues
    from collections import Counter
    cnt = Counter(protein_aa)
    if cnt.most_common(1)[0][1] / len(protein_aa) > 0.50:
        return False, "low_complexity"
    if len(cnt) < 8:
        return False, "low_diversity"
    return True, "ok"

def gc_content(dna):
    dna = clean_dna(dna)
    if not dna:
        return 0.0
    return (dna.count("G") + dna.count("C")) / len(dna)


# ----------------------------------------------------------------------------
# DSSP secondary-structure readout on a PDB file -> per-residue 3-state string.
# 8-state DSSP -> 3-state: helix H = {H,G,I}; strand E = {E,B}; coil = rest.
# ----------------------------------------------------------------------------
DSSP_H = set("HGI")
DSSP_E = set("EB")

def run_dssp_on_pdb(pdb_path):
    """Return per-residue 8-state DSSP string using Biopython's DSSP wrapper.

    Requires mkdssp on PATH. Returns "" on failure.
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import DSSP
    try:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("s", pdb_path)
        model = structure[0]
        dssp = DSSP(model, pdb_path, dssp="mkdssp")
        ss = []
        for key in dssp.keys():
            ss.append(dssp[key][2])   # 8-state code, '-' for none
        return "".join(ss)
    except Exception as e:
        sys.stderr.write(f"[dssp] failed on {pdb_path}: {e}\n")
        return ""

def three_state(ss8):
    out = []
    for c in ss8:
        if c in DSSP_H:
            out.append("H")
        elif c in DSSP_E:
            out.append("E")
        else:
            out.append("C")
    return "".join(out)

def frac_helix(ss3):
    if not ss3:
        return float("nan")
    return ss3.count("H") / len(ss3)

def frac_of(ss3, state):
    if not ss3:
        return float("nan")
    return ss3.count(state) / len(ss3)


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
