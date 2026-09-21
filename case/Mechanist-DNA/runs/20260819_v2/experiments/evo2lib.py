"""
Shared library for the Evo2-7B alpha-helix steering experiments.

Provides:
- Evo2 loader (singleton) + activation extraction + steering hooks
  (additive contrastive/probe vector, and SAE feature clamp).
- Goodfire Layer-26 BatchTopK SAE (tied) encode/decode.
- DNA utilities: ORF finder, translate, reverse-translate, GC, validity.
- Reproducible seeding.

Model is HARD-pinned to arcinstitute/evo2_7b (32 StripedHyena-2 blocks, d_model=4096).
"""
import os, sys, random, contextlib
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np
import torch

D_MODEL = 4096
N_BLOCKS = 32
ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
SAE_PATH = os.path.abspath(os.path.join(ASSETS, "evo2_l26_sae.pt"))

# ---------------- reproducibility ----------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---------------- genetic code ----------------
# Standard codon -> amino acid
CODON_TABLE = {
 'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
 'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
 'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
 'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
 'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
 'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
 'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
 'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}
# One representative (high-frequency, E. coli-leaning) codon per amino acid for reverse translation.
AA_TO_CODON = {
 'F':'TTC','L':'CTG','I':'ATC','M':'ATG','V':'GTG','S':'AGC','P':'CCG','T':'ACC',
 'A':'GCG','Y':'TAC','H':'CAC','Q':'CAG','N':'AAC','K':'AAA','D':'GAT','E':'GAA',
 'C':'TGC','W':'TGG','R':'CGT','G':'GGC','*':'TAA',
}
STOPS = {'TAA','TAG','TGA'}

# Synonymous codons per amino acid (for realistic, varied reverse translation).
SYN_CODONS = {}
for _cod, _aa in CODON_TABLE.items():
    SYN_CODONS.setdefault(_aa, []).append(_cod)

def synonymous_rev_translate(protein: str, rng) -> str:
    """Reverse-translate with random synonymous codon choice (realistic CDS).
    Always starts with an ATG start codon and ends with a stop (in-frame, frame 0)."""
    body = protein[1:] if protein[:1] == 'M' else protein
    dna = ['ATG']
    for aa in body:
        opts = SYN_CODONS.get(aa)
        dna.append(opts[int(rng.integers(len(opts)))] if opts else 'NNN')
    dna.append(['TAA', 'TGA', 'TAG'][int(rng.integers(3))])
    return "".join(dna)

def reverse_translate(protein: str) -> str:
    """Deterministic reverse translation protein -> CDS DNA (ATG ... TAA), frame 0."""
    body = protein[1:] if protein[:1] == 'M' else protein
    dna = ['ATG']
    for aa in body:
        dna.append(AA_TO_CODON.get(aa, 'NNN'))
    dna.append('TAA')
    return "".join(dna)

def translate(dna: str) -> str:
    prot = []
    for i in range(0, len(dna) - 2, 3):
        aa = CODON_TABLE.get(dna[i:i+3].upper(), 'X')
        if aa == '*':
            break
        prot.append(aa)
    return "".join(prot)

def revcomp(dna: str) -> str:
    c = {'A':'T','T':'A','G':'C','C':'G','N':'N'}
    return "".join(c.get(b, 'N') for b in reversed(dna.upper()))

def gc_content(dna: str) -> float:
    dna = dna.upper()
    if not dna: return 0.0
    return (dna.count('G') + dna.count('C')) / len(dna)

def is_valid_dna(dna: str) -> bool:
    return len(dna) > 0 and all(b in 'ACGT' for b in dna.upper())

def find_longest_orf(dna: str, min_aa: int = 20):
    """Longest ORF (ATG..stop) across 6 frames. Returns (protein, frame, start, dna_orf).
    frame in 0..2 forward, 3..5 reverse. If none, returns best stop-free translation of frame 0."""
    dna = dna.upper()
    best = None  # (len, protein, frame, start, orf_dna)
    strands = [(dna, 0), (revcomp(dna), 3)]
    for seq, foff in strands:
        for f in range(3):
            i = f
            while i < len(seq) - 2:
                if seq[i:i+3] == 'ATG':
                    j = i
                    prot = []
                    while j < len(seq) - 2:
                        cod = seq[j:j+3]
                        aa = CODON_TABLE.get(cod, 'X')
                        if aa == '*':
                            break
                        prot.append(aa); j += 3
                    if len(prot) >= min_aa:
                        cand = (len(prot), "".join(prot), foff + f, i, seq[i:j])
                        if best is None or cand[0] > best[0]:
                            best = cand
                        i = j  # skip past this ORF
                        continue
                i += 3
    if best is None:
        # fallback: translate frame 0 up to first stop
        prot = translate(dna)
        return prot, 0, 0, dna[:len(prot)*3]
    return best[1], best[2], best[3], best[4]

# ---------------- Evo2 wrapper ----------------
_EVO2 = None
def load_evo2():
    global _EVO2
    if _EVO2 is None:
        from evo2 import Evo2
        _EVO2 = Evo2('evo2_7b')
    return _EVO2

def block_name(idx: int) -> str:
    return f"blocks.{idx}"

def tokenize(m, seq: str) -> torch.Tensor:
    return torch.tensor(m.tokenizer.tokenize(seq), dtype=torch.long).unsqueeze(0).cuda()

@torch.no_grad()
def get_block_activations(m, seqs, blocks, pool="mean", batch=1):
    """Return dict block_name -> np.array [N, d_model] of pooled activations over sequence positions."""
    names = [block_name(b) for b in blocks]
    out = {n: [] for n in names}
    for s in seqs:
        ids = tokenize(m, s)
        _, emb = m.forward(ids, return_embeddings=True, layer_names=names)
        for n in names:
            h = emb[n][0].float()  # [L, d]
            if pool == "mean":
                v = h.mean(0)
            elif pool == "last":
                v = h[-1]
            else:
                v = h.mean(0)
            out[n].append(v.cpu().numpy())
    return {n: np.stack(v) for n, v in out.items()}

class SteerHook:
    """Additive residual-stream steering: h <- h + alpha * vec (vec is a torch tensor [d_model])."""
    def __init__(self, model, block_idx, vec, alpha=0.0):
        self.model = model
        self.block_idx = block_idx
        self.vec = vec
        self.state = {"alpha": alpha}
        self.handle = None
    def _hook(self, mod, inp, out):
        a = self.state["alpha"]
        if a == 0.0:
            return out
        v = self.vec
        if isinstance(out, tuple):
            hs = out[0]
            hs = hs + a * v.to(hs.dtype).to(hs.device)
            return (hs,) + tuple(out[1:])
        return out + a * self.vec.to(out.dtype).to(out.device)
    def __enter__(self):
        blk = self.model.get_submodule(block_name(self.block_idx))
        self.handle = blk.register_forward_hook(self._hook)
        return self
    def set_alpha(self, a): self.state["alpha"] = a
    def __exit__(self, *a):
        if self.handle: self.handle.remove()

# ---------------- Goodfire L26 BatchTopK SAE (tied) ----------------
class SAEClamp:
    """Load the tied BatchTopK SAE; encode->clamp target feature->decode as a residual hook on blocks.26.

    Weights: tied autoencoder, encoder W [n_features, d_model]; decoder = W^T.
    BatchTopK at inference uses per-token top-k (k=64).
    """
    def __init__(self, sae_path=SAE_PATH, k=64, device="cuda", dtype=torch.float32):
        sd = torch.load(sae_path, map_location="cpu", weights_only=False)
        # strip compile prefix
        sd = { (kk.replace("_orig_mod.", "")): vv for kk, vv in sd.items() }
        self.keys = list(sd.keys())
        # Identify weight and biases heuristically
        W = None; b_enc = None; b_dec = None; thresh = None
        for kk, vv in sd.items():
            if vv.ndim == 2 and W is None:
                W = vv.float()
            elif vv.ndim == 1:
                if b_enc is None and vv.shape[0] == (W.shape[0] if W is not None else -1):
                    b_enc = vv.float()
        # second pass for shapes now that W known
        n_feat, d_model = W.shape
        for kk, vv in sd.items():
            if vv.ndim == 1:
                if vv.shape[0] == n_feat and ('enc' in kk or 'b_enc' in kk or kk in ('b','bias')):
                    b_enc = vv.float()
                elif vv.shape[0] == d_model and ('dec' in kk or 'b_dec' in kk or 'pre' in kk):
                    b_dec = vv.float()
                elif vv.shape[0] == n_feat and ('thresh' in kk or 'jump' in kk):
                    thresh = vv.float()
        self.W = W.to(device)                       # [n_feat, d_model]
        self.b_enc = (b_enc if b_enc is not None else torch.zeros(n_feat)).to(device)
        self.b_dec = (b_dec if b_dec is not None else torch.zeros(d_model)).to(device)
        self.n_feat, self.d_model = n_feat, d_model
        self.k = k
        self.device = device
        self.raw_keys = {kk: tuple(vv.shape) for kk, vv in sd.items()}

    def encode(self, h):  # h [..., d_model] float
        x = h - self.b_dec
        pre = x @ self.W.t() + self.b_enc          # [..., n_feat]
        return pre

    def topk_acts(self, pre):
        # BatchTopK approximated by per-token top-k on positive pre-activations
        relu = torch.relu(pre)
        k = min(self.k, relu.shape[-1])
        val, idx = torch.topk(relu, k, dim=-1)
        f = torch.zeros_like(relu)
        f.scatter_(-1, idx, val)
        return f

    def decode(self, f):
        return f @ self.W + self.b_dec

    def feature_activations(self, h):
        """Full (pre-topk relu) activations for ranking. h [...,d]->[...,n_feat]."""
        return torch.relu(self.encode(h))

class SAEClampHook:
    """Hook on blocks.26: encode residual, set target feature to clamp_value, decode, replace."""
    def __init__(self, model, sae: SAEClamp, feat_idx, clamp_value=0.0, block_idx=26):
        self.model = model; self.sae = sae; self.feat_idx = feat_idx
        self.state = {"clamp": clamp_value}; self.block_idx = block_idx; self.handle=None
    def _hook(self, mod, inp, out):
        cv = self.state["clamp"]
        if cv == 0.0:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        orig_dtype = hs.dtype
        h = hs.float()
        pre = self.sae.encode(h)
        f = self.sae.topk_acts(pre)
        # force target feature(s) to clamp value
        idxs = self.feat_idx if isinstance(self.feat_idx, (list, tuple)) else [self.feat_idx]
        for ix in idxs:
            f[..., ix] = cv
        recon = self.sae.decode(f)
        # residual replacement: replace the reconstructed component (keep SAE error term)
        err = h - self.sae.decode(self.sae.topk_acts(self.sae.encode(h)))
        new = recon + err
        new = new.to(orig_dtype)
        if isinstance(out, tuple):
            return (new,) + tuple(out[1:])
        return new
    def __enter__(self):
        blk = self.model.get_submodule(block_name(self.block_idx))
        self.handle = blk.register_forward_hook(self._hook); return self
    def set_clamp(self, v): self.state["clamp"] = v
    def __exit__(self, *a):
        if self.handle: self.handle.remove()

@torch.no_grad()
def generate(m, prompts, n_tokens=300, temperature=0.7, top_k=4, top_p=1.0):
    out = m.generate(prompt_seqs=list(prompts), n_tokens=n_tokens,
                     temperature=temperature, top_k=top_k, top_p=top_p,
                     batched=True, cached_generation=True, verbose=0)
    return out.sequences, [float(x) for x in out.logprobs_mean]

@torch.no_grad()
def coding_loglik(m, seqs):
    """Mean per-token log-likelihood (Evo2 native) as coding-likelihood covariate."""
    try:
        scores = m.score_sequences(list(seqs), batch_size=1, reduce_method='mean')
        return [float(s) for s in scores]
    except Exception:
        return [float('nan')] * len(seqs)
