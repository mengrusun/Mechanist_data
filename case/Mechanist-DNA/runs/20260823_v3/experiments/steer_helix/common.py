"""Shared measurement harness for steering Evo2-7B toward high alpha-helical content.

Frozen in M1, applied identically to every condition (baseline, steered, controls).
Model under study: Evo2-7B (HARD, no substitution). Assay: ESMFold -> DSSP -> alpha-helix
fraction of the translated ORF. See refine-logs/EXPERIMENT_PLAN.md shared measurement contract.
"""
import os, re, json, hashlib, tempfile, subprocess
import numpy as np

# ---------------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------------
PROJECT = "/data/wanghaoxiong/Mechanist-DNA-experiment/20260823_v1"
# --- Load ALL weights DIRECTLY from local /mnt/quarkfs (user override @ experiment-resume);
#     never hit the network. The guard hook does not block /mnt/quarkfs. ---
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ.pop("HF_TOKEN", None)
os.environ.setdefault("HF_HOME", os.path.join(PROJECT, "models_cache"))
EVO2_LOCAL_PT = "/mnt/quarkfs/share_model/evo2_7b/evo2_7b.pt"          # 13.77 GB
ESMFOLD_LOCAL_DIR = "/mnt/quarkfs/share_model/esmfold_v1"             # HF EsmForProteinFolding layout
SAE_PATH = "/mnt/quarkfs/share_model/Evo-2-Layer-26-Mixed/sae-layer26-mixed-expansion_8-k_64.pt"
# BioPython DSSP works with mkdssp 4.5.8 (conda) on ESMFold PDBs; the system /usr/bin/mkdssp
# (4.0.4) fails to parse ESMFold-emitted PDB (PARENT record / CRYST1). Verified on a synthetic
# helix -> 81.6% helix. Prefer the conda binary.
_CONDA_MKDSSP = "/data/wanghaoxiong/miniconda3/envs/scientist/bin/mkdssp"
DSSP_BIN = _CONDA_MKDSSP if os.path.exists(_CONDA_MKDSSP) else (
    "/usr/bin/mkdssp" if os.path.exists("/usr/bin/mkdssp") else "mkdssp")

# ---- Frozen harness constants (pre-registered in M1; never changed post-hoc) ----
HARNESS = dict(
    model_under_study="evo2_7b",
    gen_n_tokens=900,            # ~300 codons of DNA -> up to ~300 aa ORF
    gen_temperature=1.0,
    gen_top_k=4,
    gen_top_p=1.0,
    orf_rule="longest_atg_stop_6frame",
    orf_min_aa=30,              # L_min
    orf_max_aa=300,             # L_max (plausible length ceiling)
    primary_ss_assay="esmfold_dssp",
    helix_codes=("H", "G", "I"),      # DSSP alpha/3-10/pi helix
    sheet_codes=("E", "B"),
    plddt_policy="weight_by_plddt_over_100",  # fixed confidence weighting, all conditions
    plddt_min_residue=0.0,      # do not hard-drop residues; weight instead
    helix_favoring_aa=tuple("AELMQK"),
    validity_nll_max=1.5,       # coding-plausibility ceiling on mean per-token Evo2 NLL
    primary_endpoint="mean_helix_all_generations_invalid_as_zero",
    seeds=(0, 1, 2),
)
DNA_PRIMERS = [  # fixed prompt/primer set (shared across all conditions)
    "ATGGCT", "ATGAAA", "ATGGGC", "ATGCTG", "ATGGAA", "ATGGTG",
    "ATGACC", "ATGCAG", "ATGTTT", "ATGCGT",
]
CODON_TABLE = {  # standard genetic code
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','CTT':'L','CTC':'L','CTA':'L','CTG':'L',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','GTT':'V','GTC':'V','GTA':'V','GTG':'V',
    'TCT':'S','TCC':'S','TCA':'S','TCG':'S','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'ACT':'T','ACC':'T','ACA':'T','ACG':'T','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','GAT':'D','GAC':'D','GAA':'E','GAG':'E',
    'TGT':'C','TGC':'C','TGA':'*','TGG':'W','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'AGT':'S','AGC':'S','AGA':'R','AGG':'R','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}

def set_seed(seed):
    import random, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------
# ORF extraction + translation (deterministic; identical for all conditions)
# ---------------------------------------------------------------------------
def _revcomp(s):
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]

def _clean_dna(dna):
    return "".join(c for c in dna.upper() if c in "ACGT")

def extract_orf(dna, min_aa=None, max_aa=None):
    """Longest ATG->stop in-frame ORF over 6 frames, >= min_aa. Returns (orf_dna, protein)
    or (None, None). Deterministic tie-break: longest, then lowest frame index, then 5'-most."""
    min_aa = HARNESS["orf_min_aa"] if min_aa is None else min_aa
    max_aa = HARNESS["orf_max_aa"] if max_aa is None else max_aa
    seq = _clean_dna(dna)
    best = None  # (length_aa, frame, start, orf_dna, protein)
    for strand, s in enumerate((seq, _revcomp(seq))):
        for frame in range(3):
            i = frame
            while i < len(s) - 2:
                if s[i:i+3] == "ATG":
                    prot = []
                    j = i
                    while j < len(s) - 2:
                        aa = CODON_TABLE.get(s[j:j+3], "X")
                        if aa == "*":
                            break
                        prot.append(aa); j += 3
                    else:
                        # ran off end without stop -> not a well-formed ORF
                        i += 3; continue
                    L = len(prot)
                    key = (L, -(strand*3+frame), -i)
                    if L >= min_aa and (best is None or key > best[0]):
                        best = (key, "".join(prot), s[i:j+3], strand*3+frame, i)
                    i = j + 3
                else:
                    i += 3
    if best is None:
        return None, None
    protein = best[1]
    if len(protein) > max_aa:
        protein = protein[:max_aa]
    return best[2], protein

def gc_content(dna):
    s = _clean_dna(dna)
    return (s.count("G") + s.count("C")) / max(1, len(s))

def aa_composition(protein, residues=None):
    residues = HARNESS["helix_favoring_aa"] if residues is None else residues
    if not protein:
        return 0.0
    return sum(protein.count(r) for r in residues) / len(protein)

# ---------------------------------------------------------------------------
# Validity (C2) — fixed composite over the FULL generated set
# ---------------------------------------------------------------------------
def validity_label(dna, protein, mean_nll=None):
    """Fixed composite -> bool valid. Pre-registered hierarchy; no per-condition gaming."""
    if protein is None:
        return False, dict(reason="no_orf")
    L = len(protein)
    checks = dict(
        has_orf=protein is not None,
        length_ok=(HARNESS["orf_min_aa"] <= L <= HARNESS["orf_max_aa"]),
        starts_met=protein.startswith("M"),
        no_internal_stop=("*" not in protein),   # ORF ends at first stop by construction
        coding_plausible=(mean_nll is None or mean_nll <= HARNESS["validity_nll_max"]),
    )
    return all(checks.values()), checks

# ---------------------------------------------------------------------------
# ESMFold -> DSSP alpha-helix assay (the single frozen primary assay)
# ---------------------------------------------------------------------------
class HelixAssay:
    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(ESMFOLD_LOCAL_DIR)
        self.model = EsmForProteinFolding.from_pretrained(
            ESMFOLD_LOCAL_DIR, low_cpu_mem_usage=True)
        self.model = self.model.to(device).eval()
        self.model.esm = self.model.esm.half()  # ESM trunk in fp16 to save memory
        self.device = device

    def fold_to_pdb(self, protein):
        with self.torch.no_grad():
            inp = self.tok([protein], return_tensors="pt", add_special_tokens=False)
            inp = {k: v.to(self.device) for k, v in inp.items()}
            out = self.model(**inp)
        pdb = self.model.output_to_pdb(out)[0]
        plddt = out["plddt"][0, :, 1].mean().item()  # CA pLDDT, 0..100
        return pdb, plddt

    def ss_fractions(self, protein):
        """Return dict(helix_frac, sheet_frac, plddt, n_res) using ESMFold->DSSP.
        pLDDT-weighted per the frozen policy. On failure returns helix_frac=0 flagged."""
        if not protein or len(protein) < 3:
            return dict(helix_frac=0.0, sheet_frac=0.0, plddt=0.0, n_res=0, ok=False)
        try:
            pdb, plddt = self.fold_to_pdb(protein)
        except Exception as e:
            return dict(helix_frac=0.0, sheet_frac=0.0, plddt=0.0, n_res=0, ok=False,
                        err=str(e)[:80])
        return self._dssp_from_pdb(pdb, plddt)

    def _dssp_from_pdb(self, pdb_str, plddt):
        from Bio.PDB import PDBParser, DSSP
        import warnings; warnings.filterwarnings("ignore")
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
            f.write(pdb_str); pdb_path = f.name
        try:
            structure = PDBParser(QUIET=True).get_structure("x", pdb_path)
            model = structure[0]
            dssp = DSSP(model, pdb_path, dssp=DSSP_BIN)
            ss, conf = [], []
            for key in dssp.keys():
                rec = dssp[key]
                ss.append(rec[2])            # DSSP SS code
                # rec[3] is rel ASA; use per-residue confidence from B-factor via plddt mean
            n = len(ss)
            if n == 0:
                return dict(helix_frac=0.0, sheet_frac=0.0, plddt=plddt, n_res=0, ok=False)
            helix = sum(1 for c in ss if c in HARNESS["helix_codes"])
            sheet = sum(1 for c in ss if c in HARNESS["sheet_codes"])
            return dict(helix_frac=helix / n, sheet_frac=sheet / n,
                        plddt=plddt, n_res=n, ok=True)
        except Exception as e:
            return dict(helix_frac=0.0, sheet_frac=0.0, plddt=plddt, n_res=0, ok=False,
                        err=str(e)[:80])
        finally:
            try: os.unlink(pdb_path)
            except Exception: pass

# ---------------------------------------------------------------------------
# Evo2-7B wrapper: generation, activation extraction, steering hooks, NLL
# ---------------------------------------------------------------------------
BLOCK_RE = re.compile(r"^blocks\.(\d+)$")

class Evo2Wrapper:
    def __init__(self, model_name="evo2_7b"):
        from evo2 import Evo2
        self.evo2 = Evo2(model_name, local_path=EVO2_LOCAL_PT)
        self.model = self.evo2.model
        self.tokenizer = self.evo2.tokenizer
        self._hooks = []
        self.block_names = sorted(
            [n for n, _ in self.model.named_modules() if BLOCK_RE.match(n)],
            key=lambda n: int(BLOCK_RE.match(n).group(1)))
        self.hidden_size = 4096

    # ---- generation ----
    def generate(self, prompts, n_tokens=None, temperature=None, top_k=None,
                 top_p=None, seed=0):
        set_seed(seed)
        h = HARNESS
        out = self.evo2.generate(
            prompt_seqs=list(prompts),
            n_tokens=n_tokens or h["gen_n_tokens"],
            temperature=temperature if temperature is not None else h["gen_temperature"],
            top_k=top_k or h["gen_top_k"],
            top_p=top_p if top_p is not None else h["gen_top_p"],
            batched=True, cached_generation=True, verbose=0)
        # vortex returns a GenerationOutput dataclass (.sequences); older versions returned a tuple
        return out.sequences if hasattr(out, "sequences") else out[0]

    # ---- per-token NLL (naturalness / coding plausibility) ----
    def sequence_nll(self, dna_seqs):
        """Mean per-token negative log-likelihood under Evo2 for each sequence."""
        import torch
        out = []
        for s in dna_seqs:
            s = _clean_dna(s)
            if len(s) < 2:
                out.append(float("nan")); continue
            ids = torch.tensor(self.tokenizer.tokenize(s), dtype=torch.long)[None].cuda()
            with torch.no_grad():
                logits, _ = self.evo2.forward(ids)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            V = logits.shape[-1]
            # robust to (batch, seq, vocab) or (seq, vocab): flatten to (seq, vocab) for batch=1
            logp = torch.log_softmax(logits.reshape(-1, V).float(), dim=-1)   # (seq, vocab)
            tgt = ids[0, 1:]                                                  # (seq-1,)
            nll = -logp[:-1].gather(1, tgt[:, None]).squeeze(1).mean().item()
            out.append(nll)
        return out

    # ---- activation extraction over blocks (mean-pooled residual) ----
    def block_activations(self, dna_seqs, block_idxs):
        """Return {block_idx: np.array[n_seq, hidden]} mean-pooled over positions."""
        import torch
        names = [f"blocks.{i}" for i in block_idxs]
        acc = {i: [] for i in block_idxs}
        for s in dna_seqs:
            s = _clean_dna(s)
            if len(s) < 2:
                for i in block_idxs: acc[i].append(np.zeros(self.hidden_size, np.float32))
                continue
            ids = torch.tensor(self.tokenizer.tokenize(s), dtype=torch.long)[None].cuda()
            _, emb = self.evo2.forward(ids, return_embeddings=True, layer_names=names)
            for i, name in zip(block_idxs, names):
                a = emb[name]
                if isinstance(a, tuple): a = a[0]
                acc[i].append(a[0].float().mean(0).cpu().numpy())
        return {i: np.stack(acc[i]) for i in block_idxs}

    # ---- steering hooks (CAA additive / SAE clamp) ----
    def add_steering_hook(self, block_idx, vector, coef):
        import torch
        vec = torch.as_tensor(vector, dtype=torch.float32).cuda()
        mod = self.model.get_submodule(f"blocks.{block_idx}")
        def hook(_, __, output):
            if isinstance(output, tuple):
                h0 = output[0]
                h0 = h0 + (coef * vec).to(h0.dtype)
                return (h0,) + tuple(output[1:])
            return output + (coef * vec).to(output.dtype)
        self._hooks.append(mod.register_forward_hook(hook))

    def add_sae_clamp_hook(self, block_idx, sae, feature_idxs, clamp_value):
        """Encode residual -> set selected features to clamp_value -> decode -> replace."""
        import torch
        mod = self.model.get_submodule(f"blocks.{block_idx}")
        feats = torch.as_tensor(feature_idxs, dtype=torch.long).cuda()
        def hook(_, __, output):
            h0 = output[0] if isinstance(output, tuple) else output
            orig_dtype = h0.dtype
            x = h0.float()
            f = sae.encode(x)
            f[..., feats] = clamp_value
            x2 = sae.decode(f)
            x2 = x2.to(orig_dtype)
            if isinstance(output, tuple):
                return (x2,) + tuple(output[1:])
            return x2
        self._hooks.append(mod.register_forward_hook(hook))

    def clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

# ---------------------------------------------------------------------------
# Goodfire Evo-2 Layer-26 TopK SAE (expansion 8, k=64) loader
# ---------------------------------------------------------------------------
class TopKSAE:
    """Goodfire Evo-2 Layer-26-Mixed TopK SAE (expansion 8 -> 32768 features, k=64).

    Real checkpoint layout (verified against the file on /mnt/quarkfs): a single tied weight
    `_orig_mod.W` of shape (d_model=4096, n_features=32768), plus `_orig_mod.b_enc` (32768,)
    and `_orig_mod.b_dec` (4096,). Standard tied TopK-SAE:
        encode(x) = TopK_k( relu( (x - b_dec) @ W + b_enc ) )
        decode(f) = f @ W.T + b_dec
    `_orig_mod.` is the torch.compile prefix and is stripped on load.
    """
    def __init__(self, ckpt_path=None, device="cuda", k=64):
        import torch
        self.torch = torch
        ckpt_path = ckpt_path or SAE_PATH
        sd = torch.load(ckpt_path, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        # strip torch.compile `_orig_mod.` prefix
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        self.sd = sd

        def find(*cands):
            for c in cands:
                for key in sd:
                    if key == c or key.endswith("." + c):
                        return sd[key]
            return None

        W = find("W", "W_enc", "encoder.weight", "w_enc")
        if W is None:
            raise KeyError(f"SAE weight not found; keys={list(sd.keys())}")
        W = W.float()
        # Orient W as (d_model, n_features): d_model is the smaller axis (4096 vs 32768).
        if W.shape[0] > W.shape[1]:
            W = W.t().contiguous()
        self.W = W.to(device)                      # (d_model, n_features)
        self.d_model, self.n_features = self.W.shape
        b_enc = find("b_enc", "encoder.bias")
        b_dec = find("b_dec", "decoder.bias", "pre_bias")
        self.b_enc = b_enc.float().to(device) if b_enc is not None else None   # (n_features,)
        self.b_dec = b_dec.float().to(device) if b_dec is not None else None   # (d_model,)
        # tied decoder weight = W.T  (n_features, d_model)
        W_dec = find("W_dec", "decoder.weight", "w_dec")
        self.W_dec = (W_dec.float().to(device) if W_dec is not None else self.W.t().contiguous())
        if self.W_dec.shape[0] != self.n_features:   # ensure (n_features, d_model)
            self.W_dec = self.W_dec.t().contiguous()
        self.k = k
        self.device = device

    def encode(self, x):
        torch = self.torch
        x = x.float()
        if self.b_dec is not None:
            x = x - self.b_dec
        f = x @ self.W                              # (.., n_features)
        if self.b_enc is not None:
            f = f + self.b_enc
        f = torch.relu(f)
        if self.k and f.shape[-1] > self.k:
            val, idx = torch.topk(f, self.k, dim=-1)
            f = torch.zeros_like(f).scatter_(-1, idx, val)
        return f

    def decode(self, f):
        x = f @ self.W_dec                          # (.., d_model)
        if self.b_dec is not None:
            x = x + self.b_dec
        return x

    def reconstruction_rel_error(self, x):
        """Sanity metric: ||x - decode(encode(x))|| / ||x||; small (<~0.5) => orientation OK."""
        torch = self.torch
        x = x.float()
        xhat = self.decode(self.encode(x))
        return float((torch.norm(x - xhat) / (torch.norm(x) + 1e-9)).item())

    def reconstruction_rel_error_normalized(self, x):
        """Recon after scaling each row to norm sqrt(d_model) (the common Goodfire/EleutherAI
        SAE input-normalization) and rescaling the output back. If this is small but the raw
        version is not, the checkpoint expects normalized inputs."""
        torch = self.torch
        x = x.float()
        scale = x.norm(dim=-1, keepdim=True) / (self.d_model ** 0.5) + 1e-9
        xhat = self.decode(self.encode(x / scale)) * scale
        return float((torch.norm(x - xhat) / (torch.norm(x) + 1e-9)).item())

# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def dev_test_split(n, dev_frac=0.5, seed=1234):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * dev_frac)
    return set(idx[:cut].tolist()), set(idx[cut:].tolist())

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)

def freeze_harness(out_path):
    save_json(dict(HARNESS=HARNESS, primers=DNA_PRIMERS,
                   codon_table="standard"), out_path)
