"""
E1 alpha-helix-content evaluation harness (Claim 2 infrastructure).

Two %H estimators:
  (1) FAST sequence-based predictor = ESM2-650M frozen embeddings + a linear
      3-state (H/E/C) secondary-structure probe trained on experimental DSSP labels.
      This is the online scorer used for M1 labeling cross-check and M2/M3/M4 %H.
  (2) STRUCTURE reference = DSSP on a 3D structure. For natural validation proteins
      the structure is the real experimental PDB crystal structure (gold standard).
      For generated sequences (no experimental structure) an optional ESMFold pass
      provides a predicted-structure DSSP %H (used for M4 subset when available).

%H convention: DSSP classes H,G,I -> helix.  3-state map: H,G,I->H ; E,B->E ; else C.
"""
import os, subprocess, tempfile, warnings
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import numpy as np
import torch

warnings.filterwarnings("ignore")

DSSP3 = {'H':'H','G':'H','I':'H','E':'E','B':'E','T':'C','S':'C','P':'C','-':'C',' ':'C','C':'C'}

# ---------------- DSSP on a PDB file ----------------
def dssp_ss_from_pdb(pdb_path):
    """Return (aa_seq, ss8_string) using mkdssp via Biopython. None on failure."""
    from Bio.PDB import PDBParser, DSSP
    try:
        s = PDBParser(QUIET=True).get_structure("x", pdb_path)
        model = s[0]
        d = DSSP(model, pdb_path, dssp="mkdssp")
        aas = []; ss = []
        for k in d.keys():
            rec = d[k]
            aas.append(rec[1]); ss.append(rec[2])
        return "".join(aas), "".join(ss)
    except Exception:
        return None

def pct_helix_from_ss8(ss8):
    if not ss8: return float('nan')
    H = sum(DSSP3.get(c, 'C') == 'H' for c in ss8)
    return 100.0 * H / len(ss8)

def pct_sheet_from_ss8(ss8):
    if not ss8: return float('nan')
    E = sum(DSSP3.get(c, 'C') == 'E' for c in ss8)
    return 100.0 * E / len(ss8)

def ss3_labels(ss8):
    return [DSSP3.get(c, 'C') for c in ss8]

# ---------------- ESM2-650M fast SS predictor ----------------
class ESM2SSProbe:
    """ESM2-650M frozen embeddings -> linear 3-state SS probe (per-residue)."""
    LABELS = ['H', 'E', 'C']
    def __init__(self, device="cuda", layer=33):
        self.device = device; self.layer = layer
        self._loaded = False
        self.W = None; self.b = None  # logistic weights [d,3],[3]

    def _load_esm(self):
        if self._loaded: return
        from transformers import AutoTokenizer, AutoModel
        self.tok = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        self.esm = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D",
                                             torch_dtype=torch.float16).to(self.device).eval()
        self.d = self.esm.config.hidden_size
        self._loaded = True

    @torch.no_grad()
    def embed(self, seq):
        """Per-residue embedding [L, d] (float32, CPU) for an amino-acid sequence."""
        self._load_esm()
        seq = seq[:1022]
        enc = self.tok(seq, return_tensors="pt", add_special_tokens=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.esm(**enc).last_hidden_state[0]  # [L+2, d]
        emb = out[1:1+len(seq)].float().cpu().numpy()
        return emb

    def fit(self, seqs, ss8_list, max_C=1.0, l2=1.0):
        """Train logistic-regression SS probe on per-residue embeddings."""
        from sklearn.linear_model import LogisticRegression
        X = []; Y = []
        for s, ss in zip(seqs, ss8_list):
            n = min(len(s), len(ss))
            if n < 5: continue
            emb = self.embed(s[:n])
            lab = ss3_labels(ss[:n])
            m = min(len(emb), len(lab))
            X.append(emb[:m]); Y.extend(lab[:m])
        X = np.concatenate(X, 0); Y = np.array(Y)
        clf = LogisticRegression(max_iter=2000, C=l2, n_jobs=-1)
        clf.fit(X, Y)
        self.clf = clf
        self.classes_ = list(clf.classes_)
        return clf.score(X, Y)

    def predict_ss3(self, seq):
        emb = self.embed(seq)
        pred = self.clf.predict(emb)
        return "".join(pred)

    def pct_helix(self, seq):
        if len(seq) == 0: return float('nan')
        ss3 = self.predict_ss3(seq)
        H = sum(c == 'H' for c in ss3)
        return 100.0 * H / len(ss3)

    def pct_sheet(self, seq):
        if len(seq) == 0: return float('nan')
        ss3 = self.predict_ss3(seq)
        E = sum(c == 'E' for c in ss3)
        return 100.0 * E / len(ss3)

    def save(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"coef": self.clf.coef_, "intercept": self.clf.intercept_,
                         "classes": self.classes_}, f)

    def load_probe(self, path):
        import pickle
        from sklearn.linear_model import LogisticRegression
        with open(path, "rb") as f:
            d = pickle.load(f)
        clf = LogisticRegression(); clf.coef_ = d["coef"]; clf.intercept_ = d["intercept"]
        clf.classes_ = np.array(d["classes"]); self.clf = clf; self.classes_ = list(d["classes"])
        clf.n_features_in_ = d["coef"].shape[1]

# ---------------- optional ESMFold structure prediction ----------------
class ESMFolder:
    def __init__(self, device="cuda"):
        from transformers import AutoTokenizer, EsmForProteinFolding
        self.tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", torch_dtype=torch.float16).to(device).eval()
        self.model.esm = self.model.esm.half()

    @torch.no_grad()
    def fold_pdb(self, seq):
        return self.model.infer_pdb(seq[:400])

    def pct_helix_sheet(self, seq):
        pdb = self.fold_pdb(seq)
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
            f.write(pdb); p = f.name
        r = dssp_ss_from_pdb(p)
        os.unlink(p)
        if r is None: return float('nan'), float('nan')
        _, ss8 = r
        return pct_helix_from_ss8(ss8), pct_sheet_from_ss8(ss8)

def esmfold_available():
    done = os.path.join(os.path.dirname(__file__), "..", "assets", ".esmfold_dl_done")
    return os.path.exists(done)
