"""External scorers for inference-time beam search.

Mirrors Evo2's design (Fig. 6): an ENSEMBLE of supervised predictors, external to and independent
of the generative model, that accepts/rejects partially generated sequences.

Evo2 used Enformer + Borzoi (chromatin accessibility). Those predict mammalian chromatin tracks and
have no mapping to alpha-helix content of a prokaryotic CDS, so the ensemble is re-instantiated for
this target property with two independent sequence-level secondary-structure scorers:
  S_cf    Chou-Fasman helix propensity      (1978 statistical scale, CPU, ~0 cost)
  S_probe ESM-2 650M + supervised SS3 head  (trained on natural CDS + experimental DSSP labels)
Ensemble = mean of per-candidate z-scores (scales are incommensurable).

Candidates whose ORF terminates early (internal stop) are REJECTED (-inf), mirroring Evo2's
accept/reject scoring function.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch

# Chou & Fasman (1978) helix propensity P(alpha)
CF_ALPHA = {"A": 1.42, "R": 0.98, "N": 0.67, "D": 1.01, "C": 0.70, "Q": 1.11, "E": 1.51,
            "G": 0.57, "H": 1.00, "I": 1.08, "L": 1.21, "K": 1.16, "M": 1.45, "F": 1.13,
            "P": 0.57, "S": 0.77, "T": 0.83, "W": 1.08, "Y": 0.69, "V": 1.06}


def chou_fasman(prots):
    out = np.zeros(len(prots), dtype=np.float32)
    for i, p in enumerate(prots):
        if not p:
            continue
        v = [CF_ALPHA.get(a) for a in p]
        v = [x for x in v if x is not None]
        out[i] = float(np.mean(v)) if v else 0.0
    return out


class ESM2HelixProbe:
    """S1: ESM-2 650M embeddings + trained per-residue helix head -> mean P(helix)."""

    def __init__(self, ckpt, device="cuda:0", batch=64, max_len=1022):
        import mc_env
        from transformers import AutoTokenizer, AutoModel
        from train_ss_probe import Head
        self.device, self.batch, self.max_len = device, batch, max_len
        self.tok = AutoTokenizer.from_pretrained(mc_env.ESM2_650M_PATH)
        self.esm = AutoModel.from_pretrained(mc_env.ESM2_650M_PATH).eval().half().to(device)
        sd = torch.load(ckpt, map_location="cpu")
        self.head = Head().to(device)
        self.head.load_state_dict(sd["head"])
        self.head.eval()
        self.report = sd.get("report", {})

    @torch.no_grad()
    def __call__(self, prots):
        out = np.zeros(len(prots), dtype=np.float32)
        idx = [i for i, p in enumerate(prots) if p and len(p) >= 5]
        for s in range(0, len(idx), self.batch):
            sub = idx[s:s + self.batch]
            seqs = [prots[i][:self.max_len] for i in sub]
            enc = self.tok(seqs, return_tensors="pt", padding=True,
                           add_special_tokens=True).to(self.device)
            h = self.esm(**enc).last_hidden_state
            logit = self.head(h.float())                      # (B, T)
            prob = torch.sigmoid(logit)
            mask = enc["attention_mask"].clone()
            mask[:, 0] = 0                                    # drop <cls>
            for j, i in enumerate(sub):
                L = len(seqs[j])
                out[i] = float(prob[j, 1:1 + L].mean())
        return out


class Ensemble:
    """Evo2-style scorer ensemble with accept/reject. Tracks wall-clock cost and call count."""

    def __init__(self, probe=None, use_cf=True, reject_internal_stop=True):
        self.probe, self.use_cf = probe, use_cf
        self.reject = reject_internal_stop
        self.n_calls = 0
        self.seconds = 0.0

    def __call__(self, prots, has_internal_stop=None):
        t0 = time.time()
        parts = []
        if self.use_cf:
            parts.append(chou_fasman(prots))
        if self.probe is not None:
            parts.append(self.probe(prots))
        zs = []
        for v in parts:
            sd = v.std()
            zs.append((v - v.mean()) / sd if sd > 1e-8 else np.zeros_like(v))
        score = np.mean(zs, axis=0) if zs else np.zeros(len(prots), dtype=np.float32)
        if self.reject and has_internal_stop is not None:
            score = np.where(np.asarray(has_internal_stop), -1e9, score)
        self.n_calls += len(prots)
        self.seconds += time.time() - t0
        return score
