"""Local per-residue secondary-structure predictor (documented eval-tool fallback
for the folding/SS readout when ESMFold weights are impractical to fetch offline).

Backbone: ESM-2 650M (facebook/esm2_t33_650M_UR50D, cached locally) — a protein LM
INDEPENDENT of Evo2 and its SAE. A linear probe on per-residue embeddings predicts
DSSP 3-state SS (H/E/C). Trained on the S0 E. coli residues (train split), validated
on val, tested on held-out test genes. Readout = predicted %helix of a protein.

This is NOT a model/SAE downscale — Evo2-7B + the released Layer-26 SAE remain exact;
this only substitutes the *readout tool* (sequence->SS predictor) for ESMFold->DSSP,
per FINAL_PROPOSAL risk mitigation and EXPERIMENT_PLAN budget-guard note.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ESM2 = "facebook/esm2_t33_650M_UR50D"
PROBE_PATH = os.path.join(ROOT, "results/ss_probe.pt")
STATES = ["H", "E", "C"]
S2I = {s: i for i, s in enumerate(STATES)}
MAXLEN = 1022


class ESM2Backbone:
    def __init__(self, device="cuda:0"):
        from transformers import AutoTokenizer, EsmModel
        self.tok = AutoTokenizer.from_pretrained(ESM2)
        self.model = EsmModel.from_pretrained(ESM2).to(device).eval().half()
        self.device = device
        self.dim = self.model.config.hidden_size   # 1280

    @torch.no_grad()
    def embed(self, seqs):
        """Return list of per-residue embedding tensors (Li, dim) float32 CPU,
        aligned to residues (special tokens stripped)."""
        enc = self.tok(seqs, return_tensors="pt", padding=True, truncation=True,
                       max_length=MAXLEN + 2, add_special_tokens=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc).last_hidden_state          # (B,T,dim)
        embs = []
        for i, s in enumerate(seqs):
            L = min(len(s), MAXLEN)
            # residues occupy positions 1..L (0 is <cls>)
            embs.append(out[i, 1:1+L].float().cpu())
        return embs


class SSProbe(nn.Module):
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, 3))
    def forward(self, x):
        return self.net(x)


class SSPredictor:
    """Readout backend with the same interface as ESMFolder.helix_readout."""
    name = "esm2_probe"
    def __init__(self, device="cuda:0", probe_path=PROBE_PATH, hi_conf=0.80):
        self.bb = ESM2Backbone(device)
        self.probe = SSProbe(self.bb.dim).to(device)
        self.probe.load_state_dict(torch.load(probe_path, map_location=device))
        self.probe.eval()
        self.device = device
        self.hi_conf = hi_conf

    @torch.no_grad()
    def helix_readout(self, aa_seq):
        emb = self.bb.embed([aa_seq])[0].to(self.device).half().float()
        logits = self.probe(emb)                            # (L,3)
        prob = torch.softmax(logits, -1)
        conf, pred = prob.max(-1)
        pred = pred.cpu().numpy(); conf = conf.cpu().numpy()
        ss3 = "".join(STATES[i] for i in pred)
        hi = conf >= self.hi_conf
        def frac(mask, state):
            sel = pred[mask] if mask is not None else pred
            if len(sel) == 0:
                return float("nan")
            return float(np.mean(sel == S2I[state]))
        return dict(
            helix_all=frac(None, "H"), sheet_all=frac(None, "E"),
            coil_all=frac(None, "C"),
            helix_hi=frac(hi, "H") if hi.sum() else float("nan"),
            n_hi=int(hi.sum()), mean_plddt=float(conf.mean()*100.0),
            ss_len=len(ss3), ss3=ss3)


# ----------------------------------------------------------------- training
def train_probe(device="cuda:0", epochs=8, batch_genes=16):
    import pandas as pd
    gdf = pd.read_parquet(os.path.join(ROOT, "data/ecoli_genes.parquet"))
    bb = ESM2Backbone(device)
    print(f"[ssprobe] ESM-2 backbone dim={bb.dim}", flush=True)

    def collect(split):
        sub = gdf[gdf.split == split]
        X, Y = [], []
        seqs = sub["aa"].tolist(); sss = sub["ss3"].tolist()
        t0 = time.time()
        for b0 in range(0, len(seqs), batch_genes):
            bs = seqs[b0:b0+batch_genes]; bss = sss[b0:b0+batch_genes]
            embs = bb.embed(bs)
            for e, ss in zip(embs, bss):
                L = min(e.shape[0], len(ss))
                X.append(e[:L]); Y.append(np.array([S2I[c] for c in ss[:L]]))
            if (b0 // batch_genes) % 20 == 0:
                print(f"[ssprobe]  embed {split} {b0}/{len(seqs)} {time.time()-t0:.0f}s",
                      flush=True)
        return torch.cat(X), torch.tensor(np.concatenate(Y))

    Xtr, Ytr = collect("train")
    Xva, Yva = collect("val")
    Xte, Yte = collect("test")
    print(f"[ssprobe] residues train={len(Ytr)} val={len(Yva)} test={len(Yte)}", flush=True)

    probe = SSProbe(bb.dim).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Xtr_d = Xtr.to(device); Ytr_d = Ytr.to(device)
    n = len(Ytr_d); bs = 65536
    best_va, best_state = -1, None
    for ep in range(epochs):
        probe.train(); perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = lossf(probe(Xtr_d[idx]), Ytr_d[idx])
            loss.backward(); opt.step()
        # val acc
        probe.eval()
        with torch.no_grad():
            va = (probe(Xva.to(device)).argmax(-1).cpu() == Yva).float().mean().item()
        print(f"[ssprobe] epoch {ep} val_acc={va:.3f}", flush=True)
        if va > best_va:
            best_va = va; best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    torch.save(probe.state_dict(), PROBE_PATH)
    # test metrics
    probe.eval()
    with torch.no_grad():
        pred = probe(Xte.to(device)).argmax(-1).cpu().numpy()
    yte = Yte.numpy()
    acc = float((pred == yte).mean())
    # per-class + helix F1
    def f1(cls):
        tp = np.sum((pred == cls) & (yte == cls)); fp = np.sum((pred == cls) & (yte != cls))
        fn = np.sum((pred != cls) & (yte == cls))
        p = tp/(tp+fp+1e-9); r = tp/(tp+fn+1e-9)
        return float(2*p*r/(p+r+1e-9))
    # gene-level helix-fraction correlation (readout fidelity for the C2/C3 estimand)
    meta = dict(test_acc=acc, helix_f1=f1(0), sheet_f1=f1(1), coil_f1=f1(2),
                best_val_acc=best_va, backbone=ESM2, n_train_res=int(len(Ytr)),
                n_test_res=int(len(Yte)))
    with open(os.path.join(ROOT, "results/ss_probe_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[ssprobe] TEST acc={acc:.3f} helixF1={meta['helix_f1']:.3f} "
          f"sheetF1={meta['sheet_f1']:.3f} coilF1={meta['coil_f1']:.3f}", flush=True)
    print(f"[ssprobe] saved probe -> {PROBE_PATH}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    if args.train:
        train_probe(epochs=args.epochs)
