"""Train the cheap external secondary-structure scorer (S1): ESM-2 650M + per-residue helix head.

This is the analogue of Evo2's Enformer/Borzoi: a SUPERVISED predictor trained on NATURAL data,
independent of the generative model, and ~100x cheaper than a structure predictor.

Labels come from round-2's M0 datasets (natural CDS + real DSSP from experimental PDB); helix is
the same HGI definition used by the paper's endpoint. Splits are by mmseqs homology cluster.

Because the beam-search scorer sees PARTIAL proteins, we additionally report held-out performance
on random truncations (20-120 aa), which is the regime it will actually run in.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
import mc_env; mc_env.patch()
import numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HELIX_HGI = set("HGI")
MAXLEN = 1022


def load_data():
    items = []
    for org in ("prokaryote", "eukaryote"):
        clus = json.load(open(os.path.join(mc_env.DATA_DIR, f"m0_clusters_{org}.json")))
        for line in open(os.path.join(mc_env.DATA_DIR, f"m0_dataset_{org}.jsonl")):
            r = json.loads(line)
            seq, ss = r["seq"][:MAXLEN], r["codon_ss"][:MAXLEN]
            n = min(len(seq), len(ss))
            seq, ss = seq[:n], ss[:n]
            lab = np.array([1 if (s in HELIX_HGI) else (0 if s else -1) for s in ss], dtype=np.int8)
            if (lab >= 0).sum() < 30:
                continue
            items.append({"acc": r["acc"], "org": org, "seq": seq, "lab": lab,
                          "cluster": f"{org}:{clus.get(r['acc'], r['acc'])}"})
    return items


@torch.no_grad()
def embed(items, device, batch=4):
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(mc_env.ESM2_650M_PATH)
    mdl = AutoModel.from_pretrained(mc_env.ESM2_650M_PATH).eval().half().to(device)
    t0 = time.time()
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        enc = tok([c["seq"] for c in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=True).to(device)
        h = mdl(**enc).last_hidden_state              # (B, 1+L+1, 1280)
        for j, c in enumerate(chunk):
            L = len(c["seq"])
            c["emb"] = h[j, 1:1 + L].float().cpu().numpy().astype(np.float16)
        if i % 200 == 0:
            print(f"[probe] embed {i}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    del mdl
    torch.cuda.empty_cache()
    return items


class Head(nn.Module):
    def __init__(self, d=1280, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(mc_env.MC, "results", "ss_probe.pt"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda:0"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    items = load_data()
    clusters = sorted({c["cluster"] for c in items})
    rng.shuffle(clusters)
    n_test = max(1, int(0.2 * len(clusters)))
    test_cl = set(clusters[:n_test])
    print(f"[probe] {len(items)} proteins, {len(clusters)} clusters -> "
          f"{len(test_cl)} test clusters (homology-split)", flush=True)

    items = embed(items, dev)
    tr = [c for c in items if c["cluster"] not in test_cl]
    te = [c for c in items if c["cluster"] in test_cl]
    Xtr = np.concatenate([c["emb"][c["lab"] >= 0] for c in tr]).astype(np.float32)
    ytr = np.concatenate([c["lab"][c["lab"] >= 0] for c in tr]).astype(np.float32)
    Xte = np.concatenate([c["emb"][c["lab"] >= 0] for c in te]).astype(np.float32)
    yte = np.concatenate([c["lab"][c["lab"] >= 0] for c in te]).astype(np.float32)
    print(f"[probe] train residues {len(ytr)} (helix {ytr.mean():.3f}) | test {len(yte)}", flush=True)

    head = Head().to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    Xtr_t = torch.from_numpy(Xtr).to(dev)
    ytr_t = torch.from_numpy(ytr).to(dev)
    bs = 8192
    for ep in range(args.epochs):
        perm = torch.randperm(len(ytr_t), device=dev)
        tot = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            loss = nn.functional.binary_cross_entropy_with_logits(head(Xtr_t[idx]), ytr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        with torch.no_grad():
            pte = torch.sigmoid(head(torch.from_numpy(Xte).to(dev))).cpu().numpy()
        print(f"[probe] epoch {ep} loss={tot/len(ytr_t):.4f} test_residue_AUROC={roc_auc_score(yte, pte):.4f}",
              flush=True)

    # protein-level: what the beam-search scorer actually uses (mean predicted helix prob)
    def prot_eval(subset, trunc=None):
        pr, tv = [], []
        with torch.no_grad():
            for c in subset:
                m = c["lab"] >= 0
                if trunc is not None:
                    L = min(trunc, len(c["seq"]))
                    m = m.copy(); m[L:] = False
                if m.sum() < 15:
                    continue
                p = torch.sigmoid(head(torch.from_numpy(c["emb"][m].astype(np.float32)).to(dev)))
                pr.append(float(p.mean())); tv.append(float((c["lab"][m] == 1).mean()))
        return spearmanr(pr, tv)[0], len(pr)

    report = {"test_residue_auroc": float(roc_auc_score(yte, pte)), "n_test_clusters": len(test_cl)}
    for trunc in (None, 120, 80, 40, 20):
        rho, n = prot_eval(te, trunc)
        key = "full" if trunc is None else f"trunc{trunc}"
        report[f"protein_rho_{key}"] = float(rho)
        print(f"[probe] protein-level Spearman rho ({key}, n={n}) = {rho:.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"head": head.state_dict(), "report": report,
                "test_clusters": sorted(test_cl)}, args.out)
    json.dump(report, open(args.out.replace(".pt", "_report.json"), "w"), indent=2)
    print(f"[probe] WROTE {args.out}", flush=True)


if __name__ == "__main__":
    main()
