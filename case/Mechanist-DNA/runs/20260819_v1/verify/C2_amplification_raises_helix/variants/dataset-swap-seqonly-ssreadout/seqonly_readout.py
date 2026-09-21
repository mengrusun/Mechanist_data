"""VERIFY C2 — DATASET/READOUT swap variant.

Replaces the main experiment's ESM-2-650M + probe predicted-%-helix readout with a
GENUINELY INDEPENDENT, protein-LM-free readout: a windowed amino-acid (GOR/PSSM-style)
3-state secondary-structure predictor. Trained on the SAME S0 DSSP labels (train-split
genes), validated on held-out TEST-split genes, then applied to the EXACT cached
generations from the main experiment (results/m3final_a0.json = baseline,
results/m3final_astar.json = alpha*=1). Only the readout TOOL changes; the generated
sequences are held identical (no regeneration).

Tests C2's flagged readout-tool sensitivity: does the alpha*=1 predicted-%-helix gain
survive an orthogonal, non-neural SS predictor?

Outputs result.json.
"""
import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../../.."))
GENES = os.path.join(ROOT, "data/ecoli_genes.parquet")
BASELINE = os.path.join(ROOT, "results/m3final_a0.json")     # alpha=0
STEERED  = os.path.join(ROOT, "results/m3final_astar.json")  # alpha*=1

AAS = "ACDEFGHIKLMNPQRSTVWY"
A2I = {a: i for i, a in enumerate(AAS)}
PAD = len(AAS)                       # index 20 = pad/unknown
NF = len(AAS) + 1                    # 21 symbols one-hot
STATES = ["H", "E", "C"]
S2I = {s: i for i, s in enumerate(STATES)}
WIN = 8                              # +/- 8 residues -> 17-wide window
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"

# Chou-Fasman helix propensity Pa (classic, model-free) — secondary corroboration only
CF_PA = {"E":1.51,"M":1.45,"A":1.42,"L":1.21,"K":1.16,"F":1.13,"Q":1.11,"W":1.08,
         "I":1.08,"V":1.06,"D":1.01,"H":1.00,"R":0.98,"T":0.83,"S":0.77,"C":0.70,
         "Y":0.69,"N":0.67,"P":0.57,"G":0.57}
# helix-favoring residues (Pa > ~1.1) for the composition control
HELIX_FAV = set("EMALKFQ")


def seq_to_idx(aa):
    return np.array([A2I.get(c, PAD) for c in aa], dtype=np.int64)


def windowed_feats(idx):
    """(L, NF*(2*WIN+1)) one-hot windowed features for a residue index array."""
    L = len(idx)
    w = 2 * WIN + 1
    padded = np.full(L + 2 * WIN, PAD, dtype=np.int64)
    padded[WIN:WIN + L] = idx
    # build (L, w) index matrix
    cols = np.stack([padded[i:i + L] for i in range(w)], axis=1)   # (L, w)
    oneh = np.zeros((L, w, NF), dtype=np.float32)
    rows = np.arange(L)[:, None]
    oneh[rows, np.arange(w)[None, :], cols] = 1.0
    return oneh.reshape(L, w * NF)


class WinSSNet(nn.Module):
    def __init__(self, din, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hidden), nn.GELU(),
                                 nn.Linear(hidden, 3))

    def forward(self, x):
        return self.net(x)


def build_split(gdf, split):
    X, Y = [], []
    sub = gdf[gdf.split == split]
    for aa, ss in zip(sub["aa"].tolist(), sub["ss3"].tolist()):
        L = min(len(aa), len(ss))
        if L < 5:
            continue
        X.append(windowed_feats(seq_to_idx(aa[:L])))
        Y.append(np.array([S2I.get(c, 2) for c in ss[:L]], dtype=np.int64))
    return np.concatenate(X), np.concatenate(Y)


def train_predictor():
    import pandas as pd
    gdf = pd.read_parquet(GENES)
    t0 = time.time()
    # FIT on train, EARLY-STOP on the existing held-out VAL split, report FINAL on
    # the untouched TEST split (no test-set model selection / leakage).
    Xtr, Ytr = build_split(gdf, "train")
    Xva, Yva = build_split(gdf, "val")
    Xte, Yte = build_split(gdf, "test")
    din = Xtr.shape[1]
    assert din == (2 * WIN + 1) * NF, (din, (2 * WIN + 1) * NF)   # 17 * 21 = 357
    print(f"[seqonly] windowed feats din={din} train_res={len(Ytr)} val_res={len(Yva)} "
          f"test_res={len(Yte)} ({time.time()-t0:.0f}s)", flush=True)
    net = WinSSNet(din).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    Xtr_t = torch.tensor(Xtr, device=DEV); Ytr_t = torch.tensor(Ytr, device=DEV)
    Xva_t = torch.tensor(Xva, device=DEV)
    n = len(Ytr_t); bs = 65536
    best_val, best_state = -1, None
    for ep in range(12):
        net.train(); perm = torch.randperm(n, device=DEV)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = lossf(net(Xtr_t[idx]), Ytr_t[idx])
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vpred = net(Xva_t).argmax(-1).cpu().numpy()
        vacc = float((vpred == Yva).mean())            # early-stop on VAL, never TEST
        if vacc > best_val:
            best_val = vacc; best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
        print(f"[seqonly] epoch {ep} val_acc={vacc:.3f}", flush=True)
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(Xte, device=DEV)).argmax(-1).cpu().numpy()   # untouched TEST

    def f1(cls):
        tp = np.sum((pred == cls) & (Yte == cls)); fp = np.sum((pred == cls) & (Yte != cls))
        fn = np.sum((pred != cls) & (Yte == cls))
        p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
        return float(2 * p * r / (p + r + 1e-9))
    meta = dict(test_acc=float((pred == Yte).mean()), helix_f1=f1(0), sheet_f1=f1(1),
                coil_f1=f1(2), best_val_acc=best_val, window=WIN, hidden=256, din=din,
                n_train_res=int(len(Ytr)), n_val_res=int(len(Yva)), n_test_res=int(len(Yte)),
                readout_name="windowed_seqonly_gor", helix_class_index=S2I["H"])
    print(f"[seqonly] HELD-OUT test acc={meta['test_acc']:.3f} helixF1={meta['helix_f1']:.3f} "
          f"sheetF1={meta['sheet_f1']:.3f} coilF1={meta['coil_f1']:.3f}", flush=True)
    torch.save(best_state, os.path.join(HERE, "winss_probe.pt"))
    return net, meta


@torch.no_grad()
def predict_helix_frac(net, aa):
    """Predicted %-helix (fraction of residues classified H) for one protein."""
    if len(aa) < 5:
        return float("nan")
    X = torch.tensor(windowed_feats(seq_to_idx(aa)), device=DEV)
    pred = net(X).argmax(-1).cpu().numpy()
    return float(np.mean(pred == 0))


def cf_helix_score(aa):
    """Mean Chou-Fasman helix propensity Pa over the sequence (model-free secondary)."""
    if not aa:
        return float("nan")
    return float(np.mean([CF_PA.get(c, 1.0) for c in aa]))


def helix_fav_frac(aa):
    if not aa:
        return float("nan")
    return float(np.mean([c in HELIX_FAV for c in aa]))


def load_proteins(path):
    """Filter ONLY on pre-readout criteria (qc_ok & protein) — NOT on the old ESM-2
    readout's output — so the new-readout sample cannot depend on the old outcome.
    Returns (list[(aa, esm_helix_or_nan)], n_per_seq, n_qc_protein)."""
    d = json.load(open(path))
    out = []; n_total = len(d["per_seq"])
    for p in d["per_seq"]:
        if p.get("qc_ok") and p.get("protein"):
            esm = p["helix_all"] if ("helix_all" in p and p["helix_all"] == p["helix_all"]) else float("nan")
            out.append((p["protein"], float(esm) if esm == esm else float("nan")))
    return out, n_total, len(out)


def boot_ci(a, b, n_boot=5000, seed=1):
    """Two-sample bootstrap CI of mean(b)-mean(a). Returns (delta, ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a); b = np.asarray(b)
    delta = float(b.mean() - a.mean())
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        diffs[i] = rng.choice(b, len(b), replace=True).mean() - rng.choice(a, len(a), replace=True).mean()
    return delta, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def perm_p(a, b, n_perm=10000, seed=2):
    """Two-sample permutation test (label shuffle) of mean(b)-mean(a), two-sided,
    with +1 correction. This is the valid null test (replaces bootstrap-sign p)."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a); b = np.asarray(b)
    obs = abs(b.mean() - a.mean())
    pool = np.concatenate([a, b]); na = len(a); n = len(pool)
    ge = 0
    for _ in range(n_perm):
        idx = rng.permutation(n)
        pa = pool[idx[:na]].mean(); pb = pool[idx[na:]].mean()
        if abs(pb - pa) >= obs:
            ge += 1
    return (1 + ge) / (1 + n_perm)


def diff_stats(a, b):
    delta, lo, hi = boot_ci(a, b)
    p = perm_p(a, b)
    return delta, lo, hi, p


def main():
    t0 = time.time()
    net, meta = train_predictor()
    base, base_ntot, base_nqc = load_proteins(BASELINE)     # alpha=0
    steer, steer_ntot, steer_nqc = load_proteins(STEERED)   # alpha*=1
    print(f"[seqonly] scoring {len(base)} baseline + {len(steer)} steered proteins ...", flush=True)

    def score(rows):
        newh, esmh, cf, favf = [], [], [], []
        for aa, esm in rows:
            nh = predict_helix_frac(net, aa)
            if nh != nh:            # require finite NEW-readout output
                continue
            newh.append(nh); esmh.append(esm)
            cf.append(cf_helix_score(aa)); favf.append(helix_fav_frac(aa))
        return (np.array(newh), np.array(esmh, dtype=float), np.array(cf), np.array(favf))

    b_new, b_esm, b_cf, b_fav = score(base)
    s_new, s_esm, s_cf, s_fav = score(steer)

    # primary: NEW readout %-helix, alpha*=1 vs baseline (permutation p + bootstrap CI)
    d_new, lo_new, hi_new, p_new = diff_stats(b_new, s_new)
    # reference: ESM-2 probe on the SAME cached set, using ITS OWN finite-value eligibility
    be = b_esm[np.isfinite(b_esm)]; se = s_esm[np.isfinite(s_esm)]
    d_esm, lo_esm, hi_esm, p_esm = diff_stats(be, se)
    # secondary: Chou-Fasman propensity (exploratory)
    d_cf, lo_cf, hi_cf, p_cf = diff_stats(b_cf, s_cf)
    # composition control: helix-favoring residue frequency
    d_fav, lo_fav, hi_fav, p_fav = diff_stats(b_fav, s_fav)
    # cross-calibration: per-seq correlation new readout vs ESM-2 probe (pooled, finite-esm)
    allnew = np.concatenate([b_new, s_new]); allesm = np.concatenate([b_esm, s_esm])
    fin = np.isfinite(allnew) & np.isfinite(allesm)
    corr = float(np.corrcoef(allnew[fin], allesm[fin])[0, 1])

    res = dict(
        variant_tag="dataset-swap-seqonly-ssreadout", dimension="dataset",
        readout_meta=meta,
        sample_counts=dict(
            baseline=dict(per_seq=base_ntot, qc_protein=base_nqc, new_readout_valid=len(b_new)),
            steered=dict(per_seq=steer_ntot, qc_protein=steer_nqc, new_readout_valid=len(s_new)),
            esm_ref=dict(baseline_finite=int(len(be)), steered_finite=int(len(se)))),
        n_baseline=len(b_new), n_steered=len(s_new),
        primary_new_readout=dict(
            baseline_helix=float(b_new.mean()), steered_helix=float(s_new.mean()),
            delta=d_new, ci=[lo_new, hi_new], p=p_new, test="two-sample permutation (two-sided, +1)"),
        reference_esm2_probe=dict(
            baseline_helix=float(be.mean()), steered_helix=float(se.mean()),
            delta=d_esm, ci=[lo_esm, hi_esm], p=p_esm),
        secondary_chou_fasman_Pa=dict(
            baseline=float(b_cf.mean()), steered=float(s_cf.mean()),
            delta=d_cf, ci=[lo_cf, hi_cf], p=p_cf),
        composition_control_helixfav_freq=dict(
            baseline=float(b_fav.mean()), steered=float(s_fav.mean()),
            delta=d_fav, ci=[lo_fav, hi_fav], p=p_fav),
        cross_calibration_corr_new_vs_esm2=corr,
        wall_s=time.time() - t0,
    )
    json.dump(res, open(os.path.join(HERE, "result.json"), "w"), indent=2)
    print(f"[seqonly] DONE ({res['wall_s']:.0f}s)")
    print(f"  predictor held-out(TEST) acc={meta['test_acc']:.3f} helixF1={meta['helix_f1']:.3f} (val-early-stopped)")
    print(f"  NEW readout (seq-only GOR): baseline {b_new.mean():.3f} -> steered {s_new.mean():.3f}  "
          f"delta {d_new:+.4f} CI[{lo_new:.4f},{hi_new:.4f}] perm_p={p_new:.4f}  (n {len(b_new)} vs {len(s_new)})")
    print(f"  ESM-2 probe (reference, same seqs): {be.mean():.3f} -> {se.mean():.3f} delta {d_esm:+.4f} perm_p={p_esm:.4f}")
    print(f"  Chou-Fasman Pa (secondary): {b_cf.mean():.3f} -> {s_cf.mean():.3f} delta {d_cf:+.4f} perm_p={p_cf:.4f}")
    print(f"  helix-fav residue freq (composition): {b_fav.mean():.3f} -> {s_fav.mean():.3f} delta {d_fav:+.4f} perm_p={p_fav:.4f}")
    print(f"  corr(new, ESM-2 probe) pooled = {corr:.3f}")


if __name__ == "__main__":
    main()
