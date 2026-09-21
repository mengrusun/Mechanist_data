"""
M5: Composition-controlled, independent-predictor re-evaluation of the C1 steered-%H endpoint.

Fixes the verify Phase-2 INCONCLUSIVE (main-experiment integrity FAIL: Check E scope +
Check F synthetic_proxy) WITHOUT changing the claim or the steering: it re-scores the
ALREADY-GENERATED steered vs baseline sequences (results/M4_library_sequences.json, 200+200;
plus results/M2_generations.json for the dose curve) with predictors INDEPENDENT of the
ESM2-650M backbone, and disentangles helix-propensity from the AT-rich-codon GC collapse.

Three independent %H estimators on the SAME proteins:
  (1) ESM2-probe   -- the original online metric (reference only).
  (2) GOR-windowed -- logistic on +/-8 aa one-hot window, fit on the SAME 600 train DSSP
                      proteins as E1. Shares NO machinery with ESM2. (Identical to the
                      verify C2 method-swap variant predictor.)
  (3) Chou-Fasman  -- classical residue helix-propensity + nucleation/extension rule.
                      Fully deterministic, no training, no ESM.

Controls the reviewer required (iteration-1):
  A. GC control: regress %H~GC within baseline; GC-STRATIFIED and nearest-neighbour
     GC-MATCHED steered-vs-baseline effect with bootstrap CI + permutation p; partial
     (composition-controlled) effect.
  B. Composition controls: amino-acid composition shift, protein length, low-complexity
     (max single-AA fraction), reported per condition; effect stratified by GC.
  C. Non-monotonicity: dose curve with per-alpha bootstrap CIs, honest Spearman, and the
     monotone high-dose working range identified explicitly.

Structural folding (ESMFold/DSSP on steered seqs) is attempted iff a LOCAL predictor is
available; the HF-mirror CDN blocked ESMFold (2.7 GB) at experiment time, so we expect to
GROUND on independent sequence predictors + composition controls and label the result
"sequence-predictor-supported, not structurally confirmed" accordingly.

Output: runs/iteration_round_1/M5_structural_gc_control.json
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import numpy as np
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from sklearn.linear_model import LogisticRegression
import evo2lib as E
from scorer import ss3_labels, ESM2SSProbe, esmfold_available

RES = os.path.join(ROOT, "results")
DATA = os.path.join(ROOT, "data")
ASSETS = os.path.join(ROOT, "assets")
OUT = os.path.join(ROOT, "runs", "iteration_round_1")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2026)

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}

# ---------- independent predictor 2: GOR-windowed (== verify C2 method-swap) ----------
W = 8
NF = len(AA) + 1
def win_features(seq):
    L = len(seq); idx = [AA_IDX.get(c, len(AA)) for c in seq]
    feats = np.zeros((L, (2*W+1)*NF), dtype=np.float32)
    for i in range(L):
        for j, off in enumerate(range(-W, W+1)):
            k = i + off
            ch = idx[k] if 0 <= k < L else len(AA)
            feats[i, j*NF+ch] = 1.0
    return feats

def fit_gor():
    prots = json.load(open(os.path.join(DATA, "proteins.json")))
    train = [p for p in prots if p["split"] == "train"][:600]
    X = []; Y = []
    for p in train:
        n = min(len(p["aa"]), len(p["ss8"]))
        if n < 5: continue
        X.append(win_features(p["aa"][:n])); Y.extend(ss3_labels(p["ss8"][:n]))
    X = np.concatenate(X, 0); Y = np.array(Y)
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1); clf.fit(X, Y)
    return clf

def gor_pctH(clf, seq):
    if len(seq) == 0: return float('nan')
    pred = clf.predict(win_features(seq))
    return 100.0 * float(np.mean(pred == 'H'))

# ---------- independent predictor 3: Chou-Fasman helix ----------
# Standard Chou-Fasman alpha-helix propensities P(a).
CF_PA = {
 'E':1.51,'M':1.45,'A':1.42,'L':1.21,'K':1.16,'F':1.13,'Q':1.11,'W':1.08,'I':1.08,
 'V':1.06,'D':1.01,'H':1.00,'R':0.98,'T':0.83,'S':0.77,'C':0.70,'Y':0.69,'N':0.67,
 'P':0.57,'G':0.57}
def cf_pctH(seq):
    """Chou-Fasman helix assignment: nucleate where a 6-window has >=4 formers (P>=1.03)
    and sum(P)>window*1.0, then extend while local P-sum stays >1.0; %H = fraction in helix."""
    n = len(seq)
    if n < 6: return float('nan')
    pa = np.array([CF_PA.get(c, 1.0) for c in seq])
    former = pa >= 1.03
    helix = np.zeros(n, dtype=bool)
    i = 0
    while i <= n - 6:
        w = slice(i, i+6)
        if former[w].sum() >= 4 and pa[w].mean() > 1.0:
            # extend left
            l = i
            while l-1 >= 0 and pa[max(0,l-4):l+1].mean() > 1.0:
                l -= 1
            # extend right
            r = i+5
            while r+1 < n and pa[r-3:min(n,r+2)].mean() > 1.0:
                r += 1
            helix[l:r+1] = True
            i = r + 1
        else:
            i += 1
    return 100.0 * float(helix.mean())

# ---------- sequence descriptors ----------
def gc(dna): return E.gc_content(dna)
def lowcomplexity(prot):
    if not prot: return float('nan')
    from collections import Counter
    return max(Counter(prot).values()) / len(prot)  # max single-AA fraction
def helixformer_frac(prot):
    if not prot: return float('nan')
    return float(np.mean([CF_PA.get(c,1.0) >= 1.03 for c in prot]))

def describe(dna_list):
    """DNA list -> per-seq dict with protein, gc, len, composition descriptors."""
    rows = []
    for s in dna_list:
        if not E.is_valid_dna(s):
            # allow non-ACGT? keep only ACGT for gc; still translate longest ORF
            pass
        prot, frame, start, orf = E.find_longest_orf(s, min_aa=15)
        rows.append({"dna": s, "prot": prot, "gc": gc(s), "len": len(prot),
                     "lowcplx": lowcomplexity(prot), "hf": helixformer_frac(prot)})
    return rows

def boot_ci(a, b, nboot=2000):
    """bootstrap CI of mean(a)-mean(b)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    diffs = []
    for _ in range(nboot):
        da = rng.choice(a, len(a), replace=True); db = rng.choice(b, len(b), replace=True)
        diffs.append(da.mean() - db.mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(a)-np.mean(b)), float(lo), float(hi)

def perm_p(a, b, nperm=5000):
    """two-sided permutation p for mean(a)-mean(b)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    obs = abs(a.mean() - b.mean()); pool = np.concatenate([a, b]); na = len(a); c = 0
    for _ in range(nperm):
        rng.shuffle(pool)
        if abs(pool[:na].mean() - pool[na:].mean()) >= obs: c += 1
    return (c + 1) / (nperm + 1)

def main():
    t0 = time.time()
    result = {"milestone": "M5", "claim": "C1", "role": "composition_controlled_reeval",
              "purpose": "independent-predictor + GC/composition control of the steered-%H endpoint",
              "grounding": None}

    print("[M5] fitting GOR-windowed independent predictor ...", flush=True)
    gor = fit_gor()

    # ESM2 probe (reference)
    probe = ESM2SSProbe(); probe.load_probe(os.path.join(ASSETS, "ss_probe.pkl"))

    # ---- load the already-generated M4 baseline vs library ----
    m4 = json.load(open(os.path.join(RES, "M4_library_sequences.json")))
    base_dna = m4["baseline"]; lib_dna = m4["library"]; chosen_alpha = m4["chosen_alpha"]
    print(f"[M5] baseline={len(base_dna)} library={len(lib_dna)} chosen_alpha={chosen_alpha}", flush=True)

    base = describe(base_dna); lib = describe(lib_dna)

    def score_all(rows, tag):
        for i, r in enumerate(rows):
            p = r["prot"]
            r["H_esm2"] = probe.pct_helix(p) if len(p) >= 5 else float('nan')
            r["H_gor"]  = gor_pctH(gor, p)
            r["H_cf"]   = cf_pctH(p)
            if (i+1) % 50 == 0: print(f"[M5] {tag} scored {i+1}/{len(rows)}", flush=True)
    score_all(base, "baseline"); score_all(lib, "library")

    def col(rows, k): return np.array([r[k] for r in rows], float)
    def clean(x, y):
        m = np.isfinite(x) & np.isfinite(y); return x[m], y[m]

    # ---------- (0) three-predictor uplift on the FULL sets ----------
    preds = {"esm2": "H_esm2", "gor": "H_gor", "cf": "H_cf"}
    uplift = {}
    for name, k in preds.items():
        hb = col(base, k); hl = col(lib, k)
        hb = hb[np.isfinite(hb)]; hl = hl[np.isfinite(hl)]
        d, lo, hi = boot_ci(hl, hb)
        try: mwu = float(mannwhitneyu(hl, hb, alternative="greater").pvalue)
        except Exception: mwu = float('nan')
        uplift[name] = {"mean_baseline": round(float(hb.mean()),2), "mean_library": round(float(hl.mean()),2),
                        "delta": round(d,2), "ci95": [round(lo,2), round(hi,2)],
                        "mwu_p_greater": mwu}
    result["full_set_uplift_by_predictor"] = uplift
    # cross-predictor agreement ON STEERED sequences specifically
    gl, cl = clean(col(lib, "H_gor"), col(lib, "H_cf"))
    el, gl2 = clean(col(lib, "H_esm2"), col(lib, "H_gor"))
    ec, cc = clean(col(lib, "H_esm2"), col(lib, "H_cf"))
    result["cross_predictor_r_on_steered"] = {
        "esm2_vs_gor": round(float(pearsonr(el, gl2)[0]), 3) if len(el)>3 else None,
        "esm2_vs_cf":  round(float(pearsonr(ec, cc)[0]), 3) if len(ec)>3 else None,
        "gor_vs_cf":   round(float(pearsonr(gl, cl)[0]), 3) if len(gl)>3 else None,
    }

    # ---------- (A) GC control ----------
    gc_b = col(base, "gc"); gc_l = col(lib, "gc")
    # regress %H ~ GC within baseline (per predictor)
    reg = {}
    for name, k in preds.items():
        x, y = clean(gc_b, col(base, k))
        if len(x) > 3:
            r_, p_ = pearsonr(x, y); slope = np.polyfit(x, y, 1)[0]
            reg[name] = {"pearson_r_H_vs_GC_baseline": round(float(r_),3), "p": float(p_),
                         "slope_pctH_per_GC": round(float(slope),1)}
    result["gc_regression_within_baseline"] = reg
    result["gc_distribution"] = {
        "baseline": {"mean": round(float(gc_b.mean()),3), "min": round(float(gc_b.min()),3), "max": round(float(gc_b.max()),3)},
        "library":  {"mean": round(float(gc_l.mean()),3), "min": round(float(gc_l.min()),3), "max": round(float(gc_l.max()),3)},
        "overlap_lo": round(float(max(gc_b.min(), gc_l.min())),3),
        "overlap_hi": round(float(min(gc_b.max(), gc_l.max())),3),
    }

    # GC-STRATIFIED uplift: within the GC-overlap band, compare steered vs baseline
    lo_ov = max(gc_b.min(), gc_l.min()); hi_ov = min(gc_b.max(), gc_l.max())
    strat = {}
    mb = (gc_b >= lo_ov) & (gc_b <= hi_ov); ml = (gc_l >= lo_ov) & (gc_l <= hi_ov)
    result["gc_overlap_counts"] = {"baseline_in_overlap": int(mb.sum()), "library_in_overlap": int(ml.sum())}
    for name, k in preds.items():
        hb = col(base, k)[mb]; hl = col(lib, k)[ml]
        hb = hb[np.isfinite(hb)]; hl = hl[np.isfinite(hl)]
        if len(hb) >= 3 and len(hl) >= 3:
            d, lo, hi = boot_ci(hl, hb); pp = perm_p(hl, hb)
            strat[name] = {"n_base": len(hb), "n_lib": len(hl),
                           "mean_baseline": round(float(hb.mean()),2), "mean_library": round(float(hl.mean()),2),
                           "delta_at_matched_GC": round(d,2), "ci95": [round(lo,2), round(hi,2)],
                           "perm_p": round(pp,4), "survives_gc_control": bool(lo > 0)}
        else:
            strat[name] = {"n_base": len(hb), "n_lib": len(hl), "note": "insufficient overlap for a matched test"}
    result["gc_overlap_matched_uplift"] = strat

    # nearest-neighbour GC-matched pairing (baseline seq matched to nearest-GC library seq)
    # (only informative if overlap is non-trivial; report matched delta if >=8 pairs)
    nn = {}
    if mb.sum() >= 5 and ml.sum() >= 5:
        bi = np.where(mb)[0]; li = np.where(ml)[0]
        lib_gc_ov = gc_l[li]
        for name, k in preds.items():
            dpairs = []
            for j in bi:
                g = gc_b[j]; nnj = li[int(np.argmin(np.abs(lib_gc_ov - g)))]
                hb = base[j][k]; hl = lib[nnj][k]
                if np.isfinite(hb) and np.isfinite(hl): dpairs.append(hl - hb)
            if len(dpairs) >= 5:
                dp = np.array(dpairs)
                lo, hi = np.percentile([rng.choice(dp, len(dp), True).mean() for _ in range(2000)], [2.5, 97.5])
                nn[name] = {"n_pairs": len(dp), "mean_paired_delta": round(float(dp.mean()),2),
                            "ci95": [round(float(lo),2), round(float(hi),2)], "survives": bool(lo > 0)}
    result["gc_nearest_neighbour_matched"] = nn or {"note": "insufficient GC overlap for NN matching"}

    # ---------- (B) composition / length / low-complexity ----------
    def summ(rows, k):
        v = col(rows, k); v = v[np.isfinite(v)]
        return {"mean": round(float(v.mean()),3), "median": round(float(np.median(v)),3)}
    result["composition_controls"] = {
        "baseline": {"len": summ(base,"len"), "lowcplx_maxAAfrac": summ(base,"lowcplx"), "helixformer_frac": summ(base,"hf")},
        "library":  {"len": summ(lib,"len"),  "lowcplx_maxAAfrac": summ(lib,"lowcplx"),  "helixformer_frac": summ(lib,"hf")},
    }
    # AA composition shift (top movers)
    from collections import Counter
    def comp(rows):
        c = Counter(); tot = 0
        for r in rows:
            for a in r["prot"]:
                c[a]+=1; tot+=1
        return {a: c[a]/tot for a in AA} if tot else {}
    cb = comp(base); cl = comp(lib)
    shift = sorted(((a, round(cl.get(a,0)-cb.get(a,0),3)) for a in AA), key=lambda x: -abs(x[1]))
    result["aa_composition_shift_top"] = shift[:8]

    # ---------- (C) non-monotonicity, dose curve with bootstrap CIs ----------
    m2gen = json.load(open(os.path.join(RES, "M2_generations.json")))
    alphas = sorted({float(k.split("_")[0][1:]) for k in m2gen}, key=float)
    dose = {}
    for a in alphas:
        seqs = []
        for key, lst in m2gen.items():
            if key.startswith(f"a{a}_"): seqs += lst
        rows = describe(seqs)
        # score with GOR + CF (independent) and ESM2
        Hg = np.array([gor_pctH(gor, r["prot"]) for r in rows], float)
        Hc = np.array([cf_pctH(r["prot"]) for r in rows], float)
        He = np.array([probe.pct_helix(r["prot"]) if len(r["prot"])>=5 else np.nan for r in rows], float)
        gcv = col(rows, "gc")
        def m_ci(v):
            v = v[np.isfinite(v)]
            if len(v) < 3: return [float('nan'), None, None]
            lo, hi = np.percentile([rng.choice(v, len(v), True).mean() for _ in range(1000)], [2.5,97.5])
            return [round(float(v.mean()),2), round(float(lo),2), round(float(hi),2)]
        dose[str(a)] = {"n": len(rows), "H_esm2": m_ci(He), "H_gor": m_ci(Hg), "H_cf": m_ci(Hc),
                        "mean_gc": round(float(gcv[np.isfinite(gcv)].mean()),3)}
    result["dose_curve"] = dose
    # honest monotonicity: spearman over full range + high-dose subrange, per predictor
    def spear(klist):
        xs = []; ys = []
        for a in alphas:
            m = dose[str(a)][klist][0]
            if m == m: xs.append(a); ys.append(m)
        if len(xs) < 3: return None
        return round(float(spearmanr(xs, ys).statistic), 3)
    result["dose_monotonicity"] = {
        "spearman_full_range": {p: spear("H_"+p) for p in preds},
        "note": "curve is NON-MONOTONIC (alpha=4 dip); report honestly.",
    }
    hi_alphas = [a for a in alphas if a >= 8]
    def spear_sub(kk, sub):
        xs = [a for a in sub]; ys = [dose[str(a)][kk][0] for a in sub]
        ys2 = [(x,y) for x,y in zip(xs,ys) if y==y]
        if len(ys2) < 2: return None
        xs2, ys2v = zip(*ys2)
        return round(float(np.sign(np.polyfit(xs2, ys2v, 1)[0])),1)
    result["dose_monotonicity"]["high_dose_slope_sign_alpha>=8"] = {p: spear_sub("H_"+p, hi_alphas) for p in preds}

    # ---------- structural grounding attempt ----------
    struct_done = False
    if esmfold_available():
        try:
            from scorer import ESMFolder
            folder = ESMFolder()
            # fold top-15 library + 15 baseline, DSSP %H, correlate with ESM2 probe
            def fold_set(rows):
                fh = []; ph = []
                order = np.argsort([-(r["H_esm2"] if r["H_esm2"]==r["H_esm2"] else -1) for r in rows])[:15]
                for i in order:
                    p = rows[int(i)]["prot"]
                    if len(p) < 15: continue
                    h, e = folder.pct_helix_sheet(p)
                    if h == h: fh.append(h); ph.append(rows[int(i)]["H_esm2"])
                return fh, ph
            lf, lp = fold_set(lib); bf, bp = fold_set(base)
            if len(lf) >= 5:
                result["structural_validation"] = {
                    "grounded": True,
                    "library_mean_esmfold_dssp_pctH": round(float(np.mean(lf)),2),
                    "baseline_mean_esmfold_dssp_pctH": round(float(np.mean(bf)),2) if bf else None,
                    "r_esm2probe_vs_esmfold_on_steered": round(float(pearsonr(lp, lf)[0]),3),
                }
                result["grounding"] = "structural (ESMFold->DSSP on steered subset)"; struct_done = True
        except Exception as ex:
            result["structural_validation"] = {"grounded": False, "error": f"{type(ex).__name__}: {str(ex)[:150]}"}
    if not struct_done:
        result["structural_validation"] = result.get("structural_validation", {"grounded": False,
            "reason": "ESMFold (2.7 GB) blocked by HF-mirror CDN; no local folding predictor available"})
        result["grounding"] = ("sequence-predictor-supported (2 independent predictors GOR + Chou-Fasman) "
                               "+ GC/composition controls; NOT structurally confirmed")

    result["gpu_hours"] = round((time.time()-t0)/3600, 3)
    outp = os.path.join(OUT, "M5_structural_gc_control.json")
    json.dump(result, open(outp, "w"), indent=2)
    print("[M5] RESULT written to", outp, flush=True)
    print(json.dumps({"full_uplift": result["full_set_uplift_by_predictor"],
                      "gc_overlap_matched": result["gc_overlap_matched_uplift"],
                      "grounding": result["grounding"]}, indent=2), flush=True)
    print("M5_DONE", flush=True)

if __name__ == "__main__":
    main()
