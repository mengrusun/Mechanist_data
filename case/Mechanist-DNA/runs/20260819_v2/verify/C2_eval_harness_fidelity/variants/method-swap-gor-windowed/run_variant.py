"""
C2 VERIFY variant -- METHOD SWAP: GOR-style windowed statistical SS predictor.

Replaces the main metric's ESM2-650M frozen-embedding linear probe with a fully
INDEPENDENT classical predictor that shares no machinery with ESM2:
  logistic regression on a +/-8-residue one-hot amino-acid window (GOR family),
  fit on the SAME 600 training proteins' experimental-DSSP labels, evaluated on the
  SAME 200 held-out proteins as E1. Everything else (data, DSSP labels, protein-level
  %H definition, Pearson protocol, frame-recovery protocol) is held FIXED from run_E1.py.

Frozen claim C2: a fast sequence-based SS->%H metric agrees with experimental-DSSP %H
at Pearson r >= 0.7, with correct reading-frame recovery.
Success (this variant): r >= 0.7 AND positive correlation AND frame recovery high.
"""
import os, sys, json, time
EXP = "/data/wanghaoxiong/Mechanist-DNA-experiment/simple_20260819_v4/experiments"
DATA = "/data/wanghaoxiong/Mechanist-DNA-experiment/simple_20260819_v4/data"
sys.path.insert(0, EXP)
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
import evo2lib as E
from scorer import ss3_labels

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
W = 8  # window radius -> 2W+1 = 17 positions
NF = len(AA) + 1  # 20 aa + 1 pad/unknown

def win_features(seq):
    """Per-residue one-hot window features [L, (2W+1)*NF]."""
    L = len(seq); idx = []
    for c in seq:
        idx.append(AA_IDX.get(c, len(AA)))  # unknown -> pad channel
    feats = np.zeros((L, (2 * W + 1) * NF), dtype=np.float32)
    for i in range(L):
        for j, off in enumerate(range(-W, W + 1)):
            k = i + off
            ch = idx[k] if 0 <= k < L else len(AA)  # pad channel out of bounds
            feats[i, j * NF + ch] = 1.0
    return feats

def main():
    t0 = time.time()
    prots = json.load(open(os.path.join(DATA, "proteins.json")))
    train = [p for p in prots if p["split"] == "train"][:600]   # SAME as E1
    held = [p for p in prots if p["split"] == "heldout"][:200]   # SAME eval set as E1
    print(f"[gor] train={len(train)} heldout_eval={len(held)}", flush=True)

    # fit GOR-style windowed logistic on train residues
    X = []; Y = []
    for p in train:
        n = min(len(p["aa"]), len(p["ss8"]))
        if n < 5: continue
        X.append(win_features(p["aa"][:n]))
        Y.extend(ss3_labels(p["ss8"][:n]))
    X = np.concatenate(X, 0); Y = np.array(Y)
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
    clf.fit(X, Y)
    train_q3 = clf.score(X, Y)
    print(f"[gor] train residue-acc(Q3)={train_q3:.3f}  n_res={len(Y)}", flush=True)

    # eval on the SAME 200 held-out proteins.
    # %H definition matched EXACTLY to run_E1.py: predict over the FULL aa sequence,
    # denominator = len(predicted); Q3 over n=min(len(pred),len(true)); reference = p['pctH'].
    fastH = []; expH = []; correct = 0; total = 0
    for p in held:
        pred = clf.predict(win_features(p["aa"]))          # full aa, same as probe.predict_ss3(p["aa"])
        true = ss3_labels(p["ss8"])
        n = min(len(pred), len(true))
        correct += int(sum(pred[i] == true[i] for i in range(n))); total += n
        fastH.append(100.0 * sum(c == 'H' for c in pred) / len(pred))   # denominator = full length (== E1)
        expH.append(p["pctH"])  # experimental DSSP %H (unchanged reference)
    q3 = correct / total
    r, pval = pearsonr(fastH, expH)
    print(f"[gor] heldout Q3={q3:.3f}  r(gor%H, expDSSP%H)={r:.4f} p={pval:.2e}", flush=True)

    # frame recovery -- identical protocol to E1 (method-independent ORF finder)
    rec = 0; nrec = 0
    for p in held[:100]:
        aa = p["aa"][:150]
        dna = E.reverse_translate(aa)
        intended = E.translate(dna)
        prot, frame, start, orf = E.find_longest_orf(dna, min_aa=20)
        nrec += 1
        if prot == intended and frame == 0 and start == 0:
            rec += 1
    frame_recovery = rec / max(1, nrec)

    result = {
        "variant_tag": "method-swap-gor-windowed", "dimension": "method",
        "claim_id": "C2", "swap": "ESM2-650M+linear-probe -> GOR-style windowed logistic",
        "predictor": f"logistic on +/-{W} aa one-hot window, fit on 600 train DSSP labels",
        "n_train_proteins": len(train), "n_heldout_proteins_eval": len(held),
        "train_Q3": round(float(train_q3), 4), "heldout_Q3": round(float(q3), 4),
        "pearson_r_gorH_vs_expDSSP_H": round(float(r), 4), "pearson_p": float(pval),
        "frame_recovery_rate": round(frame_recovery, 4),
        "mean_gorH": round(float(np.mean(fastH)), 2), "mean_expH": round(float(np.mean(expH)), 2),
        "main_experiment_r": 0.9874,
        "pass_criterion": "Pearson r >= 0.7 AND positive AND frame recovery >= 0.9",
        "pass": bool(r >= 0.7 and frame_recovery >= 0.9),
        "wall_hours": round((time.time() - t0) / 3600, 4),
    }
    outp = os.path.join(os.path.dirname(__file__), "result.json")
    json.dump(result, open(outp, "w"), indent=2)
    print("[gor] RESULT", json.dumps(result), flush=True)
    print("VARIANT_DONE", flush=True)

if __name__ == "__main__":
    main()
