"""
E1: build + validate the alpha-helical-content evaluation harness (Claim 2).

- Trains the fast ESM2-650M SS-probe on train-split experimental DSSP labels.
- Validates on the disjoint held-out split:
    * Pearson r between fast-predicted %H and experimental-structure DSSP %H  (pass: r >= 0.7)
    * per-residue Q3 accuracy
    * ORF/reading-frame recovery rate on reverse-translated CDS
    * (optional) ESMFold->DSSP %H agreement, if ESMFold weights are available
Outputs: results/E1_eval_harness_validation.json ; assets/ss_probe.pkl
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from scipy.stats import pearsonr
import evo2lib as E
from scorer import ESM2SSProbe, ss3_labels

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
ASSETS = os.path.abspath(os.path.join(HERE, "..", "assets"))
os.makedirs(RES, exist_ok=True)

def main():
    t0 = time.time()
    prots = json.load(open(os.path.join(DATA, "proteins.json")))
    train = [p for p in prots if p["split"] == "train"]
    held = [p for p in prots if p["split"] == "heldout"]
    print(f"[E1] train={len(train)} heldout={len(held)}", flush=True)

    probe = ESM2SSProbe()
    # cap training proteins for speed but keep it large
    tr = train[:600]
    acc_train = probe.fit([p["aa"] for p in tr], [p["ss8"] for p in tr])
    print(f"[E1] probe train residue-acc(Q3)={acc_train:.3f}", flush=True)
    probe.save(os.path.join(ASSETS, "ss_probe.pkl"))

    # held-out per-residue Q3 + %H correlation
    fastH = []; expH = []; correct = 0; total = 0
    hv = held[:200]
    for p in hv:
        ss3_pred = probe.predict_ss3(p["aa"])
        ss3_true = ss3_labels(p["ss8"])
        n = min(len(ss3_pred), len(ss3_true))
        correct += sum(ss3_pred[i] == ss3_true[i] for i in range(n)); total += n
        fastH.append(100.0 * sum(c == 'H' for c in ss3_pred) / len(ss3_pred))
        expH.append(p["pctH"])
    q3 = correct / total
    r_fast_exp, p_fast_exp = pearsonr(fastH, expH)
    print(f"[E1] heldout Q3={q3:.3f}  r(fast%H, expDSSP%H)={r_fast_exp:.3f}", flush=True)

    # frame recovery: reverse-translate (frame-0 ATG-start), then ORF-find; the recovered
    # protein must equal the intended in-frame (frame-0) translation of the CDS.
    rec = 0; nrec = 0
    for p in hv[:100]:
        aa = p["aa"][:150]
        dna = E.reverse_translate(aa)
        intended = E.translate(dna)             # frame-0 translation
        prot, frame, start, orf = E.find_longest_orf(dna, min_aa=20)
        nrec += 1
        if prot == intended and frame == 0 and start == 0:
            rec += 1
    frame_recovery = rec / max(1, nrec)
    print(f"[E1] frame recovery = {frame_recovery:.3f} ({rec}/{nrec})", flush=True)

    result = {
        "milestone": "E1", "claim": "C2",
        "n_train_proteins": len(tr), "n_heldout_proteins_eval": len(hv),
        "probe_train_Q3": round(acc_train, 4),
        "heldout_Q3": round(q3, 4),
        "pearson_r_fastH_vs_expDSSP_H": round(float(r_fast_exp), 4),
        "pearson_p": float(p_fast_exp),
        "frame_recovery_rate": round(frame_recovery, 4),
        "pass_criterion": "Pearson r >= 0.7 AND frame recovery high",
        "pass": bool(r_fast_exp >= 0.7 and frame_recovery >= 0.9),
        "mean_fastH": round(float(np.mean(fastH)), 2),
        "mean_expH": round(float(np.mean(expH)), 2),
    }

    # optional ESMFold agreement
    if os.path.exists(os.path.join(ASSETS, ".esmfold_dl_done")):
        try:
            from scorer import ESMFolder
            folder = ESMFolder()
            efH = []; ex2 = []; fa2 = []
            for p in hv[:25]:
                h, e = folder.pct_helix_sheet(p["aa"][:350])
                if h == h:
                    efH.append(h); ex2.append(p["pctH"])
                    fa2.append(100.0*sum(c=='H' for c in probe.predict_ss3(p["aa"]))/len(p["aa"]))
            if len(efH) >= 5:
                r_ef_exp = pearsonr(efH, ex2)[0]
                r_fast_ef = pearsonr(fa2, efH)[0]
                result["esmfold_subset_n"] = len(efH)
                result["pearson_r_ESMFoldDSSP_vs_expDSSP"] = round(float(r_ef_exp), 4)
                result["pearson_r_fastH_vs_ESMFoldDSSP"] = round(float(r_fast_ef), 4)
                print(f"[E1] ESMFold subset r(ESMFold,exp)={r_ef_exp:.3f} r(fast,ESMFold)={r_fast_ef:.3f}", flush=True)
        except Exception as ex:
            result["esmfold_note"] = f"ESMFold scoring skipped: {type(ex).__name__}: {str(ex)[:120]}"
    else:
        result["esmfold_note"] = "ESMFold weights not available at run time; structure reference = experimental PDB DSSP (gold standard)."

    result["gpu_hours"] = round((time.time() - t0) / 3600, 3)
    json.dump(result, open(os.path.join(RES, "E1_eval_harness_validation.json"), "w"), indent=2)
    print("[E1] RESULT", json.dumps(result), flush=True)
    print("E1_DONE", flush=True)

if __name__ == "__main__":
    main()
