"""
C2 VERIFY variant -- DATASET SWAP: same ESM2-650M+linear-probe metric, FRESH disjoint proteins.

Holds the METHOD fixed (loads the exact trained probe assets/ss_probe.pkl used by E1) and
swaps only the evaluation dataset: a fresh DEDUPLICATED held-out set of unseen RCSB proteins
(proteins.json heldout split, indices >=200 -- never scored in E1's held[:200]), near-duplicate
chains dropped by the aa[:60] key (same redundancy key used at data-build). Correlate the
predicted %H with experimental-DSSP %H. Everything else fixed from run_E1.py.

Frozen claim C2: a fast sequence-based SS->%H metric agrees with experimental-DSSP %H at
Pearson r >= 0.7. Success (this variant): r >= 0.7 AND positive correlation on the fresh set.
"""
import os, sys, json, time
EXP = "/data/wanghaoxiong/Mechanist-DNA-experiment/simple_20260819_v4/experiments"
DATA = "/data/wanghaoxiong/Mechanist-DNA-experiment/simple_20260819_v4/data"
ASSETS = "/data/wanghaoxiong/Mechanist-DNA-experiment/simple_20260819_v4/assets"
sys.path.insert(0, EXP)
import numpy as np
from scipy.stats import pearsonr
from scorer import ESM2SSProbe, ss3_labels

def main():
    t0 = time.time()
    prots = json.load(open(os.path.join(DATA, "proteins.json")))
    held = [p for p in prots if p["split"] == "heldout"]
    e1_used = held[:200]                     # E1's original eval proteins
    fresh_raw = held[200:]                    # unseen in E1
    # redundancy control: drop near-duplicate chains (also vs the E1-used set)
    seen = set(p["aa"][:60] for p in e1_used)
    fresh = []
    for p in fresh_raw:
        k = p["aa"][:60]
        if k in seen:
            continue
        seen.add(k); fresh.append(p)
    print(f"[dset] E1_used={len(e1_used)} fresh_raw={len(fresh_raw)} fresh_dedup={len(fresh)}", flush=True)

    probe = ESM2SSProbe()
    probe.load_probe(os.path.join(ASSETS, "ss_probe.pkl"))   # EXACT trained probe from E1

    fastH = []; expH = []; correct = 0; total = 0
    for i, p in enumerate(fresh):
        ss3_pred = probe.predict_ss3(p["aa"])
        ss3_true = ss3_labels(p["ss8"])
        n = min(len(ss3_pred), len(ss3_true))
        correct += int(sum(ss3_pred[j] == ss3_true[j] for j in range(n))); total += n
        fastH.append(100.0 * sum(c == 'H' for c in ss3_pred) / len(ss3_pred))
        expH.append(p["pctH"])
        if (i + 1) % 50 == 0:
            print(f"[dset] scored {i+1}/{len(fresh)}", flush=True)
    # guard rails: nonempty, finite, non-constant before pearsonr
    assert len(fresh) >= 20, f"too few fresh proteins ({len(fresh)}) for a stable correlation"
    assert total > 0 and all(np.isfinite(fastH)) and all(np.isfinite(expH)), "non-finite %H values"
    assert np.std(fastH) > 1e-6 and np.std(expH) > 1e-6, "constant %H vector -> pearsonr undefined"
    q3 = correct / total
    r, pval = pearsonr(fastH, expH)
    print(f"[dset] fresh Q3={q3:.3f}  r(fast%H, expDSSP%H)={r:.4f} p={pval:.2e}", flush=True)

    # frame-recovery test -- identical protocol to run_E1.py, on the fresh set (part of frozen claim C2)
    import evo2lib as E
    rec = 0; nrec = 0
    for p in fresh[:100]:
        aa = p["aa"][:150]
        dna = E.reverse_translate(aa)
        intended = E.translate(dna)
        prot, frame, start, orf = E.find_longest_orf(dna, min_aa=20)
        nrec += 1
        if prot == intended and frame == 0 and start == 0:
            rec += 1
    frame_recovery = rec / max(1, nrec)
    print(f"[dset] frame recovery = {frame_recovery:.3f} ({rec}/{nrec})", flush=True)

    result = {
        "variant_tag": "dataset-swap-heldout-proteins", "dimension": "dataset",
        "claim_id": "C2", "swap": "E1 held[:200] -> fresh deduplicated held[200:] unseen proteins",
        "method": "UNCHANGED: ESM2-650M frozen emb + linear 3-state probe (assets/ss_probe.pkl)",
        "n_fresh_proteins_eval": len(fresh), "n_dropped_as_duplicate": len(fresh_raw) - len(fresh),
        "heldout_Q3": round(float(q3), 4),
        "pearson_r_fastH_vs_expDSSP_H": round(float(r), 4), "pearson_p": float(pval),
        "frame_recovery_rate": round(frame_recovery, 4),
        "mean_fastH": round(float(np.mean(fastH)), 2), "mean_expH": round(float(np.mean(expH)), 2),
        "main_experiment_r": 0.9874,
        "pass_criterion": "Pearson r >= 0.7 AND positive AND frame recovery >= 0.9",
        "pass": bool(r >= 0.7 and frame_recovery >= 0.9),
        "wall_hours": round((time.time() - t0) / 3600, 4),
    }
    outp = os.path.join(os.path.dirname(__file__), "result.json")
    json.dump(result, open(outp, "w"), indent=2)
    print("[dset] RESULT", json.dumps(result), flush=True)
    print("VARIANT_DONE", flush=True)

if __name__ == "__main__":
    main()
