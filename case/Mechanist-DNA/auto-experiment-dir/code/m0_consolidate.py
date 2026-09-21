"""
Consolidate the 12 M0 grid runs into results/m0_feature_set.json: freeze feature set S, compute
robustness (Jaccard overlap across seeds; holds across organism/helix_def), pick beta-sheet
off-target features and activation-matched control features for M3, per-feature steering scales
s_f, and the overall 4-state phenomenon verdict.
Run: python code/m0_consolidate.py
"""
import os, sys, json, glob, itertools
import numpy as np
from scipy import sparse
sys.path.insert(0, "code")
import m0_data as D
from m0_scoring import sparse_auroc_all

RES = "/data/wanghaoxiong/intergene_mechanist_v6/results"
ALPHA_FEAT, BETA_FEAT = 28741, 22326


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(len(a | b), 1)


def main():
    runs = {}
    for p in glob.glob(os.path.join(RES, "m0_*_s*.json")):
        r = json.load(open(p))
        if "config" not in r:
            continue
        c = r["config"]
        runs[(c["organism"], c["helix_def"], c["split_seed"])] = r
    print(f"[consolidate] {len(runs)} M0 runs found", flush=True)

    # robustness: seed Jaccard within each (org, helix_def)
    seed_jaccard = {}
    for org in D.ORGANISMS:
        for hd in ["HGI", "H_only"]:
            Ss = [runs[(org, hd, s)]["S_features"] for s in [42, 200, 201] if (org, hd, s) in runs]
            if len(Ss) >= 2:
                js = [jaccard(a, b) for a, b in itertools.combinations(Ss, 2)]
                seed_jaccard[f"{org}|{hd}"] = float(np.mean(js))

    # per-config verdict summary
    verdicts = {k: v["verdict"] for k, v in runs.items()}
    established_cfgs = [k for k, vd in verdicts.items() if vd in ("established", "conditional")]

    # choose primary steering organism/helix_def: prefer an 'established' prokaryote HGI, else any established
    def score_cfg(k):
        org, hd, seed = k
        r = runs[k]
        return (r["verdict"] == "established", org == "prokaryote", hd == "HGI",
                r.get("best_test_auroc", 0))
    primary = max(runs.keys(), key=score_cfg)
    org_p, hd_p, seed_p = primary
    print(f"[consolidate] primary config: {primary} verdict={runs[primary]['verdict']}", flush=True)

    # robust S for the primary org/helix: strict features appearing in >=2 of the 3 seeds; if the
    # strict single-feature bar produced an empty/thin S, fall back to the relaxed selective set
    # (BH-significant + margin-positive helix features) which is the distributed helix code C1 refers to.
    from collections import Counter
    def field_across_seeds(field):
        cnt = Counter()
        for s in [42, 200, 201]:
            if (org_p, hd_p, s) in runs:
                cnt.update(runs[(org_p, hd_p, s)].get(field, []))
        return cnt
    cnt_strict = field_across_seeds("S_features")
    robust_S = sorted([f for f, c in cnt_strict.items() if c >= 2], key=lambda f: -cnt_strict[f])
    used_relaxed = False
    if len(robust_S) < 3:
        cnt_relaxed = field_across_seeds("relaxed_S_features")
        relaxed_robust = sorted([f for f, c in cnt_relaxed.items() if c >= 2], key=lambda f: -cnt_relaxed[f])
        if not relaxed_robust:
            relaxed_robust = runs[primary].get("relaxed_S_features", [])[:50]
        # union strict into relaxed, keep relaxed ordering
        robust_S = list(dict.fromkeys(list(robust_S) + list(relaxed_robust)))[:80]
        used_relaxed = True
    if not robust_S:
        robust_S = runs[primary].get("combined_set_features", runs[primary]["S_features"])
    print(f"[consolidate] robust |S| = {len(robust_S)} (used_relaxed={used_relaxed})", flush=True)

    # holds across organism groups / helix defs?
    orgs_established = set(k[0] for k in established_cfgs)
    helix_defs_established = set(k[1] for k in established_cfgs)
    holds_multi_org = len(orgs_established) >= 2
    holds_multi_helixdef = len(helix_defs_established) >= 2

    # ---- load cached acts of the primary steer organism for beta / matched / s_f ----
    X = sparse.load_npz(os.path.join(D.DATA_DIR, f"m0_acts_{org_p}.npz")).tocsc()
    L = np.load(os.path.join(D.DATA_DIR, f"m0_labels_{org_p}.npz"))
    n_codons, n_feat = X.shape
    sheet_y = L["sheet"].astype(bool)
    auroc_sheet = sparse_auroc_all(X, sheet_y)
    helix_y = (L["helix_hgi"] if hd_p == "HGI" else L["helix_h"]).astype(bool)
    auroc_helix = sparse_auroc_all(X, helix_y)

    # beta features: top sheet-AUROC features that are NOT in S and are more sheet- than helix-selective
    beta_rank = np.argsort(-auroc_sheet)
    beta_features = [int(f) for f in beta_rank
                     if f not in set(robust_S) and auroc_sheet[f] - auroc_helix[f] > 0.1][:5]

    # activation frequency & mean magnitude per feature
    freq = np.asarray((X > 0).sum(0)).ravel()
    mag = np.zeros(n_feat)
    Xc = X.tocsc()
    for f in range(n_feat):
        s, e = Xc.indptr[f], Xc.indptr[f + 1]
        if e > s:
            mag[f] = Xc.data[s:e].mean()

    def s_f_of(feats):
        return [float(mag[f]) if mag[f] > 0 else 1.0 for f in feats]

    # matched-control features: for each f in S, pick a random feature with similar freq & mag,
    # not in S and not helix-selective (auroc_helix < 0.6) and not beta.
    rng = np.random.default_rng(0)
    exclude = set(robust_S) | set(beta_features)
    matched = []
    order = np.arange(n_feat)
    for f in robust_S:
        cands = order[(np.abs(np.log1p(freq) - np.log1p(freq[f])) < 0.3) &
                      (np.abs(mag - mag[f]) < 0.5 * (mag[f] + 1e-6)) &
                      (auroc_helix < 0.6)]
        cands = [c for c in cands if c not in exclude and c not in matched]
        if cands:
            matched.append(int(rng.choice(cands)))
    if len(matched) < max(1, len(robust_S) // 2):  # relax if too few
        cands = [c for c in order if c not in exclude and auroc_helix[c] < 0.6 and freq[c] > 0]
        matched = list(rng.choice(cands, size=min(len(robust_S), len(cands)), replace=False).astype(int))
    print(f"[consolidate] beta_features={beta_features[:5]} matched_control n={len(matched)}", flush=True)

    # overall verdict
    any_established = any(runs[k]["verdict"] == "established" for k in runs)
    all_floor_ok = all(runs[k]["floor_ok"] for k in runs)
    if not all_floor_ok and not any_established:
        verdict = "inconclusive"
    elif any_established and (holds_multi_org or holds_multi_helixdef) and len(robust_S) >= 1:
        verdict = "established"
    elif any_established and len(robust_S) >= 1:
        verdict = "conditional"
    elif any(runs[k]["verdict"] in ("established", "conditional") for k in runs):
        verdict = "conditional"
    else:
        verdict = "not-established"

    cond_boundary = None
    if verdict == "conditional":
        cond_boundary = {"organisms_holding": sorted(orgs_established),
                         "helix_defs_holding": sorted(helix_defs_established),
                         "note": f"phenomenon established under {sorted(established_cfgs)[:5]}"}

    out = {
        "verdict": verdict,
        "phenomenon_status": verdict,
        "primary_config": {"organism": org_p, "helix_def": hd_p, "seed": seed_p},
        "steer_organism": org_p,
        "organism_primary": org_p,
        "helix_features": [int(f) for f in robust_S],
        "s_f": s_f_of(robust_S),
        "beta_features": beta_features,
        "beta_s_f": s_f_of(beta_features),
        "matched_control_features": matched,
        "matched_control_s_f": s_f_of(matched),
        "S_size": len(robust_S),
        "seed_jaccard": seed_jaccard,
        "holds_multi_organism": holds_multi_org, "holds_multi_helixdef": holds_multi_helixdef,
        "conditional_boundary": cond_boundary,
        "per_config_verdict": {f"{k[0]}|{k[1]}|s{k[2]}": v for k, v in verdicts.items()},
        "per_config_best_test_auroc": {f"{k[0]}|{k[1]}|s{k[2]}": runs[k].get("best_test_auroc")
                                       for k in runs},
        "per_config_combined_set_auroc": {f"{k[0]}|{k[1]}|s{k[2]}": runs[k].get("combined_set_test_auroc")
                                          for k in runs},
        "per_config_relaxed_S_size": {f"{k[0]}|{k[1]}|s{k[2]}": runs[k].get("relaxed_S_size")
                                      for k in runs},
        "used_relaxed_set_for_steering": used_relaxed,
        "known_feature_anchor": {
            "f28741_in_S": ALPHA_FEAT in set(robust_S),
            "f28741_helix_auroc": float(auroc_helix[ALPHA_FEAT]),
            "f28741_helix_rank": int((auroc_helix > auroc_helix[ALPHA_FEAT]).sum()),
            "f22326_sheet_auroc": float(auroc_sheet[BETA_FEAT]),
            "f22326_sheet_rank": int((auroc_sheet > auroc_sheet[BETA_FEAT]).sum()),
        },
    }
    json.dump(out, open(os.path.join(RES, "m0_feature_set.json"), "w"), indent=2, default=lambda o: o.item() if hasattr(o,"item") else (o.tolist() if hasattr(o,"tolist") else str(o)))
    print(f"[consolidate] VERDICT={verdict} |S|={len(robust_S)} beta={len(beta_features)} "
          f"matched={len(matched)}", flush=True)
    print(f"[consolidate] known anchor: f28741 helix_auroc={out['known_feature_anchor']['f28741_helix_auroc']:.3f} "
          f"rank={out['known_feature_anchor']['f28741_helix_rank']}; "
          f"f22326 sheet_auroc={out['known_feature_anchor']['f22326_sheet_auroc']:.3f} "
          f"rank={out['known_feature_anchor']['f22326_sheet_rank']}", flush=True)


if __name__ == "__main__":
    main()
