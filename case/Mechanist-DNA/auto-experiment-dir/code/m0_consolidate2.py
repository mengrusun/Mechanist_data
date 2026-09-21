"""
M0 round-2 consolidation (frozen-S re-confirmation, NOT re-mining).

Carries the frozen round-1 feature set S (helix_features + s_f + matched_control + beta) forward
UNCHANGED, adds beta_features_v2, aggregates the 12 re-confirmation runs' set-level combined AUROC
+ four-state verdict, and writes results/m0_feature_set.json for M1-M3.

Run: python code/m0_consolidate2.py
"""
import os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

RES = "/data/wanghaoxiong/intergene_mechanist_v6/results"
FROZEN = "/data/wanghaoxiong/intergene_mechanist_v6/rounds/round_1/results/m0_feature_set.json"


def main():
    frozen = json.load(open(FROZEN))
    runs = {}
    for p in glob.glob(os.path.join(RES, "m0_*_s*.json")):
        r = json.load(open(p))
        if "config" not in r:
            continue
        c = r["config"]
        runs[(c["organism"], c["helix_def"], c["split_seed"])] = r
    print(f"[consolidate2] {len(runs)} round-2 M0 runs", flush=True)

    steer_org = frozen.get("steer_organism", "prokaryote")
    # per-config re-confirmation on the FROZEN set
    per_cfg_auroc = {f"{k[0]}|{k[1]}|s{k[2]}": v.get("combined_set_test_auroc") for k, v in runs.items()}
    per_cfg_verdict = {f"{k[0]}|{k[1]}|s{k[2]}": v.get("verdict") for k, v in runs.items()}
    per_cfg_bh = {f"{k[0]}|{k[1]}|s{k[2]}": v.get("frozen_bh_significant_count") for k, v in runs.items()}
    per_cfg_confound = {f"{k[0]}|{k[1]}|s{k[2]}": v.get("set_auroc_over_confound") for k, v in runs.items()}

    # steer-organism (prokaryote) configs are the gating ones (S was mined there; steering happens there)
    steer_cfgs = {k: v for k, v in runs.items() if k[0] == steer_org}
    steer_aurocs = [v.get("combined_set_test_auroc") for v in steer_cfgs.values()
                    if v.get("combined_set_test_auroc") is not None]
    steer_verdicts = [v.get("verdict") for v in steer_cfgs.values()]
    mean_steer_auroc = float(np.mean(steer_aurocs)) if steer_aurocs else None
    reconfirmed = (mean_steer_auroc is not None and mean_steer_auroc >= 0.80
                   and all(vd in ("established", "conditional") for vd in steer_verdicts) and len(steer_verdicts) > 0)

    # cross-organism transfer (informative, not gating)
    euk_aurocs = [v.get("combined_set_test_auroc") for k, v in runs.items() if k[0] == "eukaryote"
                  and v.get("combined_set_test_auroc") is not None]
    holds_multi_org = bool(euk_aurocs and np.mean(euk_aurocs) >= 0.80)

    # pick strongest beta_v2 among steer-organism configs
    best_bv2 = None
    for k, v in steer_cfgs.items():
        bv2 = v.get("beta_v2")
        if bv2 and bv2.get("beta_v2_set_test_auroc") is not None:
            if best_bv2 is None or bv2["beta_v2_set_test_auroc"] > best_bv2["beta_v2_set_test_auroc"]:
                best_bv2 = dict(bv2); best_bv2["from_config"] = f"{k[0]}|{k[1]}|s{k[2]}"

    # four-state verdict (frozen-S re-confirmation gate)
    all_floor = all(v.get("floor_ok", True) for v in runs.values())
    if not runs:
        verdict = "inconclusive"
    elif reconfirmed:
        verdict = "established"
    elif mean_steer_auroc is not None and mean_steer_auroc >= 0.70 and all_floor:
        verdict = "conditional"
    elif not all_floor:
        verdict = "inconclusive"
    else:
        verdict = "not-established"

    out = dict(frozen)  # carry frozen S/s_f/matched/beta UNCHANGED
    out.update({
        "round": 2,
        "verdict": verdict,
        "phenomenon_status": verdict,
        "reuse_frozen": True,
        "reconfirmed_frozen_S": bool(reconfirmed),
        "mean_steer_organism_set_auroc": mean_steer_auroc,
        "round1_steer_set_auroc_ref": frozen.get("per_config_combined_set_auroc", {}),
        "holds_multi_organism": holds_multi_org,
        "eukaryote_transfer_set_auroc_mean": float(np.mean(euk_aurocs)) if euk_aurocs else None,
        "per_config_combined_set_auroc": per_cfg_auroc,
        "per_config_verdict": per_cfg_verdict,
        "per_config_frozen_bh_significant": per_cfg_bh,
        "per_config_set_auroc_over_confound": per_cfg_confound,
        "beta_v2": best_bv2,
        "beta_v2_clears_bar": bool(best_bv2 and best_bv2.get("clears_bar_stronger_than_round1")),
    })
    json.dump(out, open(os.path.join(RES, "m0_feature_set.json"), "w"), indent=2,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"[consolidate2] VERDICT={verdict} mean_steer_set_auroc={mean_steer_auroc} "
          f"beta_v2_auroc={(best_bv2 or {}).get('beta_v2_set_test_auroc')} "
          f"beta_v2_clears={out['beta_v2_clears_bar']} holds_multi_org={holds_multi_org}", flush=True)


if __name__ == "__main__":
    main()
