"""Phase 8 (judge) + Phase 10 (aggregate) helper for the C1 model-swap variant.

Computes, on the 262k variant's own result.json (mirroring the main experiment's C1 tests):
  - dose ladder helix_all per coef (mean over seeds), validity, prot_len, gc;
  - dose-response Spearman rho across all (coef, seed) points, one-sided positive permutation p;
  - winning (coef 1.0) vs in-model baseline (coef 0.0) helix delta + Cliff's delta + Mann-Whitney;
then emits a claim_supported / consistent_with_main_experiment judgment for verdict.json.
"""
import os, sys, json
import numpy as np
from scipy import stats

VDIR = "/data/wanghaoxiong/Mechanist-DNA-experiment/20260823_v1/verify/C1_helix_gain_doseresponse/variants/model-swap-evo2-7b-262k"
RES = os.path.join(VDIR, "result.json")


def cliffs_delta(a, b):
    a = np.asarray(a); b = np.asarray(b)
    gt = sum((a[:, None] > b[None, :]).sum(axis=1))
    lt = sum((a[:, None] < b[None, :]).sum(axis=1))
    return (gt - lt) / (len(a) * len(b))


def perm_trend_p(coefs, vals, n_perm=20000, seed=0):
    rho, _ = stats.spearmanr(coefs, vals)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        r, _ = stats.spearmanr(coefs, rng.permutation(vals))
        if r >= rho:
            cnt += 1
    return rho, (cnt + 1) / (n_perm + 1)


def main():
    res = json.load(open(RES))
    rows = res["rows"]
    coefs_sorted = sorted(set(r["coef"] for r in rows))
    ladder = {}
    for c in coefs_sorted:
        rr = [r for r in rows if r["coef"] == c]
        ladder[c] = dict(
            helix_all=float(np.mean([r["helix_all"] for r in rr])),
            validity=float(np.mean([r["validity_rate"] for r in rr])),
            prot_len=float(np.mean([r["prot_len"] for r in rr])),
            gc=float(np.mean([r["gc"] for r in rr])),
            nll=float(np.mean([r["mean_nll"] for r in rr])))

    # dose-response trend over all (coef, seed) points
    pc = [r["coef"] for r in rows]
    pv = [r["helix_all"] for r in rows]
    rho, p_perm = perm_trend_p(np.array(pc), np.array(pv))

    # winning (1.0) vs in-model baseline (0.0), per-generation
    base = np.concatenate([np.array(r["per_gen_helix"]) for r in rows if r["coef"] == 0.0])
    win = np.concatenate([np.array(r["per_gen_helix"]) for r in rows if r["coef"] == 1.0])
    delta = float(win.mean() - base.mean())
    cd = float(cliffs_delta(win, base))
    mw = stats.mannwhitneyu(win, base, alternative="greater")

    # peak gain (any coef vs baseline)
    peak_coef = max(coefs_sorted, key=lambda c: ladder[c]["helix_all"])
    peak_delta = ladder[peak_coef]["helix_all"] - ladder[0.0]["helix_all"]

    # ---- judgment ----
    monotone_positive = rho > 0 and p_perm < 0.05
    win_gain = delta > 0 and mw.pvalue < 0.05
    claim_supported = "pass" if (monotone_positive and win_gain) else "fail"
    # main experiment verdict on C1 = supported -> consistent = claim_supported (no flip)
    consistent = claim_supported

    out = dict(
        variant_tag="model-swap-evo2-7b-262k", dimension="model", claim_id="C1",
        model="evo2_7b_262k", main_experiment_model="evo2_7b",
        dose_ladder=ladder,
        dose_response=dict(spearman_rho=float(rho), perm_p_onesided=float(p_perm),
                           n_points=len(rows)),
        winning_vs_baseline=dict(coef=1.0, helix_win=float(win.mean()),
                                 helix_baseline=float(base.mean()), delta=delta,
                                 cliffs_delta=cd, mannwhitney_p_greater=float(mw.pvalue),
                                 n_win=int(len(win)), n_base=int(len(base))),
        peak=dict(coef=peak_coef, delta_vs_baseline=float(peak_delta)),
        judgment=dict(monotone_positive_dose_response=bool(monotone_positive),
                      helix_gain_at_winning=bool(win_gain),
                      claim_supported=claim_supported,
                      consistent_with_main_experiment=consistent))
    json.dump(out, open(os.path.join(VDIR, "analysis.json"), "w"), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
