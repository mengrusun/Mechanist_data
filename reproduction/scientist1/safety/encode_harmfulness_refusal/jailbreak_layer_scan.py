"""Per-layer projection of jailbreak samples onto v_h (harmfulness) and v_p (refusal).

Focus on the *prefill* attack (which has abundant successful jailbreaks) to test whether the
harmfulness signal is preserved (proj_h remains high, similar to refused harmful baseline) while
the refusal signal is suppressed (proj_r drops toward the benign level).
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RES_DIR = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/results"
JB_DIR  = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/jailbreak_llama3"
ACT_DIR = "/data/zhenqian/Reproduction1/cc/safety/encode_harmfulness_refusal/acts_llama3"


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def project_layers(hs, v):
    """hs: (N, L+1, D); v: (L+1, D). Returns (N, L+1) projections onto unit-norm v[l]."""
    hs = hs.astype(np.float32)
    v_n = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)
    return np.einsum("nld,ld->nl", hs, v_n)


def load_acts(name, kind):
    """kind in {'hs_inst', 'hs_post'}"""
    z = np.load(os.path.join(ACT_DIR, f"{name}.npz"), allow_pickle=True)
    return z[kind].astype(np.float32)


def load_jb(name, kind):
    z = np.load(os.path.join(JB_DIR, f"{name}.npz"), allow_pickle=True)
    return z[kind].astype(np.float32), z["refusal"].astype(bool)


def main():
    d = np.load(os.path.join(RES_DIR, "directions_llama3.npz"))
    v_inst = d["v_inst"]  # (L+1, D)  computed at t_inst
    v_post = d["v_post"]  # (L+1, D)  computed at t_post

    # Baseline projections
    print("Baselines (mean projection across layers 5..25):\n")
    harm_i  = load_acts("advbench_test", "hs_inst")
    harm_p  = load_acts("advbench_test", "hs_post")
    ben_i   = load_acts("alpaca_test",   "hs_inst")
    ben_p   = load_acts("alpaca_test",   "hs_post")

    proj_h_harm  = project_layers(harm_i, v_inst)   # projection onto v_h at each layer (t_inst pos)
    proj_h_ben   = project_layers(ben_i,  v_inst)
    proj_r_harm  = project_layers(harm_p, v_post)
    proj_r_ben   = project_layers(ben_p,  v_post)

    # Prefill attack (Claim 4 target)
    print("Loading prefill attack ...")
    hi_pre, ref_pre = load_jb("prefill", "hs_inst")
    hp_pre, _        = load_jb("prefill", "hs_post")
    proj_h_pre = project_layers(hi_pre, v_inst)
    proj_r_pre = project_layers(hp_pre, v_post)
    # split
    jb_mask = ~ref_pre
    print(f"prefill: n={len(ref_pre)}  jailbreak_count={jb_mask.sum()}  refused_count={ref_pre.sum()}\n")

    Lp1 = v_inst.shape[0]

    # Print per-layer comparison at t_inst (harmfulness) and t_post (refusal)
    print(f"{'L':>3s}  {'H_harm':>7s} {'H_ben':>7s} {'H_pfl_jb':>9s}  |  {'R_harm':>7s} {'R_ben':>7s} {'R_pfl_jb':>9s}")
    rows = []
    for l in range(1, Lp1):
        row = {
            "layer": l,
            "H_harm":     float(proj_h_harm[:, l].mean()),
            "H_ben":      float(proj_h_ben[:, l].mean()),
            "H_pfl_all":  float(proj_h_pre[:, l].mean()),
            "H_pfl_jb":   float(proj_h_pre[jb_mask, l].mean()) if jb_mask.any() else None,
            "R_harm":     float(proj_r_harm[:, l].mean()),
            "R_ben":      float(proj_r_ben[:, l].mean()),
            "R_pfl_all":  float(proj_r_pre[:, l].mean()),
            "R_pfl_jb":   float(proj_r_pre[jb_mask, l].mean()) if jb_mask.any() else None,
        }
        rows.append(row)
        if l % 2 == 1:
            print(f"{l:>3d}  {row['H_harm']:+7.2f} {row['H_ben']:+7.2f} {row['H_pfl_jb']:+9.2f}  |  {row['R_harm']:+7.2f} {row['R_ben']:+7.2f} {row['R_pfl_jb']:+9.2f}")

    with open(os.path.join(RES_DIR, "jailbreak_layer_scan.json"), "w") as f:
        json.dump({"rows": rows, "n_prefill": int(len(ref_pre)),
                   "n_jailbroken_prefill": int(jb_mask.sum())}, f, indent=2)
    print(f"\nSaved {os.path.join(RES_DIR, 'jailbreak_layer_scan.json')}")

    # Key quantitative test: at the "best" layer (e.g. 15),
    # test whether proj_h(prefill_jb) is close to proj_h(harm baseline) while
    # proj_r(prefill_jb) is close to proj_r(benign baseline).
    for l_test in [11, 15, 20]:
        H_pre = proj_h_pre[jb_mask, l_test]; H_harm = proj_h_harm[:, l_test]; H_ben = proj_h_ben[:, l_test]
        R_pre = proj_r_pre[jb_mask, l_test]; R_harm = proj_r_harm[:, l_test]; R_ben = proj_r_ben[:, l_test]
        # Normalize: fraction of the harm↔benign gap that prefill_jb covers.
        def frac(pre_mean, harm_mean, ben_mean):
            gap = harm_mean - ben_mean
            return (pre_mean - ben_mean) / (gap + 1e-6)
        fh = frac(H_pre.mean(), H_harm.mean(), H_ben.mean())
        fr = frac(R_pre.mean(), R_harm.mean(), R_ben.mean())
        print(f"\nLayer {l_test}: prefill_jb harmfulness position in the harm↔benign gap: "
              f"{fh:.3f}  (1.0 = harm, 0.0 = benign)")
        print(f"          prefill_jb refusal   position in the harm↔benign gap: "
              f"{fr:.3f}  (1.0 = harm, 0.0 = benign)")


if __name__ == "__main__":
    main()
