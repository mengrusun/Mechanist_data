"""Aggregate the beam-search grid: quality vs compute, yield, scorer reliability, diversity."""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(__file__))
import mc_env
import numpy as np
from scipy.stats import spearmanr

R = os.path.join(mc_env.MC, "results")
K = "struct_esmfold"


def load():
    cfg = {}
    for f in sorted(glob.glob(os.path.join(R, "beam_*.json"))):
        d = json.load(open(f))
        cfg[(d["config"]["arm"], d["config"]["width"])] = d
    return cfg


def vals(d, key):
    """per-prompt endpoint, keyed by prompt idx (None where no valid ORF / fold failed)"""
    out = {}
    for r in d["records"]:
        s = r.get(K)
        out[r["idx"]] = s.get(key) if s else None
    return out


def paired_boot(a, b, n_boot=5000, seed=0):
    """paired bootstrap over prompts of mean(a) - mean(b); a,b are {idx: val or None}"""
    sh = [i for i in a if a[i] is not None and b.get(i) is not None]
    if len(sh) < 5:
        return None
    d = np.array([a[i] - b[i] for i in sh])
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return {"diff": float(d.mean()), "lo": float(np.percentile(bs, 2.5)),
            "hi": float(np.percentile(bs, 97.5)), "n_paired": len(sh)}


def main():
    cfg = load()
    ref = cfg.get(("base", 0))
    rows = []
    for (arm, W), d in sorted(cfg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        h = vals(d, "helix_hgi"); hw = vals(d, "helix_hgi_w"); pl = vals(d, "struct_mean_plddt")
        fv = [v for v in h.values() if v is not None]
        c = d["cost"]; y = d["yield"]
        row = {
            "arm": arm, "width": W, "n_delivered": y["n_delivered"],
            "n_folded": len(fv), "valid_orf_rate": y["valid_orf_rate"],
            "helix_hgi": float(np.mean(fv)) if fv else None,
            "helix_hgi_sem": float(np.std(fv) / np.sqrt(len(fv))) if fv else None,
            "helix_hgi_w": float(np.mean([v for v in hw.values() if v is not None])) if fv else None,
            "plddt": float(np.mean([v for v in pl.values() if v is not None])) if fv else None,
            "nt_per_delivered": c["nt_generated_per_delivered"],
            "scorer_calls_per_delivered": c["scorer_calls_per_delivered"],
            "gpu_s_per_delivered": c["gpu_seconds_per_delivered"],
            "gpu_s_per_valid": c["gpu_seconds_per_delivered"] / max(y["valid_orf_rate"], 1e-9),
        }
        if ref is not None:
            row["vs_base_W0"] = paired_boot(h, vals(ref, "helix_hgi"))
        rows.append(row)

    # scorer reliability on GENERATED sequences (the number the comparison hinges on)
    sp, scf, hv = [], [], []
    for d in cfg.values():
        for r in d["records"]:
            if r.get(K) and r.get("score_probe") is not None:
                sp.append(r["score_probe"]); scf.append(r["score_cf"])
                hv.append(r[K]["helix_hgi"])
    scorer = {}
    if len(hv) > 20:
        z = lambda v: (np.array(v) - np.mean(v)) / (np.std(v) + 1e-9)
        scorer = {"n": len(hv),
                  "rho_probe_vs_esmfold_helix": float(spearmanr(sp, hv)[0]),
                  "rho_chou_fasman_vs_esmfold_helix": float(spearmanr(scf, hv)[0]),
                  "rho_ensemble_vs_esmfold_helix": float(spearmanr(z(sp) + z(scf), hv)[0])}

    div = {}
    dpath = os.path.join(R, "diversity.json")
    if os.path.exists(dpath):
        div = json.load(open(dpath))

    out = {"rows": rows, "scorer_reliability_on_generated": scorer, "diversity": div}
    json.dump(out, open(os.path.join(R, "summary.json"), "w"), indent=2)

    # ---- markdown ----
    L = []
    L.append("| arm | W | nt/seq | scorer calls/seq | GPU-s/seq | valid-ORF | helix_hgi | pLDDT | Δ vs base W=0 [95% CI] |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        v = r.get("vs_base_W0")
        dv = f"{v['diff']:+.4f} [{v['lo']:+.4f}, {v['hi']:+.4f}]" if v else "—"
        hx = (f"{r['helix_hgi']:.4f} ± {r['helix_hgi_sem']:.4f}"
              if r["helix_hgi"] is not None else "not folded")
        pl = f"{r['plddt']:.1f}" if r["plddt"] is not None else "—"
        L.append(f"| {r['arm']} | {r['width']} | {r['nt_per_delivered']:.0f} | "
                 f"{r['scorer_calls_per_delivered']:.0f} | {r['gpu_s_per_delivered']:.2f} | "
                 f"{r['valid_orf_rate']:.2f} | {hx} | {pl} | {dv} |")
    md = "\n".join(L)
    if scorer:
        md += ("\n\n**Scorer reliability on generated sequences** (Spearman rho vs held-out "
               f"ESMFold+DSSP helix, n={scorer['n']}): probe {scorer['rho_probe_vs_esmfold_helix']:.3f}, "
               f"Chou-Fasman {scorer['rho_chou_fasman_vs_esmfold_helix']:.3f}, "
               f"ensemble {scorer['rho_ensemble_vs_esmfold_helix']:.3f}")
    if div:
        md += "\n\n| config | n | clusters@50% | clusters@90% | mean pairwise id |\n|---|---|---|---|---|\n"
        for k, v in sorted(div.items()):
            md += (f"| {k} | {v['n']} | {v['clusters_50']} | {v['clusters_90']} | "
                   f"{v['mean_pairwise_identity']:.3f} |\n")
    open(os.path.join(R, "summary.md"), "w").write(md)
    print(md, flush=True)


if __name__ == "__main__":
    main()
