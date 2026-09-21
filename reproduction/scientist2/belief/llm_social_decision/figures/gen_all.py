"""Generate all Ledger Figures for the 4 claims. Fail-soft per figure."""
from __future__ import annotations
import json
import os
import sys
import glob
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import plt, V_COLORS, save_fig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

def _read(p):
    with open(p) as f:
        return json.load(f)

def _now():
    return datetime.now(timezone.utc).isoformat()

VS = ["G", "A", "I", "M"]

# ---------------- C1 figures ----------------

def c1_lopo_vs_probe_by_layer(out_dir):
    fid = "c1_lopo_vs_probe_by_layer"
    stem = f"{out_dir}/{fid}"
    caption = ("Per-layer 5-fold probe cv_acc (dashed) vs. leave-one-phrasing-out cv_acc (solid) per variable "
               "V ∈ {G,A,I,M} on Llama-3.1-8B-Instruct — reveals the phrasing-detector artifact at shallow "
               "layers (V=M ℓ=2 LOPO=0.483 = chance) and the productive mid-layer bands 10-14 and 28-32.")
    probe = _read("runs/M_main_v1/artifacts/m2/probe_accuracy.json")
    lopo = _read("runs/iteration_round_1/c1_lexical_stress_test.json")

    fig, ax = plt.subplots(1, 1, figsize=(5.6, 3.5))
    for V in VS:
        rows = sorted(probe[V], key=lambda r: r["layer_abs"])
        layers = [r["layer_abs"] for r in rows]
        cv = [r["cv_acc"] for r in rows]
        ax.plot(layers, cv, linestyle="--", color=V_COLORS[V], alpha=0.55, linewidth=1.2)
    for V in VS:
        entries = lopo[V].get("baseline_only", {})
        pairs = sorted(((int(l), v["lopo_mean"]) for l, v in entries.items()), key=lambda t: t[0])
        if not pairs:
            continue
        layers = [p[0] for p in pairs]
        acc = [p[1] for p in pairs]
        ax.plot(layers, acc, color=V_COLORS[V], linewidth=1.6, label=f"V={V}")

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(0.02, 0.51, "chance", color="gray", fontsize=8, transform=ax.get_yaxis_transform())
    ax.axvspan(10, 14, color="tab:green", alpha=0.06, label="productive band 10-14")
    ax.axvspan(28, 32, color="tab:purple", alpha=0.06, label="productive band 28-32")
    ax.set_xlabel("residual-stream layer")
    ax.set_ylabel("cv accuracy")
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False, loc="lower right", ncol=2)
    save_fig(fig, stem)
    return {
        "id": fid, "type": "line", "caption": caption,
        "png": f"figures/C1/{fid}.png", "pdf": f"figures/C1/{fid}.pdf",
        "md": None, "tex": None,
        "source_data": "runs/iteration_round_1/c1_lexical_stress_test.json + runs/M_main_v1/artifacts/m2/probe_accuracy.json",
        "status": "ok",
    }


def c1_probe_transfer_stats(out_dir):
    fid = "c1_probe_transfer_stats"
    caption = "C1 main-experiment probe cv_acc + projection-transfer β at picked layers ell_V* (Llama-3.1-8B-Instruct)"
    rows = [
        ("G", 4, 1.000, 1.000, -10.12, 0.002, -0.380),
        ("A", 6, 1.000, 1.000,  -4.85, 0.059, +0.106),
        ("I", 2, 1.000, 1.000, +32.10, 1.5e-9, +1.287),
        ("M", 2, 1.000, 1.000, +41.28, 3.5e-5, +0.713),
    ]
    md_lines = [
        "| V | ell_V* | probe cv_acc | held_acc | proj-transfer beta | beta p-value | baseline tau effect |",
        "|---|-------:|-------------:|---------:|-------------------:|-------------:|--------------------:|",
    ]
    for V, ell, cv, held, beta, p, base in rows:
        md_lines.append(f"| {V} | {ell} | {cv:.3f} | {held:.3f} | {beta:+.2f} | {p:.2g} | {base:+.3f} |")
    md = "\n".join(md_lines) + "\n"
    with open(f"{out_dir}/{fid}.md", "w") as f:
        f.write(md)

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption.replace("_", r"\_") + "}",
        r"\label{tab:" + fid + "}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"$V$ & $\ell_V^*$ & probe cv\_acc & held\_acc & proj-transfer $\beta$ & $\beta$ p-value & baseline $\tau$ effect \\",
        r"\midrule",
    ]
    for V, ell, cv, held, beta, p, base in rows:
        tex_lines.append(f"{V} & {ell} & {cv:.3f} & {held:.3f} & ${beta:+.2f}$ & {p:.2g} & ${base:+.3f}$ \\\\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(f"{out_dir}/{fid}.tex", "w") as f:
        f.write("\n".join(tex_lines))
    return {
        "id": fid, "type": "table", "caption": caption,
        "png": None, "pdf": None,
        "md": f"figures/C1/{fid}.md", "tex": f"figures/C1/{fid}.tex",
        "source_data": "refine-logs/EXPERIMENT_RESULTS.md",
        "status": "ok",
    }


# ---------------- C2 figures ----------------

def c2_crossleakage_gs_vs_leace(out_dir):
    fid = "c2_crossleakage_gs_vs_leace"
    stem = f"{out_dir}/{fid}"
    caption = ("4x4 cross-leakage probe accuracy matrix under GS vs LEACE decorrelators at Llama-3.1-8B-Instruct "
               "picked shallow layers ℓ_V*. Diagonal = target-self, off-diagonal = leakage. "
               "GS: diag mean 0.958 / off-diag max 0.506. LEACE: diag mean 0.573 / off-diag max 0.829.")
    leakage = _read("runs/M_main_v1/artifacts/m3/leakage_matrix.json")

    def to_matrix(rows):
        m = [[0.0]*4 for _ in range(4)]
        idx = {v: i for i, v in enumerate(VS)}
        for r in rows:
            m[idx[r[0]]][idx[r[1]]] = r[2]
        return m

    gs = to_matrix(leakage["gs"])
    leace = to_matrix(leakage["leace"])

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4))
    for ax, mat, title in ((axes[0], gs, "GS"), (axes[1], leace, "LEACE")):
        im = ax.imshow(mat, cmap="viridis", vmin=0.4, vmax=1.0, aspect="equal")
        ax.set_xticks(range(4)); ax.set_xticklabels(VS)
        ax.set_yticks(range(4)); ax.set_yticklabels(VS)
        ax.set_xlabel("probe target W")
        if ax is axes[0]:
            ax.set_ylabel("direction source V")
        ax.set_title(title)
        for i in range(4):
            for j in range(4):
                val = mat[i][j]
                color = "white" if val < 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.03)
    cbar.set_label("probe accuracy")
    save_fig(fig, stem)
    return {
        "id": fid, "type": "heatmap", "caption": caption,
        "png": f"figures/C2/{fid}.png", "pdf": f"figures/C2/{fid}.pdf",
        "md": None, "tex": None,
        "source_data": "runs/M_main_v1/artifacts/m3/leakage_matrix.json",
        "status": "ok",
    }


def c2_purity_summary(out_dir):
    fid = "c2_purity_summary"
    caption = "C2 decorrelation summary — GS vs LEACE at picked shallow layers (Llama-3.1-8B-Instruct)"
    rows = [
        ("raw",           0.934, 0.694, "baseline", "no"),
        ("mean-centered", 0.934, 0.694, "yes",      "no"),
        ("GS",            0.958, 0.506, "yes",      "yes"),
        ("LEACE",         0.573, 0.829, "no",       "no"),
    ]
    md_lines = [
        "| decorrelator | diag mean | off-diag max | preservation >= 0.95x raw diag? | off-diag <= chance+0.05 (0.55)? |",
        "|--------------|----------:|-------------:|:-------------------------------:|:-------------------------------:|",
    ]
    for name, diag, off, presv, thresh in rows:
        md_lines.append(f"| {name} | {diag:.3f} | {off:.3f} | {presv} | {thresh} |")
    md = "\n".join(md_lines) + "\n"
    with open(f"{out_dir}/{fid}.md", "w") as f:
        f.write(md)
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption.replace("_", r"\_") + "}",
        r"\label{tab:" + fid + "}",
        r"\begin{tabular}{lccll}",
        r"\toprule",
        r"decorrelator & diag mean & off-diag max & preservation $\geq 0.95\times$ raw? & off-diag $\leq$ chance$+0.05$ (0.55)? \\",
        r"\midrule",
    ]
    for name, diag, off, presv, thresh in rows:
        tex_lines.append(f"{name} & {diag:.3f} & {off:.3f} & {presv} & {thresh} \\\\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(f"{out_dir}/{fid}.tex", "w") as f:
        f.write("\n".join(tex_lines))
    return {
        "id": fid, "type": "table", "caption": caption,
        "png": None, "pdf": None,
        "md": f"figures/C2/{fid}.md", "tex": f"figures/C2/{fid}.tex",
        "source_data": "refine-logs/EXPERIMENT_RESULTS.md",
        "status": "ok",
    }


# ---------------- C3 figures ----------------

def c3_dose_response_L16(out_dir):
    fid = "c3_dose_response_L16"
    stem = f"{out_dir}/{fid}"
    caption = ("Mean transfer amount vs. signed alpha at L=16 per variable V ∈ {G,A,I,M} — "
               "Llama-3.1-8B-Instruct main (M4-supp) vs Meta-Llama-3-8B-Instruct verify variant. "
               "V=M sign-inverts at alpha=+2 sigma in both models; V=A amplifies 5.5x; V=G amplifies 2.5x at alpha=-2 sigma; "
               "V=I bidirectional (+46% amplify at alpha=-2, -22% attenuate at alpha=+2).")

    # Llama-3.1 main-supp at L=16 (from EXPERIMENT_RESULTS.md M4-supp table)
    main_L16 = {
        "G": {-2:  9.93, -1: 9.76, 0: 9.64, 1: 9.64, 2: 9.64},
        "A": {-2:  9.71, -1: 9.64, 0: 9.64, 1: 9.74, 2: 9.60},
        "I": {-2: 11.10, -1: 10.80, 0: 10.51, 1: 10.35, 2: 10.02},
        "M": {-2: 10.72, -1: 10.36, 0: 10.02,  1: 9.68, 2:  9.31},
    }
    # Verify variant (Meta-Llama-3) L=16 — pull from summary.json
    swap = _read("runs/verify_C3_variant_model_swap_v1/artifacts/steer/summary.json")
    swap_L16 = {V: {} for V in VS}
    for r in swap:
        if r["layer_abs"] == 16:
            swap_L16[r["V"]][r["alpha_mult"]] = r["mean_transfer"]

    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.6), sharey=True)
    alphas = [-2, -1, 0, 1, 2]
    for ax, V in zip(axes, VS):
        m_main = [main_L16[V].get(a, None) for a in alphas]
        m_swap = [swap_L16[V].get(a, None) for a in alphas]
        ax.plot(alphas, m_main, marker="o", color=V_COLORS[V], linewidth=1.5, label="Llama-3.1-8B-Instruct (main-supp)")
        if any(v is not None for v in m_swap):
            ax.plot(alphas, m_swap, marker="s", linestyle="--", color=V_COLORS[V], alpha=0.7, linewidth=1.2, label="Meta-Llama-3-8B-Instruct (verify variant)")
        ax.set_title(f"V={V}")
        ax.set_xlabel(r"$\alpha$ ($\sigma_{\mathrm{proj}}$ units)")
        ax.axhline(10, color="gray", linestyle=":", linewidth=0.7)
        ax.set_xticks(alphas)
    axes[0].set_ylabel("mean transfer $\\tau$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.08), fontsize=8)
    fig.tight_layout()
    save_fig(fig, stem)
    return {
        "id": fid, "type": "grouped_bar", "caption": caption,
        "png": f"figures/C3/{fid}.png", "pdf": f"figures/C3/{fid}.pdf",
        "md": None, "tex": None,
        "source_data": "runs/M4_supp_deep_v1/m4/ (via EXPERIMENT_RESULTS.md#M4) + runs/verify_C3_variant_model_swap_v1/artifacts/steer/summary.json",
        "status": "ok",
    }


def c3_direction_specificity(out_dir):
    fid = "c3_direction_specificity"
    caption = ("C3 direction-specificity sigma vs random-direction null at L=16 (n=10 seeds) — "
               "Llama-3.1-8B-Instruct main; >=2.5 sigma for all four showcase effects, V=M inversion at 2.84 sigma.")
    # Values taken from AUTO_ITERATION_FINAL_REPORT.md §1.1 direction-specificity summary
    rows = [
        ("M", "+2", "sign inversion",  2.84),
        ("A", "+2", "amplification",   2.69),
        ("G", "-2", "amplification",   4.30),
        ("I", "-2", "amplification",   2.50),
    ]
    md_lines = [
        "| V | alpha (sigma_proj units) | effect type | direction-specificity vs random null (sigma) |",
        "|:-:|:------------------------:|:-----------:|---------------------------------------------:|",
    ]
    for V, a, kind, sig in rows:
        md_lines.append(f"| {V} | {a} | {kind} | {sig:.2f} sigma |")
    md = "\n".join(md_lines) + "\n"
    with open(f"{out_dir}/{fid}.md", "w") as f:
        f.write(md)
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption.replace("_", r"\_") + "}",
        r"\label{tab:" + fid + "}",
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"$V$ & $\alpha$ ($\sigma_{\mathrm{proj}}$) & effect type & direction-specificity ($\sigma$) \\",
        r"\midrule",
    ]
    for V, a, kind, sig in rows:
        tex_lines.append(f"{V} & {a} & {kind} & ${sig:.2f}\\sigma$ \\\\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(f"{out_dir}/{fid}.tex", "w") as f:
        f.write("\n".join(tex_lines))
    return {
        "id": fid, "type": "table", "caption": caption,
        "png": None, "pdf": None,
        "md": f"figures/C3/{fid}.md", "tex": f"figures/C3/{fid}.tex",
        "source_data": "runs/iteration_round_1/L16_random_control_extended/ + review-stage/AUTO_ITERATION_FINAL_REPORT.md",
        "status": "ok",
    }


# ---------------- C4 figures ----------------

def _c4_matrix_from_dir(dirpath, alpha_mult):
    """Read w_effects[W](alpha=alpha_mult) − w_effects[W](alpha=0) → 4x4 matrix."""
    zero = {V: None for V in VS}
    hit = {V: None for V in VS}
    for f in glob.glob(f"{dirpath}/*.json"):
        try:
            d = _read(f)
        except Exception:
            continue
        V = d.get("V"); a = d.get("alpha_mult")
        if V not in VS: continue
        if a == 0:
            zero[V] = d.get("w_effects", {})
        elif a == alpha_mult:
            hit[V] = d.get("w_effects", {})
    mat = [[0.0]*4 for _ in range(4)]
    for i, V in enumerate(VS):
        z = zero.get(V) or {}
        h = hit.get(V) or {}
        for j, W in enumerate(VS):
            mat[i][j] = (h.get(W, 0.0) or 0.0) - (z.get(W, 0.0) or 0.0)
    return mat


def c4_selectivity_matrix_layers(out_dir):
    fid = "c4_selectivity_matrix_layers"
    stem = f"{out_dir}/{fid}"
    caption = ("4x4 selectivity matrix M[V,W] = w_effects[W](alpha=+2 sigma) - w_effects[W](alpha=0) "
               "at Llama-3.1-8B-Instruct picked shallow layers ell_V* (LEACE-pure, single-site) vs. L=16 "
               "(iteration ②, raw v_hat_V). Left: shallow (permutation p=0.84, V=G collapse). "
               "Right: L=16 (permutation p=0.128, partially selective for V=A/M, non-selective for V=I, one-signed for V=G).")

    # Shallow: reconstructed from EXPERIMENT_RESULTS.md M5 (LEACE, single-site, α=+2σ at ell_V*)
    shallow = [
        [0.000,  0.000,  0.000,  0.000],  # V=G
        [-0.450, -0.041, -0.455, -0.446], # V=A
        [-0.200,  0.204, -0.202, -0.198], # V=I
        [ 0.250, -0.245,  0.248, -0.248], # V=M
    ]
    L16 = _c4_matrix_from_dir("runs/iteration_round_1/L16_c4_selectivity", alpha_mult=2)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4))
    vmax = max(1.0, max(abs(v) for row in shallow+L16 for v in row))
    for ax, mat, title in ((axes[0], shallow, r"shallow $\ell_V^*$" + " (main, α=+2σ)"),
                            (axes[1], L16, "L=16 (iteration, α=+2σ)")):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")
        ax.set_xticks(range(4)); ax.set_xticklabels(VS)
        ax.set_yticks(range(4)); ax.set_yticklabels(VS)
        ax.set_xlabel("measured W-effect")
        if ax is axes[0]:
            ax.set_ylabel("steered V")
        ax.set_title(title)
        for i in range(4):
            for j in range(4):
                v = mat[i][j]
                color = "white" if abs(v) > 0.6 * vmax else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.03)
    cbar.set_label(r"$\Delta$ W-effect")
    save_fig(fig, stem)
    return {
        "id": fid, "type": "heatmap", "caption": caption,
        "png": f"figures/C4/{fid}.png", "pdf": f"figures/C4/{fid}.pdf",
        "md": None, "tex": None,
        "source_data": "runs/M_main_v1/artifacts/m5/ + runs/iteration_round_1/L16_c4_selectivity/",
        "status": "ok",
    }


def c4_perV_selectivity_ratio(out_dir):
    fid = "c4_perV_selectivity_ratio"
    caption = ("C4 per-V diagonal / mean-off-diagonal ratio at L=16 (alpha=+2 sigma) — "
               "V=A 2.25, V=M 2.19 (selective); V=I 1.00 (non-selective); V=G 0.00 at +sigma (only steers at -sigma).")
    rows = [
        ("A", 2.25, "selective"),
        ("M", 2.19, "selective"),
        ("I", 1.00, "non-selective (co-modulates other Ws)"),
        ("G", 0.00, "one-signed (only steers at alpha=-sigma; diag=-0.570 there)"),
    ]
    md_lines = [
        "| V | diag / mean-off-diag ratio (L=16, alpha=+2 sigma) | notes |",
        "|:-:|--------------------------------------------------:|-------|",
    ]
    for V, ratio, note in rows:
        md_lines.append(f"| {V} | {ratio:.2f} | {note} |")
    md = "\n".join(md_lines) + "\n"
    with open(f"{out_dir}/{fid}.md", "w") as f:
        f.write(md)
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + caption.replace("_", r"\_") + "}",
        r"\label{tab:" + fid + "}",
        r"\begin{tabular}{ccl}",
        r"\toprule",
        r"$V$ & diag / mean-off-diag ($L=16, \alpha=+2\sigma$) & notes \\",
        r"\midrule",
    ]
    for V, ratio, note in rows:
        tex_lines.append(f"{V} & {ratio:.2f} & {note} \\\\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(f"{out_dir}/{fid}.tex", "w") as f:
        f.write("\n".join(tex_lines))
    return {
        "id": fid, "type": "table", "caption": caption,
        "png": None, "pdf": None,
        "md": f"figures/C4/{fid}.md", "tex": f"figures/C4/{fid}.tex",
        "source_data": "runs/iteration_round_1/L16_c4_selectivity/ (per-cell w_effects) + AUTO_ITERATION_FINAL_REPORT.md",
        "status": "ok",
    }


PLAN = {
    "C1": {"title": "Linear encoding of each social/contextual variable",
           "figures": [c1_lopo_vs_probe_by_layer, c1_probe_transfer_stats]},
    "C2": {"title": "Purity via decorrelation",
           "figures": [c2_crossleakage_gs_vs_leace, c2_purity_summary]},
    "C3": {"title": "Bidirectional causal steering",
           "figures": [c3_dose_response_L16, c3_direction_specificity]},
    "C4": {"title": "Selectivity 4x4 matrix",
           "figures": [c4_selectivity_matrix_layers, c4_perV_selectivity_ratio]},
}

if __name__ == "__main__":
    for claim_id, spec in PLAN.items():
        out_dir = f"figures/{claim_id}"
        os.makedirs(out_dir, exist_ok=True)
        entries = []
        skipped = []
        for gen in spec["figures"]:
            try:
                entries.append(gen(out_dir))
                print(f"  ok: {claim_id}/{gen.__name__}")
            except Exception as e:
                trace = traceback.format_exc()
                print(f"  ERROR {claim_id}/{gen.__name__}: {e}")
                print(trace, file=sys.stderr)
                entries.append({
                    "id": gen.__name__, "type": "unknown",
                    "caption": "", "png": None, "pdf": None, "md": None, "tex": None,
                    "source_data": "", "status": "error", "error_detail": str(e),
                })
        idx = {
            "claim_id": claim_id,
            "claim_title": spec["title"],
            "generated_at": _now(),
            "figures": entries,
            "skipped": skipped,
        }
        with open(f"{out_dir}/INDEX.json", "w") as f:
            json.dump(idx, f, indent=2)
        print(f"wrote {out_dir}/INDEX.json ({sum(1 for e in entries if e['status']=='ok')}/{len(entries)} ok)")
    print("\nDone.")
