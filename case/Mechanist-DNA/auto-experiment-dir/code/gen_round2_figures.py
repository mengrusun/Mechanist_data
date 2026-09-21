"""Round-2 publication figures for the Evo2-7B Layer-26 SAE alpha-helix steering knob.
Style matches ../figures_for_paper/gen_composite_figure.py (round-1 reference).
All data pulled from paper_figures/_figure_data.json (built by extract_figure_data.py
from results/*.json -- real experiment output, nothing invented).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

matplotlib.rcParams.update({
    'font.size': 10, 'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 8, 'savefig.dpi': 300, 'axes.grid': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'text.usetex': False, 'mathtext.fontset': 'stix',
})
C = plt.cm.tab10.colors
FS = 10
OUT = "paper_figures"

data = json.load(open(f"{OUT}/_figure_data.json"))
c1, c2, c3 = data["C1"], data["C2"], data["C3"]


def save(fig, name):
    for fmt in ("pdf", "png"):
        p = f"{OUT}/{name}.{fmt}"
        fig.savefig(p, bbox_inches="tight", pad_inches=0.06)
        print("Saved:", p)


# =====================================================================
# Figure 1 (panel a) -- C1 set-level AUROC, grouped bar, mean +/- std (n=3 seeds)
# =====================================================================
def make_c1(ax=None, standalone=True):
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(5.4, 4.2))
        ax = fig.add_subplot(111)

    labels = [f"{c['organism'][:4]}\n{c['helix_def']}" for c in c1["conditions"]]
    means = [c["set_auroc_mean"] for c in c1["conditions"]]
    stds = [c["set_auroc_std"] for c in c1["conditions"]]
    confound = [c["confound_only_mean"] for c in c1["conditions"]]
    shuffle = [c["shuffle_null_mean"] for c in c1["conditions"]]
    x = np.arange(len(labels))

    bars = ax.bar(x, means, yerr=stds, capsize=4, color=[C[0] if 'proka' in l else C[1] for l in labels],
                   alpha=0.85, width=0.6, edgecolor='black', linewidth=0.5,
                   error_kw=dict(elinewidth=1.2, capthick=1.2))
    # per-seed scatter (n=3)
    rng = np.random.default_rng(0)
    for i, cond in enumerate(c1["conditions"]):
        jit = rng.uniform(-0.10, 0.10, size=len(cond["set_auroc_per_seed"]))
        ax.scatter(x[i] + jit, cond["set_auroc_per_seed"], s=14, color='black', zorder=5, alpha=0.7)
    # confound-only / shuffle-null reference markers per bar
    for i in range(len(labels)):
        ax.plot([x[i] - 0.32, x[i] + 0.32], [confound[i]] * 2, color=C[3], lw=1.4, zorder=4,
                 label='confound-only (GC3+codon-pos)' if i == 0 else None)
        ax.plot([x[i] - 0.32, x[i] + 0.32], [shuffle[i]] * 2, color='0.35', lw=1.2, ls=':', zorder=4,
                 label='shuffle-null' if i == 0 else None)
    ax.axhline(0.75, color=C[3], lw=0.8, ls='--', alpha=0.5)
    ax.text(len(labels) - 0.4, 0.755, r'$\tau$=0.75 (single-feature bar)', fontsize=FS - 3, color=C[3], ha='right')

    for i, cond in enumerate(c1["conditions"]):
        top = max(max(cond["set_auroc_per_seed"]), means[i] + stds[i])
        ax.text(x[i], top + 0.018, f"{means[i]:.3f}", ha='center', va='bottom', fontsize=FS - 2)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Set-level test AUROC\n(frozen 19-feature set S)")
    ax.set_ylim(0.45, 1.0)
    ax.legend(frameon=False, loc='upper right', fontsize=FS - 3, ncol=1)
    ax.text(0.02, 0.985, "n=3 seeds/condition (42, 200, 201)\nerror bars = $\\pm$1 SD across seeds",
            transform=ax.transAxes, fontsize=FS - 3, va='top', ha='left', color='0.3')
    if standalone:
        ax.set_title("C1 -- $\\alpha$-helix-selective SAE feature set (M0 existence gate)", fontsize=FS, pad=8)
    return fig, ax


fig, ax = make_c1()
ax.text(-0.18, 1.05, 'a', transform=ax.transAxes, fontsize=15, weight='bold', va='top')
save(fig, "fig1_c1_feature_set_auroc")
plt.close(fig)


# =====================================================================
# Figure 2 (panels b, c) -- C2 dose-response (dual predictor) + valid-ORF overlay
# =====================================================================
def make_c2_dose(ax=None, standalone=True):
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(5.6, 4.2))
        ax = fig.add_subplot(111)
    doses = c2["doses"]
    xpos = np.arange(len(doses))
    xlab = [f"{d:g}" for d in doses]
    markers = {"esmfold": "o", "omegafold": "s"}
    colors = {"esmfold": C[0], "omegafold": C[2]}

    for pred in ["esmfold", "omegafold"]:
        pts = c2["per_predictor"][pred]["points"]
        means = [p["mean"] for p in pts]
        lo = [p["mean"] - p["ci_lo"] for p in pts]
        hi = [p["ci_hi"] - p["mean"] for p in pts]
        ax.errorbar(xpos, means, yerr=[lo, hi], marker=markers[pred], markersize=5,
                     color=colors[pred], capsize=3, linewidth=1.6,
                     label=f"{pred} (mean $\\pm$95% cluster-boot CI)")

    baseline = c2["per_predictor"]["esmfold"]["points"][doses.index(0.0)]["mean"]
    ax.axhline(baseline, color='0.4', linestyle='--', linewidth=0.9, zorder=0)
    ax.text(xpos[0], baseline - 0.03, f"baseline ($c$=0) = {baseline:.3f}", ha='left', va='top',
            fontsize=FS - 2, color='0.4')

    istar = doses.index(c2["c_star"])
    y_star = c2["per_predictor"]["esmfold"]["points"][istar]["mean"]
    i_base = doses.index(0.0)  # capability_preserved_region starts at c=0 (baseline), NOT at the -5.37 sign-check point
    ax.axvspan(xpos[i_base] - 0.5, xpos[istar] + 0.5, color=C[9], alpha=0.06, zorder=0)
    ax.annotate(f"interior optimum\n$c^*$={c2['c_star']:.2f} $\\sigma_{{proj}}$", xy=(xpos[istar], y_star),
                xytext=(xpos[istar] - 2.3, y_star + 0.10), fontsize=FS - 2, ha='center',
                arrowprops=dict(arrowstyle='->', color='0.3', lw=0.8))

    ax.set_xticks(xpos); ax.set_xticklabels(xlab, rotation=40, ha='right', fontsize=FS - 2)
    ax.set_xlabel("Steering strength $c$ ($\\sigma_{proj}$ units)")
    ax.set_ylabel("pLDDT-weighted $\\alpha$-helix\nfraction (helix\\_hgi\\_w)")
    ax.set_ylim(0.4, 0.95)
    rho_e = c2["per_predictor"]["esmfold"]["spearman_rho"]; p_e = c2["per_predictor"]["esmfold"]["spearman_p"]
    rho_o = c2["per_predictor"]["omegafold"]["spearman_rho"]; p_o = c2["per_predictor"]["omegafold"]["spearman_p"]
    ax.text(0.02, 0.97, f"ESM $\\rho$={rho_e:.3f}, p={p_e:.1e}\nOMG $\\rho$={rho_o:.3f}, p={p_o:.1e}",
            transform=ax.transAxes, va='top', ha='left', fontsize=FS - 2)
    ax.legend(frameon=False, loc='lower right', fontsize=FS - 3)
    if standalone:
        ax.set_title("C2 -- dose-response to interior optimum $c^*$", fontsize=FS, pad=8)
    return fig, ax, xpos, xlab, istar


def make_c2_validorf(ax=None, xpos=None, xlab=None, istar=None, standalone=True):
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(5.6, 4.2))
        ax = fig.add_subplot(111)
    doses = c2["doses"]
    for pred, mk, col in [("esmfold", "o", C[0]), ("omegafold", "s", C[2])]:
        pts = c2["per_predictor"][pred]["points"]
        valid = [p["valid_orf_rate"] for p in pts]
        ax.plot(xpos, valid, marker=mk, markersize=5, color=col, linewidth=1.6, label=f"{pred} valid-ORF rate")
    base_valid = c2["base_valid_orf"]
    ax.axhline(base_valid, color='0.4', linestyle='--', linewidth=0.9, zorder=0)
    ax.axhline(0.95 * base_valid, color=C[3], linestyle=':', linewidth=1.0, zorder=0)
    ax.text(xpos[-1], base_valid + 0.010, f"baseline = {base_valid:.3f}", ha='right', va='bottom',
            fontsize=FS - 2, color='0.4')
    ax.text(xpos[-1], 0.95 * base_valid - 0.012, f"0.95$\\times$baseline (capability floor)", ha='right', va='top',
            fontsize=FS - 3, color=C[3])
    i_degraded_start = 6  # dose index 32.22 is first degraded point
    ax.axvspan(xpos[i_degraded_start] - 0.5, xpos[-1] + 0.5, color=C[3], alpha=0.10, zorder=0)
    ax.text((xpos[i_degraded_start] + xpos[-1]) / 2, 0.635, 'degraded region\n(excluded, plan P7)',
            ha='center', va='bottom', fontsize=FS - 2, color=C[3])
    ax.annotate(f"$c^*$={c2['c_star']:.2f}\ncapability preserved", xy=(xpos[istar], c2["per_predictor"]["esmfold"]["points"][istar]["valid_orf_rate"]),
                xytext=(xpos[istar] - 2.6, 0.755), fontsize=FS - 2, ha='center',
                arrowprops=dict(arrowstyle='->', color='0.3', lw=0.8))
    ax.set_xticks(xpos); ax.set_xticklabels(xlab, rotation=40, ha='right', fontsize=FS - 2)
    ax.set_xlabel("Steering strength $c$ ($\\sigma_{proj}$ units)")
    ax.set_ylabel("Valid-ORF rate")
    ax.set_ylim(0.60, 0.96)
    ax.legend(frameon=False, loc='lower left', fontsize=FS - 3)
    if standalone:
        ax.set_title("C2 -- capability-preservation gate (valid-ORF)", fontsize=FS, pad=8)
    return fig, ax


fig = plt.figure(figsize=(11.0, 4.4))
gs = GridSpec(1, 2, figure=fig, wspace=0.35, left=0.07, right=0.98, top=0.90, bottom=0.14)
axb = fig.add_subplot(gs[0, 0])
_, _, xpos, xlab, istar = make_c2_dose(ax=axb, standalone=False)
axb.text(-0.16, 1.06, 'b', transform=axb.transAxes, fontsize=15, weight='bold', va='top')
axc = fig.add_subplot(gs[0, 1])
make_c2_validorf(ax=axc, xpos=xpos, xlab=xlab, istar=istar, standalone=False)
axc.text(-0.16, 1.06, 'c', transform=axc.transAxes, fontsize=15, weight='bold', va='top')
save(fig, "fig2_c2_dose_response")
plt.close(fig)


# =====================================================================
# Figure 3 (panels d, e) -- C3 specificity: 48-direction random null, both predictors
# =====================================================================
def make_c3(pred, ax=None, standalone=True):
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(5.2, 4.4))
        ax = fig.add_subplot(111)
    pdat = c3["per_predictor"][pred]
    deltas = np.array(pdat["null_deltas"])
    S_delta = pdat["S_delta"]["delta"]
    S_lo, S_hi = pdat["S_delta"]["lo"], pdat["S_delta"]["hi"]
    matched_delta = pdat["matched_control_delta"]["delta"]
    matched_lo = pdat["matched_control_delta"]["lo"]
    matched_hi = pdat["matched_control_delta"]["hi"]
    z = pdat["primary_random_null"]["z_score"]
    p = pdat["primary_random_null"]["empirical_one_sided_p"]
    n = pdat["primary_random_null"]["n_directions"]
    n_ge = pdat["primary_random_null"]["n_null_ge_S"]

    bp = ax.boxplot([deltas], positions=[0], widths=0.5, patch_artist=True, showfliers=False,
                     medianprops=dict(color='black'))
    bp['boxes'][0].set_facecolor(C[8]); bp['boxes'][0].set_alpha(0.55)
    rng = np.random.default_rng(1 if pred == "esmfold" else 2)
    jit = rng.uniform(-0.14, 0.14, size=len(deltas))
    ax.scatter(jit, deltas, s=16, color=C[7], alpha=0.8, zorder=3, edgecolor='white', linewidth=0.3,
               label=f'random directions (n={n})')
    ax.errorbar([0.55], [S_delta], yerr=[[S_delta - S_lo], [S_hi - S_delta]], fmt='none',
                ecolor='black', elinewidth=1.2, capsize=4, zorder=4)
    ax.scatter([0.55], [S_delta], marker='*', s=260, color=C[1], zorder=5, edgecolor='black',
               linewidth=0.6, label='SAE set $S$ (19 feat.)')
    ax.errorbar([1.05], [matched_delta], yerr=[[matched_delta - matched_lo], [matched_hi - matched_delta]],
                fmt='none', ecolor='black', elinewidth=1.2, capsize=4, zorder=4)
    ax.scatter([1.05], [matched_delta], marker='D', s=70, color=C[4], zorder=5, edgecolor='black',
               linewidth=0.5, label='matched control (19 feat., unrelated)')
    ax.axhline(0.0, color='0.5', linestyle='--', linewidth=0.8, zorder=0)
    ax.annotate(f"S = +{S_delta:.3f}", xy=(0.55, S_delta), xytext=(0.62, S_delta + 0.01),
                va='center', fontsize=FS - 1)
    ax.set_xticks([0, 0.55, 1.05]); ax.set_xticklabels(['random\nnull', '$S$', 'matched\ncontrol'])
    ax.set_xlim(-0.5, 1.4)
    ax.set_ylabel(f"$\\Delta$ helix\\_hgi\\_w vs baseline\n({pred})")
    ax.text(0.03, 0.97, f'{n_ge}/{n} null $\\geq$ S\np = {p:.4f}, z = {z:.2f}',
            transform=ax.transAxes, va='top', ha='left', fontsize=FS - 2)
    ax.legend(frameon=False, loc='lower right', fontsize=FS - 3)
    if standalone:
        ax.set_title(f"C3 -- specificity null ({pred})", fontsize=FS, pad=8)
    return fig, ax


fig = plt.figure(figsize=(10.6, 4.6))
gs = GridSpec(1, 2, figure=fig, wspace=0.40, left=0.08, right=0.98, top=0.90, bottom=0.12)
axd = fig.add_subplot(gs[0, 0])
make_c3("esmfold", ax=axd, standalone=False)
axd.text(-0.20, 1.06, 'd', transform=axd.transAxes, fontsize=15, weight='bold', va='top')
axe = fig.add_subplot(gs[0, 1])
make_c3("omegafold", ax=axe, standalone=False)
axe.text(-0.20, 1.06, 'e', transform=axe.transAxes, fontsize=15, weight='bold', va='top')
save(fig, "fig3_c3_specificity_null")
plt.close(fig)


# =====================================================================
# Composite main figure -- a (C1) / b,c (C2) / d,e (C3)
# =====================================================================
fig = plt.figure(figsize=(15.5, 9.2))
gs = GridSpec(2, 6, figure=fig, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.85,
              left=0.045, right=0.99, top=0.95, bottom=0.08)

ax_a = fig.add_subplot(gs[0, 0:2])
make_c1(ax=ax_a, standalone=False)
ax_a.text(-0.28, 1.06, 'a', transform=ax_a.transAxes, fontsize=16, weight='bold', va='top')

ax_b = fig.add_subplot(gs[0, 2:4])
_, _, xpos, xlab, istar = make_c2_dose(ax=ax_b, standalone=False)
ax_b.text(-0.20, 1.06, 'b', transform=ax_b.transAxes, fontsize=16, weight='bold', va='top')

ax_c = fig.add_subplot(gs[0, 4:6])
make_c2_validorf(ax=ax_c, xpos=xpos, xlab=xlab, istar=istar, standalone=False)
ax_c.text(-0.20, 1.06, 'c', transform=ax_c.transAxes, fontsize=16, weight='bold', va='top')

ax_d = fig.add_subplot(gs[1, 0:3])
make_c3("esmfold", ax=ax_d, standalone=False)
ax_d.text(-0.14, 1.06, 'd', transform=ax_d.transAxes, fontsize=16, weight='bold', va='top')

ax_e = fig.add_subplot(gs[1, 3:6])
make_c3("omegafold", ax=ax_e, standalone=False)
ax_e.text(-0.14, 1.06, 'e', transform=ax_e.transAxes, fontsize=16, weight='bold', va='top')

fig.suptitle("Evo2-7B Layer-26 SAE $\\alpha$-helix steering knob -- Round 2 (hardened)", fontsize=13, y=0.995)
save(fig, "composite_figure_round2")
plt.close(fig)

print("\nAll figures generated.")
