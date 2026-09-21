"""Regenerate panels b and c from the RAW experiment records.

Panel b  grouped bars -- Rate of alpha-helix (%) for three conditions at three pLDDT cuts.
Panel c  dose-response -- alpha-helix (%) + valid-ORF vs the steering coefficient alpha.

Everything is recomputed from results/*.json on every run; no cached CSV is read back in.
The CSVs this script writes are provenance dumps (outputs, never inputs).

Data provenance
---------------
predictor            ESMFold only ('per_predictor.esmfold').
metric               UNWEIGHTED helix_hgi = DSSP H/G/I residues / full length, per sample,
                     averaged over samples, x100.
steering arm         300 unique prokaryote CDS prompts x 3 seeds {42, 200, 201} per dose.
random arm           48 norm-matched random directions x 120 prompts, single global seed,
                     pooled across all directions (16 files x 3 directions).
alpha axis           alpha = c_sigma / 2.685; the locked dose c* = 21.4801 sigma_proj -> alpha = 8.
inclusion            the only exclusion is the ORF filter (translate_orf, protein >= 30 aa);
                     ESMFold folded every valid ORF, so n_plotted = n_valid_orf = n_folded.

Known deviation from the paper figure
-------------------------------------
One bar label differs by 0.1: the pLDDT >= 0.5 / helix-steer bar. The raw data give
56.945880 %, which is 56.9 to one decimal. The published figure prints 57.0 -- a double
rounding (56.9459 -> 56.95 in the data-source table -> 57.0 on the figure). This script
prints the correct 56.9. Every other label, curve point, valid-ORF value and sample count
reproduces the published figure exactly.

Bootstrap note
--------------
The helix CI is a 2000-resample percentile bootstrap of the mean (numpy default_rng(0)).
Resampling indexes the pooled array, so the CI depends on the ORDER the three seeds were
concatenated in. The original code pooled in unsorted glob order, i.e. filesystem readdir
order at that moment -- which is not reproduced by today's glob, nor by mtime, inode or
filename order (4/10, 3/10, 1/10, 0/10 doses respectively).

That order was instead recovered by exhaustive search: for each dose all 3! = 6 seed
permutations were bootstrapped and compared against the published bounds. Exactly one
permutation matches per dose, all 10 doses, to < 0.0005 pp -- they are frozen in POOL_ORDER
below, so the band reproduces the paper's panel c bit-for-bit.

Pass --canonical-order to pool in ascending seed order instead. That is the order-independent
choice, but it shifts the band bounds by up to 0.16 pp away from the published figure. It
changes no point estimate, valid-ORF value or sample count.

Run
---
    conda run -n scientist python paper_figure/scripts/make_panels_bc.py
Outputs (into paper_figure/): panel_b.{pdf,png}, panel_c.{pdf,png},
                              panel_b_data.csv, panel_c_data.csv
"""
import argparse
import collections
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "paper_figure")

PRED = "esmfold"
SEEDS = [42, 200, 201]
C_STAR = 21.4801                      # locked interior dose, in sigma_proj units
SCALE = C_STAR / 8.0                  # c_sigma -> alpha ; 21.4801 -> 8
N_BOOT = 2000
BOOT_SEED = 0

# panel b: (legend label, results glob, key)
CONDITIONS = [
    ("Evo2-7B (no steer)",              f"m2_alpha_helix_S_c0.0_s*.json",       "no_steer"),
    ("steer random feature",            f"m3_random_c{C_STAR:g}_d*.json",       "random"),
    (r"steer $\alpha$-helix features",  f"m2_alpha_helix_S_c{C_STAR:g}_s*.json", "helix"),
]
# panel b: (tick label, pLDDT cut on the 0-1 scale; None = no filter)
CUTS = [("overall", None), ("pLDDT ≥ 0.4", 0.4), ("pLDDT ≥ 0.5", 0.5)]

# Per-dose seed concatenation order for the panel-c bootstrap. THIS IS AN INPUT, not a
# check value: resampling indexes the pooled array, so this order determines the CI band.
# The paper's panel c used the unsorted-glob (readdir) order of that moment, which no current
# file attribute reproduces (mtime 4/10 doses, inode 3/10, filename 1/10, today's glob 0/10).
# Recovered by exhaustive search instead: per dose all 3! = 6 permutations were bootstrapped
# against the published bounds and exactly one matches, for all 10 doses, to < 0.0005 pp.
# Change it and the band moves by up to 0.16 pp; point estimates and counts are unaffected.
POOL_ORDER = {
    -2.0: (200, 42, 201),  0.0: (200, 201, 42),  1.0: (42, 200, 201),
     2.0: (201, 42, 200),  4.0: (42, 200, 201),  8.0: (42, 201, 200),
    12.0: (200, 42, 201), 16.0: (42, 201, 200), 24.0: (42, 200, 201),
    32.0: (201, 42, 200)}

# Colours sampled from a 400-dpi render of the paper's panel b/c (not copied from older code).
C = {"green": "#4C9866", "green_dark": "#2E7B45", "grey": "#6D7075", "red": "#BF4F4C",
     "blue": "#5A7EB2", "ink": "#222222", "shade": "#ECECEC", "grid": "#E5E8E8"}
BAR = {"no_steer": ("#D9D9D9", "#7B7B7B", "#6D7075"),      # fill, edge, value-label
       "random":   ("#C9D8EE", "#4875B5", "#5B7EB3"),
       "helix":    ("#D9EADF", "#227652", "#2E7B45")}
FONT = 9


# --------------------------------------------------------------------------- data
def _per_sample(path):
    """per-sample records for PRED from one results json (m2-style or m3_random-style)."""
    d = json.load(open(path))
    out = []
    if d.get("per_predictor"):                       # m2: one run, one dose/seed
        out += d["per_predictor"][PRED]["per_sample"]
    if "directions" in d:                            # m3_random: several directions per file
        for dd in d["directions"]:
            out += dd["per_predictor"][PRED]["per_sample"]
    return out


def _boot_ci(values, B=N_BOOT, seed=BOOT_SEED):
    """Percentile bootstrap CI of the mean. Order-sensitive: resampling indexes `values`,
    so the caller decides the concatenation order (see POOL_ORDER)."""
    a = np.asarray(values, float)
    if a.size < 2:
        m = float(a.mean()) if a.size else float("nan")
        return m, m
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(B, a.size))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def panel_b_data():
    """-> list of {key, label, rates[3], counts[3]} over CUTS."""
    rows = []
    for label, pat, key in CONDITIONS:
        files = sorted(glob.glob(os.path.join(RES, pat)))
        if not files:
            raise SystemExit(f"no results matched {pat} under {RES}")
        hgi, plddt = [], []
        for f in files:
            for s in _per_sample(f):
                hgi.append(s["helix_hgi"])
                plddt.append(s["plddt"] / 100.0)     # per_sample.plddt is 0-100
        hgi, plddt = np.asarray(hgi, float), np.asarray(plddt, float)
        rates, counts = [], []
        for _, cut in CUTS:
            m = np.ones_like(plddt, bool) if cut is None else plddt >= cut
            rates.append(100.0 * float(hgi[m].mean()))
            counts.append(int(m.sum()))
        rows.append({"key": key, "label": label, "n_files": len(files),
                     "rates": rates, "counts": counts})
    return rows


def panel_c_data(canonical_order=False):
    """-> list of per-alpha dicts, ascending in alpha.

    Seeds are concatenated per POOL_ORDER so the band reproduces the paper's panel c;
    canonical_order=True concatenates in ascending seed order instead.
    """
    hgi = collections.defaultdict(dict)              # c_sigma -> {seed: [helix_hgi]}
    orf = collections.defaultdict(dict)              # c_sigma -> {seed: valid_orf_rate}
    for f in sorted(glob.glob(os.path.join(RES, "m2_alpha_helix_S_c*_s*.json"))):
        d = json.load(open(f))
        cfg = d["config"]
        if cfg["seed"] not in SEEDS:
            continue
        e = d["per_predictor"][PRED]
        hgi[cfg["c_sigma"]][cfg["seed"]] = [s["helix_hgi"] for s in e["per_sample"]]
        orf[cfg["c_sigma"]][cfg["seed"]] = e["valid_orf_rate"]
    if not hgi:
        raise SystemExit(f"no m2_alpha_helix_S_c*_s*.json under {RES}")

    rows = []
    for c in sorted(hgi):
        alpha = round(c / SCALE, 4)
        order = sorted(hgi[c]) if canonical_order else POOL_ORDER.get(alpha)
        if order is None or set(order) != set(hgi[c]):
            if not canonical_order:
                print(f"  ! alpha={alpha}: no recorded pooling order for seeds "
                      f"{sorted(hgi[c])}; falling back to ascending order")
            order = sorted(hgi[c])
        pooled = [v for s in order for v in hgi[c][s]]
        lo, hi = _boot_ci(pooled)
        seed_orf = [float(orf[c][s]) for s in SEEDS]
        rows.append({"alpha": alpha, "c_sigma": round(c, 4),
                     "helix_mean": 100.0 * float(np.mean(pooled)),
                     "helix_lo": 100.0 * lo, "helix_hi": 100.0 * hi,
                     "orf_mean": float(np.mean(seed_orf)),
                     "orf_lo": float(np.min(seed_orf)), "orf_hi": float(np.max(seed_orf)),
                     "n_helix": len(pooled),
                     **{f"orf_s{s}": v for s, v in zip(SEEDS, seed_orf)}})
    return rows


def report(brows, crows):
    """Print the computed numbers behind both panels."""
    print("panel b -- Rate of alpha-helix (%)")
    print(f"  {'condition':<26}" + "".join(f"{lab:>20}" for lab, _ in CUTS))
    for r in brows:
        print(f"  {r['label'][:26]:<26}" +
              "".join(f"{v:9.2f} n={n:<7}" for v, n in zip(r["rates"], r["counts"])))

    print("\npanel c -- dose-response")
    print(f"  {'alpha':>6}  {'helix_mean':>10}  {'helix [lo, hi]':>18}  {'orf_mean':>8}"
          f"  {'orf [lo, hi]':>16}  {'n':>5}")
    for r in crows:
        print(f"  {r['alpha']:6.1f}  {r['helix_mean']:10.3f}  "
              f"[{r['helix_lo']:7.3f},{r['helix_hi']:7.3f}]  {r['orf_mean']:8.4f}  "
              f"[{r['orf_lo']:6.4f},{r['orf_hi']:6.4f}]  {r['n_helix']:5d}")


def dump_csv(brows, crows):
    p = os.path.join(OUT, "panel_b_data.csv")
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "cut", "helix_mean_pct", "n_samples", "n_files"])
        for r in brows:
            for (lab, _), rate, n in zip(CUTS, r["rates"], r["counts"]):
                w.writerow([r["key"], lab.replace("$\\geq$", ">="), f"{rate:.3f}", n, r["n_files"]])
    q = os.path.join(OUT, "panel_c_data.csv")
    fields = ["alpha", "c_sigma", "helix_mean", "helix_lo", "helix_hi", "orf_mean",
              "orf_lo", "orf_hi"] + [f"orf_s{s}" for s in SEEDS] + ["n_helix"]
    # helix at 3 dp, ORF at 4 dp -- the precision the paper's panel-c numbers were stored at,
    # so this dump diffs to exactly zero against them.
    dp = {"helix_mean": 3, "helix_lo": 3, "helix_hi": 3}
    with open(q, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in crows:
            w.writerow({k: (round(r[k], dp.get(k, 4)) if isinstance(r[k], float) else r[k])
                        for k in fields})
    print(f"wrote {os.path.relpath(p, ROOT)} and {os.path.relpath(q, ROOT)}")


# --------------------------------------------------------------------------- plotting
def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "text.usetex": False,
        "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": None,
    })
    return plt


def _bare(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["ink"])
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=C["ink"], labelcolor=C["ink"], width=0.8, length=2.6,
                   labelsize=FONT - 1)


def plot_panel_b(rows, letter=True):
    plt = _style()
    from matplotlib.patches import Patch
    mm = 1 / 25.4
    fig = plt.figure(figsize=(100 * mm, 62 * mm))
    ax = fig.add_axes([0.115, 0.125, 0.865, 0.845])
    _bare(ax)

    x = np.arange(len(CUTS), dtype=float)
    w = 0.24
    for i, r in enumerate(rows):
        fill, edge, lab_c = BAR[r["key"]]
        pos = x + (i - 1) * w
        ax.bar(pos, r["rates"], w, color=fill, edgecolor=edge, linewidth=0.9, zorder=3)
        for px, v in zip(pos, r["rates"]):
            ax.text(px, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=FONT - 2.5,
                    color=lab_c, fontweight="bold", zorder=4)

    ax.set_ylim(0, 84)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_xlim(-0.5, len(CUTS) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, _ in CUTS], fontsize=FONT)
    ax.set_ylabel(r"Rate of $\alpha$-helix (%)", fontsize=FONT, color=C["ink"])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=C["grid"], lw=0.8)
    ax.tick_params(axis="x", length=0)

    handles = [Patch(facecolor=BAR[r["key"]][0], edgecolor=BAR[r["key"]][1], lw=0.9,
                     label=r["label"]) for r in rows]
    ax.legend(handles=handles, frameon=False, fontsize=FONT, loc="upper left",
              handlelength=1.5, handleheight=1.0, labelspacing=0.32, borderpad=0.1)
    if letter:
        fig.text(0.012, 0.975, "b", fontsize=13, fontweight="bold", va="top", ha="left",
                 color=C["ink"])

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"panel_b.{ext}"))
    plt.close(fig)
    print("wrote paper_figure/panel_b.pdf/.png")


def plot_panel_c(rows, letter=True):
    plt = _style()
    from matplotlib.ticker import MultipleLocator
    g = lambda k: np.array([r[k] for r in rows], float)
    x, med, hlo, hhi = g("alpha"), g("helix_mean"), g("helix_lo"), g("helix_hi")
    om, olo, ohi = g("orf_mean"), g("orf_lo"), g("orf_hi")

    i0 = int(np.where(x == 0.0)[0][0])
    istar = int(np.where(x == 8.0)[0][0])
    base_h, base_orf, peak_h = med[i0], om[i0], med[istar]
    # the figure prints the gain from the ROUNDED endpoints: 56.6 - 43.8 = 12.8
    gain = round(peak_h, 1) - round(base_h, 1)
    xlo, xhi = x[0] - 1.2, x[-1] + 1.2

    mm = 1 / 25.4
    fig = plt.figure(figsize=(95 * mm, 60 * mm))
    axo = fig.add_axes([0.150, 0.170, 0.830, 0.200])          # valid-ORF strip (bottom)
    ax = fig.add_axes([0.150, 0.405, 0.830, 0.570])           # alpha-helix (top)

    for a in (ax, axo):
        _bare(a)
        a.axvspan(8.0, xhi, color=C["shade"], lw=0, zorder=1)
        a.axvline(8.0, ls=(0, (4, 3)), lw=1.1, color=C["red"], zorder=5)
        a.set_xlim(xlo, xhi)
        a.xaxis.set_major_locator(MultipleLocator(8))
        a.xaxis.set_minor_locator(MultipleLocator(4))

    # ---- top: alpha-helix dose-response + bootstrap CI band ----
    for yv in (40, 60, 80, 100):
        ax.axhline(yv, color=C["grid"], lw=0.9, zorder=1.5)
    ax.fill_between(x, hlo, hhi, color=C["green"], alpha=0.28, lw=0, zorder=2)
    ax.plot(x, med, "-", color=C["green"], marker="o", markersize=3.2, linewidth=1.4,
            markeredgecolor="white", markeredgewidth=0.5, zorder=4)
    ax.axhline(base_h, ls=(0, (5, 3)), lw=1.1, color=C["grey"], zorder=3)
    ax.text(15.5, base_h - 1.6, "Baseline: Evo2-7B", ha="center", va="top",
            color=C["ink"], fontsize=FONT - 1.5)
    ax.set_ylim(34.0, 102.0)
    ax.set_yticks([40, 60, 80, 100])
    ax.set_xticklabels([])
    ax.set_ylabel(r"Rate of $\alpha$-helix (%)", fontsize=FONT, color=C["ink"])
    ax.yaxis.set_label_coords(-0.125, 0.52)

    ax.text(xlo + 0.5, 100.5, "Max " + r"$\alpha$-helical" + "\ncontent among\nvalid sequences",
            ha="left", va="top", fontsize=FONT - 1, color=C["ink"], linespacing=1.25)
    ax.annotate(f"{peak_h:.1f}%\n({gain:+.1f}%)\n" + r"at $\alpha$ = 8",
                xy=(8.0, peak_h), xytext=(xlo + 0.5, 76.0),
                fontsize=FONT - 1, color=C["green_dark"], va="top", ha="left", linespacing=1.25,
                arrowprops=dict(arrowstyle="->", color=C["green_dark"], lw=1.1,
                                shrinkA=2, shrinkB=3))
    ax.text(20.5, 72.0, "capability collapse\nwith low valid-ORF", ha="center", va="bottom",
            color=C["red"], fontsize=FONT - 1, linespacing=1.15, zorder=6)
    # vertical grey arrow crossing from the helix axis down into the valid-ORF recovery
    ax.annotate("", xy=(22.5, 0.805), xycoords=axo.transData,
                xytext=(22.5, 70.0), textcoords=ax.transData,
                arrowprops=dict(arrowstyle="->", color="#9A9C9F", lw=1.3,
                                shrinkA=2, shrinkB=2), zorder=6)

    # ---- bottom: valid-ORF strip + 3-seed spread band ----
    for yv in (0.7, 0.8, 0.9):
        axo.axhline(yv, color=C["grid"], lw=0.9, zorder=1.5)
    axo.fill_between(x, olo, ohi, color=C["blue"], alpha=0.25, lw=0, zorder=2)
    axo.plot(x, om, "-", color=C["blue"], marker="s", markersize=3.4, linewidth=1.5,
             markeredgecolor="white", markeredgewidth=0.5, zorder=4)
    axo.set_ylim(float(olo.min()) - 0.035, max(float(ohi.max()), float(base_orf)) + 0.03)
    axo.set_yticks([0.7, 0.8, 0.9])
    axo.set_xlabel(r"Steering coefficient  $\alpha$", fontsize=FONT + 1, color=C["ink"],
                   labelpad=2)
    axo.set_ylabel("valid-ORF", fontsize=FONT, color=C["ink"])
    axo.yaxis.set_label_coords(-0.125, 0.5)

    if letter:
        fig.text(0.012, 0.975, "c", fontsize=13, fontweight="bold", va="top", ha="left",
                 color=C["ink"])

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"panel_c.{ext}"))
    plt.close(fig)
    print("wrote paper_figure/panel_c.pdf/.png")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-letter", action="store_true", help="omit the bold panel letter")
    ap.add_argument("--canonical-order", action="store_true",
                    help="pool seeds in ascending order instead of POOL_ORDER; "
                         "order-independent but shifts the CI band by up to 0.16 pp")
    args = ap.parse_args()

    brows, crows = panel_b_data(), panel_c_data(canonical_order=args.canonical_order)
    report(brows, crows)
    print()
    dump_csv(brows, crows)
    plot_panel_b(brows, letter=not args.no_letter)
    plot_panel_c(crows, letter=not args.no_letter)


if __name__ == "__main__":
    main()
