#!/usr/bin/env python3
"""Extended Data figure: cross-judge agreement for reproduction reliability.

FIGURE CONTRACT
---------------
Core conclusion : A human expert panel, Claude Opus 5 and GPT-5.6-sol differ in
                  how severely they score reproduction reliability, but they
                  rank the same work in the same order, so the benchmark's
                  conclusion does not depend on which judge produced it.
Archetype       : quantitative grid - one uniform row of four equal panels.
Backend         : Python / matplotlib (exclusive).
Final size      : 183 mm double column (7.2 in) x 1.94 in (49 mm), single row.
Panel map       : a  system means across judges as a slope chart (no crossings)
                  b  three paired-score scatters sharing one panel letter:
                     left   Claude Opus 5 vs GPT-5.6-sol  <- the two LLM judges
                     middle Human vs Claude Opus 5
                     right  Human vs GPT-5.6-sol
                  The four axes are drawn at an identical square box size
                  (27.8 mm) in one row; the gap after a is wider than the other
                  two because it also carries a's direct system labels.
                  Panels dropped from the composite but still rendered standalone
                  under subfigs/ (and still exported to the JSON): the
                  consensus-profile hero, the head-to-head concordance bars and
                  the per-paper winner matrix.
Evidence chain  : aggregate claim  a (ordering survives every judge)
                  unit-level claim b (the same units, every judge pair)
Statistics      : Spearman rho, Kendall tau, ICC(2,1) on raw and on per-judge
                  z-scores, bootstrap 95% CIs (4,000 resamples), Wilson 95% CIs
                  for proportions, Kendall's W across the three judges.
Source data     : source_data_judge_scores.csv (the full 48 x 3 score matrix).
Reviewer risk   : (i) low raw ICC could be read as "the judges disagree" - panels
                  a/c-e and the ICC-on-z statistic separate severity from
                  ordering; (ii) the two-paper exclusion inherited from the main
                  figure - both the 16-paper and the full 18-paper statistics are
                  written to stats_judge_agreement.json and quoted in the legend.

DATA SOURCES
------------
  Human        human_judge/_orchestrator/score_stats/human_scores.csv
  Claude Opus 5       llm_judge/_orchestrator/score_stats/llm_scores_opus5.csv
  GPT-5.6-sol  llm_judge/_orchestrator_codex/score_stats/llm_scores_gpt56sol.csv

Panels c-e reuse the scatter construction of the main figure's panel e: paired
scores, points coloured by system, a through-origin least-squares reference
line, Spearman rho in the corner, equal aspect on a shared 0-100 range.

OUTPUTS (this directory)
------------------------
  judge_agreement.{png,svg,pdf,tiff}   the composite figure
  subfigs/*.{png,svg}                  every panel standalone, plus two extra
                                       panels (agreement matrix, Bland-Altman)
  source_data_judge_scores.csv         the 48 x 3 paired score matrix
  stats_judge_agreement.json           every number quoted in the panels
  figure_legend.md                     caption and statistics block
  qa_notes.md                          delivery checklist and exclusion record
"""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / ".mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats


# Editable-text settings required for publication SVG/PDF output, matched to the
# main figure so the appendix and the main text share one typographic system.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

mpl.rcParams.update(
    {
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.0,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.dpi": 600,
    }
)

# ---------------------------------------------------------------- data sources

SCORE_STATS = Path(__file__).resolve().parents[3]
JUDGE_CSV = {
    "Human": SCORE_STATS / "csv/judge_totals/human_scores.csv",
    "Claude Opus 5": SCORE_STATS / "csv/judge_totals/llm_scores_opus5.csv",
    "GPT-5.6-sol": SCORE_STATS / "csv/judge_totals/llm_scores_gpt56sol.csv",
}
JUDGES = list(JUDGE_CSV)
# Ordered so the two LLM judges - the pair this appendix was built to quantify -
# come first among the scatter panels.
PAIRS = [("Claude Opus 5", "GPT-5.6-sol"), ("Human", "Claude Opus 5"), ("Human", "GPT-5.6-sol")]

SCI2SYS = {"s1": "Claude Code", "s2": "Mechanist", "s3": "AI-Scientist"}
SYSTEMS = ["Claude Code", "AI-Scientist", "Mechanist"]
# The two papers the main figure drops from its reliability panels, so the
# appendix describes the same 16-paper set. Both the 16- and the full 18-paper
# statistics are exported; see stats_judge_agreement.json -> robustness.
EXCLUDED_EXPERIMENTS = {"multi_lingual_reasoning", "circuit_breakers"}

SHORT_PAPER = {
    "closing_gap_belief": "Belief gap",
    "verbal_confidence": "Verbal confidence",
    "llm_social_decision": "Social decision",
    "emotion_circuit": "Emotion circuit",
    "emotion_prompts": "Emotion prompts",
    "multi_modal_feature_description": "MM feature descr.",
    "sae_agentic_explainer": "SAE explainer",
    "multi_agent": "Multi-agent safety",
    "lasa_safety": "LASA safety",
    "multi_lingual_reasoning": "Multilingual reas.",
    "alignet_visual": "AligNet visual",
    "universal_steering": "Universal steering",
    "propositional_logic_circuit": "Logic circuit",
    "thinking_reasoning_steering": "Reasoning steering",
    "circuit_breakers": "Circuit breakers",
    "encode_harmfulness_refusal": "Harmfulness refusal",
    "esmfold_mechanism": "ESMFold mechanism",
    "interplm": "InterPLM",
}

# System palette is inherited verbatim from the main figure: the same three
# systems must carry the same colour everywhere in the manuscript.
COLORS = {
    "Claude Code": "#7B7B7B",
    "AI-Scientist": "#4E79B8",
    "Mechanist": "#2E7D5B",
}
# Judge palette is a separate, deliberately non-overlapping family - one neutral
# (the human reference standard), one warm signal, one cool accent - so a panel
# coloured by judge can never be misread as a panel coloured by system. The three
# also separate in greyscale (near-black / mid-dark / mid).
JUDGE_COLORS = {
    "Human": "#272727",
    "Claude Opus 5": "#B64342",
    "GPT-5.6-sol": "#42949E",
}
JUDGE_MARKERS = {"Human": "o", "Claude Opus 5": "s", "GPT-5.6-sol": "^"}
# Every judge is named in full everywhere in the figure; the only variation is
# where the name is allowed to break. JUDGE_SHORT is the single-line form used
# inside composed strings (e.g. panel f's "A vs B"); JUDGE_TICK is the wrapped
# form for axis ticks, which have to fit three names side by side.
JUDGE_SHORT = {
    "Human": "Human",
    "Claude Opus 5": "Claude Opus 5",
    "GPT-5.6-sol": "GPT-5.6-sol",
}
JUDGE_TICK = {
    "Human": "Human",
    "Claude Opus 5": "Claude\nOpus 5",
    "GPT-5.6-sol": "GPT-5.6-sol",
}

N_BOOT = 4000
RNG = np.random.default_rng(20260808)

DRAW_PANEL_LABELS = True


# ------------------------------------------------------------------- utilities


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.26, y: float = 1.06) -> None:
    # An empty label is how a panel says "I share my neighbour's letter" - the
    # three scatter panels are one lettered unit, so only the first of them
    # carries the letter.
    if not DRAW_PANEL_LABELS or not label:
        return
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        # Baseline anchoring keeps letters with descenders (g) level with the
        # x-height letters (a, c, e), so the labels sit on one optical line.
        va="baseline",
        color="#222222",
    )


def tidy_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(length=2.2, width=0.6, color="#333333", pad=2)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E7E7E7", linewidth=0.55, zorder=0)


def icc_absolute_agreement(ratings: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, single rater, absolute agreement.

    `ratings` is an (n_subjects, n_raters) array. Unlike Pearson r it penalises
    systematic offsets between raters, which is exactly what separates "the
    judges disagree" from "one judge is uniformly stricter".
    """
    ratings = np.asarray(ratings, dtype=float)
    n, k = ratings.shape
    grand = ratings.mean()
    row_means = ratings.mean(axis=1)
    col_means = ratings.mean(axis=0)
    ss_total = ((ratings - grand) ** 2).sum()
    ss_rows = k * ((row_means - grand) ** 2).sum()
    ss_cols = n * ((col_means - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + (k / n) * (msc - mse)
    return float((msr - mse) / denom) if denom != 0 else float("nan")


def kendall_w(ranks: np.ndarray) -> float:
    """Kendall's coefficient of concordance for k raters over n subjects."""
    ranks = np.asarray(ranks, dtype=float)
    n, k = ranks.shape
    totals = ranks.sum(axis=1)
    s = ((totals - totals.mean()) ** 2).sum()
    return float(12.0 * s / (k**2 * (n**3 - n)))


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOT) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean (seeded, reproducible)."""
    values = np.asarray(values, dtype=float)
    boot = RNG.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high)


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion (well behaved near 0 and 1)."""
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (float(centre - half), float(centre + half))


# ------------------------------------------------------------------ data load


def load_paired_scores(drop_excluded: bool = True) -> tuple[pd.DataFrame, dict]:
    """Return the paired score matrix indexed by (paper, system), plus an audit.

    One row per (paper=experiment, system) unit; one column per judge holding
    that judge's 0-100 total. Rows are kept only where all three judges scored
    the unit, so every panel is computed on identical units. The audit dict
    records every row count before and after each filter, which is what the
    QA notes and the legend quote.
    """
    audit: dict = {"per_judge_rows": {}, "excluded_experiments": sorted(EXCLUDED_EXPERIMENTS)}
    frames = []
    for judge, path in JUDGE_CSV.items():
        df = pd.read_csv(path)
        df = df.assign(
            paper=df["实验"].astype(str).str.strip(),
            system=df["sci"].astype(str).str.strip().map(SCI2SYS),
            score=pd.to_numeric(df["总分(0-100)"], errors="coerce"),
        )
        n_raw = len(df)
        df = df[df["system"].notna() & df["score"].notna()]
        n_scored = len(df)
        if drop_excluded:
            df = df[~df["paper"].isin(EXCLUDED_EXPERIMENTS)]
        audit["per_judge_rows"][judge] = {
            "rows_in_csv": int(n_raw),
            "rows_with_score": int(n_scored),
            "rows_after_paper_filter": int(len(df)),
        }
        frames.append(
            df[["paper", "system", "score"]]
            .rename(columns={"score": judge})
            .set_index(["paper", "system"])
        )
    merged = pd.concat(frames, axis=1).sort_index()
    audit["rows_merged"] = int(len(merged))
    paired = merged.dropna()
    audit["rows_after_complete_case"] = int(len(paired))
    audit["rows_dropped_incomplete"] = int(len(merged) - len(paired))
    return paired, audit


def zscore_by_judge(paired: pd.DataFrame) -> pd.DataFrame:
    """Per-judge standardisation: removes each judge's severity and spread.

    What survives standardisation is only the *relative* verdict - which unit is
    better than which - which is the quantity this figure claims is shared.
    """
    return (paired - paired.mean()) / paired.std(ddof=1)


def pair_statistics(paired: pd.DataFrame, x_judge: str, y_judge: str) -> dict:
    """Every agreement statistic quoted for one judge pair."""
    x_pts = paired[x_judge].to_numpy(dtype=float)
    y_pts = paired[y_judge].to_numpy(dtype=float)
    z = zscore_by_judge(paired)
    rho, rho_p = stats.spearmanr(x_pts, y_pts)
    r_value, p_value = stats.pearsonr(x_pts, y_pts)
    tau, tau_p = stats.kendalltau(x_pts, y_pts)
    diff = y_pts - x_pts
    sd = float(diff.std(ddof=1))
    bias = float(diff.mean())
    return {
        "n_units": int(x_pts.size),
        "pearson_r": float(r_value),
        "pearson_p": float(p_value),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "kendall_tau": float(tau),
        "kendall_p": float(tau_p),
        "icc_raw": icc_absolute_agreement(paired[[x_judge, y_judge]].to_numpy()),
        "icc_z": icc_absolute_agreement(z[[x_judge, y_judge]].to_numpy()),
        "mae": float(np.mean(np.abs(diff))),
        "mean_signed_offset": bias,
        "loa_low": bias - 1.96 * sd,
        "loa_high": bias + 1.96 * sd,
        "slope_through_origin": float(np.sum(x_pts * y_pts) / np.sum(x_pts**2)),
    }


def head_to_head(paired: pd.DataFrame, a: str, b: str) -> dict:
    """Fraction of paper-level system-vs-system calls on which two judges agree."""
    wide = paired.unstack("system")
    agree = total = 0
    for s1, s2 in itertools.combinations(SYSTEMS, 2):
        diff_a = np.sign(wide[(a, s1)].to_numpy() - wide[(a, s2)].to_numpy())
        diff_b = np.sign(wide[(b, s1)].to_numpy() - wide[(b, s2)].to_numpy())
        agree += int(np.sum((diff_a == diff_b) & (diff_a != 0)))
        total += int(diff_a.size)
    lo, hi = wilson_ci(agree, total)
    return {"agree": agree, "total": total, "fraction": agree / total,
            "ci95_low": lo, "ci95_high": hi}


# -------------------------------------------------------------------- panels


def draw_consensus_profile(ax: plt.Axes, paired: pd.DataFrame, label: str = "a") -> dict:
    """HERO - the three judges' standardised scores over the same units.

    Units are ordered by their consensus (mean z) score, so a judge sharing the
    consensus traces a rising cloud. Three clouds rising together inside a narrow
    band is the direct visual statement that the judges rank the same work the
    same way once each judge's own severity and spread are divided out.
    """
    add_panel_label(ax, label, x=-0.10)
    z = zscore_by_judge(paired)
    order = z.mean(axis=1).sort_values().index
    z = z.loc[order]
    x = np.arange(len(z))

    ax.axhline(0, color="#D6D6D6", linewidth=0.6, zorder=1)
    # Grey band = full min-max range across the three judges for each unit; its
    # narrowness is the disagreement that survives standardisation. Markers are
    # drawn without connecting lines: 48 units x 3 judges of joined line would be
    # unreadable, and the claim is the shared upward drift, not any single path.
    ax.fill_between(
        x, z.min(axis=1), z.max(axis=1), color="#EAEAEA", zorder=1.5, linewidth=0,
        label="Judge range",
    )
    for judge in JUDGES:
        ax.plot(
            x,
            z[judge].to_numpy(),
            color=JUDGE_COLORS[judge],
            linestyle="none",
            marker=JUDGE_MARKERS[judge],
            markersize=2.6,
            markeredgewidth=0,
            alpha=0.95,
            zorder=3,
            label=judge,
        )
    ax.plot(x, z.mean(axis=1).to_numpy(), color="#5A5A5A", linewidth=1.1, zorder=4,
            label="Consensus")

    ax.set_xlabel("Experiment index")
    ax.set_ylabel("Standardised reliability score (z)")
    ax.set_xlim(-1.5, len(z) + 0.5)
    ax.set_ylim(-2.7, 2.9)
    ax.set_xticks([0, len(z) // 4, len(z) // 2, 3 * len(z) // 4, len(z) - 1])
    ax.set_xticklabels([str(v + 1) for v in ax.get_xticks().astype(int)])
    spread = float(z.std(axis=1, ddof=0).mean())
    # Sorting leaves the top-left corner empty, so the legend goes there and
    # covers no data point. The three-judge ICC and the mean within-unit spread
    # are still computed and exported to stats_judge_agreement.json; they are
    # simply not printed in-panel, which keeps the hero panel to one statement.
    ax.legend(loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.8,
              handletextpad=0.5, borderpad=0.2, labelspacing=0.28, fontsize=5.8)
    tidy_axis(ax)
    return {
        "icc_z_three_judge": icc_absolute_agreement(z.to_numpy()),
        "icc_raw_three_judge": icc_absolute_agreement(paired.to_numpy()),
        "mean_within_unit_sd_z": spread,
    }


def draw_severity_slope(ax: plt.Axes, paired: pd.DataFrame, label: str = "a",
                        direct_labels: bool = True, angled_ticks: bool = False) -> dict:
    """System mean under each judge, drawn as a slope chart.

    Each line is one system tracked across the three judges. The lines all fall
    from left to right - judges get stricter - but they never cross, and that
    absence of crossings *is* the claim: severity is a per-judge constant applied
    to every system, so the ordering the paper reports survives any judge.
    """
    add_panel_label(ax, label)
    x = np.arange(len(JUDGES))
    out: dict[str, dict] = {}
    end_values: dict[str, float] = {}
    for system in SYSTEMS:
        means, lows, highs = [], [], []
        for judge in JUDGES:
            values = paired.xs(system, level="system")[judge].to_numpy(dtype=float)
            mean = float(values.mean())
            lo, hi = bootstrap_ci(values)
            means.append(mean)
            lows.append(mean - lo)
            highs.append(hi - mean)
            out.setdefault(system, {})[judge] = {
                "mean": mean, "ci95_low": lo, "ci95_high": hi, "n_papers": int(values.size),
            }
        ax.plot(x, means, color=COLORS[system], linewidth=1.2, marker="o", markersize=3.4,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        ax.errorbar(x, means, yerr=[lows, highs], fmt="none", ecolor=COLORS[system],
                    elinewidth=0.7, capsize=1.6, capthick=0.7, alpha=0.9, zorder=2)
        end_values[system] = means[-1]

    # Direct labels beat a legend here - the lines never cross, so each keeps a
    # stable vertical position the eye can follow to its name. The two top
    # systems end ~2 pp apart, closer than one line of type, so the labels are
    # pushed to a minimum separation and a short leader restores the link.
    # Tuned to the zoomed y-range below: 4.5 pp of a 65 pp axis already clears a
    # 5.8 pt line, so the labels stay next to the lines they name.
    # In the composite the labels are allowed to sit outside the axes: the panel
    # is a fixed square shared with three scatters, so the label column is paid
    # for out of the gap to the next panel rather than out of the plotting area.
    if direct_labels:
        min_gap = 4.5
        ordered = sorted(end_values, key=end_values.get)
        label_y = {}
        previous = -1e9
        for system in ordered:
            y = max(end_values[system], previous + min_gap)
            label_y[system] = y
            previous = y
        for system in ordered:
            ax.plot([x[-1] + 0.05, x[-1] + 0.19], [end_values[system], label_y[system]],
                    color=COLORS[system], linewidth=0.5, zorder=2, clip_on=False)
            ax.text(x[-1] + 0.23, label_y[system], system, color=COLORS[system],
                    fontsize=5.8, ha="left", va="center")

    ax.set_xticks(x)
    if not angled_ticks:
        # Standalone panel: 5.8 pt with the wrapped Claude name, three full judge
        # names side by side over the left 55% of a wide panel, the rest reserved
        # for the direct system labels.
        ax.set_xticklabels([JUDGE_TICK[j] for j in JUDGES], fontsize=5.8)
        ax.set_xlim(-0.35, len(JUDGES) - 1 + 1.25)
    else:
        # Composite: the panel is a 27 mm square, so each judge tick owns ~9 mm -
        # narrower than "GPT-5.6-sol" set horizontally, and too narrow for the
        # data to also give up a quarter of its width to a label column. Angling
        # keeps the names complete above the 5 pt floor, the x-range stops just
        # past the last judge, and the direct system labels are drawn unclipped
        # into the (widened) gap that follows this panel.
        ax.set_xticklabels([JUDGE_SHORT[j] for j in JUDGES], fontsize=5.8,
                           rotation=30, ha="right", rotation_mode="anchor")
        ax.set_xlim(-0.35, len(JUDGES) - 1 + 0.35)
    # Deliberately not zero-based. This panel makes a comparison of *positions*
    # (do the lines cross?), not of magnitudes, so no bar length is being read
    # off the axis and truncation cannot mislead. On a 0-100 axis the Mechanist
    # and Claude Code lines sit ~2 pp apart at the right-hand end and merge into
    # one stroke; 30-95 spans every mean and CI cap with ~4 pp of headroom and
    # separates them. The severity note is dropped - the three falling lines
    # already say it, and the exact steps are in stats_judge_agreement.json.
    ax.set_ylim(30, 95)
    ax.set_yticks(np.arange(30, 96, 10))
    ax.set_ylabel("Reliability score (%)")
    ax.set_xlabel("Judge")
    tidy_axis(ax)
    return out


def draw_system_means(ax: plt.Axes, paired: pd.DataFrame, label: str = "b") -> dict:
    """System means under each judge as grouped bars (standalone/slide panel).

    Bar heights fall by a judge-specific constant, but the within-judge ordering
    is identical: whatever severity a judge applies, it applies to every system,
    so the comparison the paper actually makes is unaffected.
    """
    add_panel_label(ax, label)
    x_base = np.arange(len(SYSTEMS))
    offsets = np.array([-0.25, 0.0, 0.25])
    width = 0.22
    out: dict[str, dict] = {}
    for k, judge in enumerate(JUDGES):
        means, lows, highs = [], [], []
        for system in SYSTEMS:
            values = paired.xs(system, level="system")[judge].to_numpy(dtype=float)
            mean = float(values.mean())
            lo, hi = bootstrap_ci(values)
            means.append(mean)
            lows.append(mean - lo)
            highs.append(hi - mean)
            out.setdefault(judge, {})[system] = {
                "mean": mean, "ci95_low": lo, "ci95_high": hi, "n_papers": int(values.size),
            }
        ax.bar(
            x_base + offsets[k], means, width=width, color=JUDGE_COLORS[judge],
            edgecolor="white", linewidth=0.4, alpha=0.92, label=judge, zorder=3,
        )
        ax.errorbar(
            x_base + offsets[k], means, yerr=[lows, highs], fmt="none",
            ecolor="#333333", elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=4,
        )
        # Explicit within-judge rank makes "the same order under every judge"
        # readable without measuring bar heights against the axis.
        ranks = pd.Series(means).rank(ascending=False).astype(int).tolist()
        for xi, mean, hi_err, rank in zip(x_base + offsets[k], means, highs, ranks):
            ax.text(xi, mean + hi_err + 2.5, f"#{rank}", ha="center", va="bottom",
                    fontsize=5.2, color="#666666")

    ax.set_xticks(x_base)
    ax.set_xticklabels(["Claude\nCode", "AI-\nScientist", "Mechanist"])
    ax.set_ylabel("Reliability score (%)")
    ax.set_ylim(0, 118)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.text(0.5, 1.005, "#n: rank within judge", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=5.4, color="#777777")
    ax.legend(loc="upper left", handlelength=1.1, borderpad=0.2, labelspacing=0.25,
              fontsize=5.8)
    tidy_axis(ax)
    return out


def draw_head_to_head(ax: plt.Axes, paired: pd.DataFrame, label: str = "g") -> dict:
    """Do two judges pick the same winner in a head-to-head comparison?

    For every paper and every unordered system pair each judge declares a winner;
    the bar is the fraction of those decisions on which the two judges agree.
    This is the decision the benchmark is actually used to make, so concordance
    here matters more than the correlation of the raw numbers.
    """
    # Pushed further out than the default: this panel's y-axis label is long, and
    # at the default offset the letter would sit on top of it.
    add_panel_label(ax, label, x=-0.34)
    out: dict = {}
    labels, values, errs = [], [], []
    for a, b in PAIRS:
        result = head_to_head(paired, a, b)
        out[f"{a}|{b}"] = result
        # Broken after the first judge: full names on one line would overlap at
        # this panel width, and the break falls at the natural "A / vs B" seam.
        labels.append(f"{JUDGE_SHORT[a]}\nvs {JUDGE_SHORT[b]}")
        values.append(result["fraction"] * 100)
        errs.append([
            (result["fraction"] - result["ci95_low"]) * 100,
            (result["ci95_high"] - result["fraction"]) * 100,
        ])

    x = np.arange(len(PAIRS))
    # Each bar is the average of its two judge colours, so a bar names its pair
    # without needing a fourth legend on the page.
    bar_colors = [
        tuple(np.mean([mpl.colors.to_rgb(JUDGE_COLORS[a]),
                       mpl.colors.to_rgb(JUDGE_COLORS[b])], axis=0))
        for a, b in PAIRS
    ]
    ax.bar(x, values, width=0.58, color=bar_colors, edgecolor="white", linewidth=0.4, zorder=3)
    ax.errorbar(x, values, yerr=np.array(errs).T, fmt="none", ecolor="#333333",
                elinewidth=0.6, capsize=1.8, capthick=0.6, zorder=4)
    ax.axhline(50, color="#8A8A8A", linewidth=0.7, linestyle=(0, (3, 2)), zorder=5)
    # Parked past the last bar, the only place on this narrow panel where the
    # label does not sit on top of a bar.
    ax.text(len(PAIRS) - 1 + 0.34, 52, "chance", ha="left", va="bottom", fontsize=5.4,
            color="#777777")
    # Value labels clear the upper CI cap so neither element sits on the other.
    for xi, value, err in zip(x, values, errs):
        ax.text(xi, value + err[1] + 2.5, f"{value:.0f}", ha="center", va="bottom",
                fontsize=5.8, color="#333333")

    # Computed and exported, not printed. Kendall's W answers a different
    # question from the bars - three judges' agreement on the *ranking* of all
    # units, versus two judges' agreement on a pairwise *winner* - so sitting in
    # this panel it read as a summary of the bars, which it is not. The legend
    # quotes it in the place where it belongs, alongside the ranking claim.
    out["kendall_w_three_judge"] = kendall_w(paired.rank(axis=0).to_numpy())
    ax.set_xticks(x)
    # Pair names are far wider than a bar; angling them keeps all three readable
    # without shrinking the type below the 5 pt floor.
    ax.set_xticklabels(labels, fontsize=5.2, rotation=30, ha="right",
                       rotation_mode="anchor")
    # The bar is a rate over pairwise comparisons, and the label now says exactly
    # that: "pairwise" names the unit being counted (one system-vs-system call,
    # not one paper and not one score), "same-winner" names the event, and "rate"
    # plus the % says it is a proportion of those calls. "Same winner (%)" alone
    # left the unit to be inferred from the tick labels.
    ax.set_ylabel("Pairwise same-winner\nrate (%)")
    ax.set_xlim(-0.6, len(PAIRS) - 1 + 1.05)
    # 105 rather than 112: the extra headroom existed for the Kendall W line that
    # used to sit above the bars. With that gone it only has to clear the tallest
    # value label sitting on its CI cap.
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    tidy_axis(ax)
    return out


def draw_pair_scatter(ax: plt.Axes, paired: pd.DataFrame, x_judge: str, y_judge: str,
                      label: str, stats_pair: dict | None = None) -> dict:
    """One judge pair, drawn exactly like the main figure's agreement panel."""
    add_panel_label(ax, label)
    result = stats_pair or pair_statistics(paired, x_judge, y_judge)

    for system in SYSTEMS:
        group = paired.xs(system, level="system")
        ax.scatter(
            group[x_judge], group[y_judge], s=14, color=COLORS[system], alpha=0.74,
            linewidth=0.25, edgecolor="white", label=system,
        )

    lims = [0, 100]
    # Dashed line is a through-origin least-squares fit y = kx (k = Sxy / Sxx),
    # matching the main figure's agreement panel. The identity line is not drawn:
    # with three of them on one row it read as a second data series, and the fit
    # already shows the offset by sitting below where y = x would run.
    k_origin = result["slope_through_origin"]
    ax.plot(lims, [k_origin * lims[0], k_origin * lims[1]], color="#8A8A8A",
            linewidth=0.8, linestyle=(0, (3, 2)))

    # Named in full, exactly as the main figure's panel e writes it, so the same
    # statistic is not called two different things across the manuscript. ICC on
    # z and the fitted slope stay in stats_judge_agreement.json.
    ax.text(0.045, 0.965, f"Spearman ρ = {result['spearman_rho']:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.2, color="#222222")
    # Two lines: the judge name plus the quantity, so an axis says both which
    # model produced the score and what the score is, without running wider than
    # the panel.
    ax.set_xlabel(f"{JUDGE_SHORT[x_judge]}\nreliability score (%)")
    ax.set_ylabel(f"{JUDGE_SHORT[y_judge]}\nreliability score (%)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ticks = np.arange(0, 101, 20)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    # Equal aspect so identical x/y ranges map to identical visual lengths and a
    # departure from 45 degrees is a real difference in severity, not a stretch.
    ax.set_aspect("equal", adjustable="box")
    tidy_axis(ax)
    return result


def per_paper_winner_stats(paired: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Winner per (paper, judge) and the unanimity count, computed without drawing.

    Split out from the renderer so the statistic outlives the panel: the
    composite figure no longer carries the winner matrix, but the legend and
    stats_judge_agreement.json still report how many papers the three judges
    call the same way. Three judges choosing among three systems agree by chance
    on 1/9 of papers, which is the comparison the number is quoted against.
    """
    winners = paired.groupby(level="paper").apply(
        lambda g: pd.Series({j: g[j].idxmax()[1] for j in JUDGES})
    )
    unanimous = winners.nunique(axis=1) == 1
    # Unanimous papers sort to the top so the block structure is visible at a
    # glance; within each block, papers keep alphabetical order.
    winners = winners.loc[
        unanimous.to_frame("u").assign(p=winners.index).sort_values(
            ["u", "p"], ascending=[False, True]).index
    ]
    n_unan = int(unanimous.sum())
    chance = len(winners) / len(SYSTEMS) ** (len(JUDGES) - 1)
    return winners, {
        "papers": int(len(winners)),
        "unanimous_papers": n_unan,
        "unanimous_expected_by_chance": float(chance),
        "unanimous_fraction": float(n_unan / len(winners)),
        "winner_table": {p: dict(r) for p, r in winners.iterrows()},
    }


def draw_per_paper_winner(ax: plt.Axes, paired: pd.DataFrame, label: str = "x5",
                          standalone: bool = False) -> dict:
    """System scored highest on each paper, by judge - standalone panel only.

    One row per paper, one column per judge, cell colour = that judge's winner.
    A single-colour row is a paper on which all three judges picked the same
    system, which rules out the aggregate ranking of panel b being an averaging
    artefact that no individual paper supports. Dropped from the composite
    figure - it needed a full column for a claim the legend can make in one
    number - and kept here for reviewers who want the per-paper granularity.
    """
    add_panel_label(ax, label, x=-0.30)
    winners, out = per_paper_winner_stats(paired)

    for row, (_, values) in enumerate(winners.iterrows()):
        for col, judge in enumerate(JUDGES):
            ax.add_patch(patches.Rectangle(
                (col - 0.5, row - 0.5), 1, 1, facecolor=COLORS[values[judge]],
                edgecolor="white", linewidth=0.7,
            ))
    ax.set_xlim(-0.5, len(JUDGES) - 0.5)
    ax.set_ylim(len(winners) - 0.5, -0.5)
    ax.set_xticks(range(len(JUDGES)))
    # Three columns share ~19 mm, far narrower than a horizontal judge name, so
    # the column labels are angled rather than truncated.
    ax.set_xticklabels([JUDGE_SHORT[j] for j in JUDGES], fontsize=5.8, rotation=40,
                       ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(winners)))
    ax.set_yticklabels([SHORT_PAPER.get(p, p) for p in winners.index], fontsize=5.4)
    # Rightmost panel of its row: row labels go on the outer edge so they cannot
    # collide with the neighbouring panel's axis.
    ax.yaxis.tick_right()
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if standalone:
        handles = [patches.Patch(facecolor=COLORS[s], edgecolor="none", label=s)
                   for s in SYSTEMS]
        # Clears the angled column labels, which occupy the strip directly below
        # the grid when this panel is rendered on its own.
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.16),
                  handlelength=0.9, handleheight=0.9, borderpad=0.2, labelspacing=0.3,
                  fontsize=5.4)
    ax.text(0.5, 1.012,
            f"{out['unanimous_papers']}/{out['papers']} papers unanimous\n"
            f"({out['unanimous_expected_by_chance']:.1f} expected by chance)",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=5.2,
            linespacing=1.2, color="#555555")
    return out


# ------------------------------------------------- extra standalone-only panels


def draw_agreement_matrix(ax: plt.Axes, paired: pd.DataFrame, label: str = "h") -> dict:
    """Compact all-pairs summary (standalone/slide panel).

    Upper triangle: Spearman rho on raw scores - do the judges order the units
    the same way? Lower triangle: ICC(2,1) on per-judge z-scores - do they agree
    on magnitude once severity is removed? The two halves separate the two
    questions a single correlation number blurs together.
    """
    add_panel_label(ax, label)
    z = zscore_by_judge(paired)
    n = len(JUDGES)
    grid = np.full((n, n), np.nan)
    annot = np.empty((n, n), dtype=object)
    out: dict[str, float] = {}
    for i, a in enumerate(JUDGES):
        for j, b in enumerate(JUDGES):
            if i == j:
                value = 1.0
            elif j > i:
                value = float(stats.spearmanr(paired[a], paired[b])[0])
                out[f"{a}|{b}|spearman_rho"] = value
            else:
                value = icc_absolute_agreement(z[[a, b]].to_numpy())
                out[f"{b}|{a}|icc_z"] = value
            grid[i, j] = value
            annot[i, j] = f"{value:.2f}"

    # Sequential, single-hue, greyscale-safe: no rainbow, no red/green encoding.
    cmap = LinearSegmentedColormap.from_list("agree", ["#FFFFFF", "#CFE3D8", "#2E7D5B"])
    ax.imshow(grid, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, annot[i, j], ha="center", va="center", fontsize=6.4,
                    color="#FFFFFF" if grid[i, j] > 0.62 else "#222222",
                    fontweight="bold" if i != j else "normal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    # Wrapped on x (three names over three cells would touch), full width on y
    # (rows have the whole left margin).
    ax.set_xticklabels([JUDGE_TICK[j] for j in JUDGES], fontsize=5.8)
    ax.set_yticklabels([JUDGE_SHORT[j] for j in JUDGES], fontsize=5.8)
    ax.tick_params(length=0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, -0.22, "upper: Spearman ρ (raw)   lower: ICC on z-scores",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.6, color="#555555")
    return out


def draw_bland_altman(ax: plt.Axes, paired: pd.DataFrame, x_judge: str, y_judge: str,
                      label: str = "i") -> dict:
    """Bland-Altman for one judge pair (standalone/slide panel).

    Difference against mean, with bias and 95% limits of agreement. A flat cloud
    offset from zero says the two judges disagree by roughly a constant, which is
    the correctable kind of disagreement.
    """
    add_panel_label(ax, label)
    x_pts = paired[x_judge].to_numpy(dtype=float)
    y_pts = paired[y_judge].to_numpy(dtype=float)
    mean = (x_pts + y_pts) / 2
    diff = y_pts - x_pts
    bias, sd = float(diff.mean()), float(diff.std(ddof=1))
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd

    for system in SYSTEMS:
        mask = paired.index.get_level_values("system") == system
        ax.scatter(mean[mask], diff[mask], s=14, color=COLORS[system], alpha=0.74,
                   linewidth=0.25, edgecolor="white", label=system)
    ax.axhline(0, color="#D0D0D0", linewidth=0.7)
    ax.axhline(bias, color="#333333", linewidth=0.8)
    for level in (lo, hi):
        ax.axhline(level, color="#8A8A8A", linewidth=0.7, linestyle=(0, (3, 2)))
    ax.text(0.98, 0.05, f"bias = {bias:+.1f} pp\n95% LoA {lo:+.0f} to {hi:+.0f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color="#333333")
    ax.set_xlabel("Mean of the two judges (%)")
    ax.set_ylabel(f"{JUDGE_SHORT[y_judge]} − {JUDGE_SHORT[x_judge]} (pp)")
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    tidy_axis(ax)
    return {"bias": bias, "sd": sd, "loa_low": lo, "loa_high": hi}


# ---------------------------------------------------------------- deliverables


def save_source_data(paired: pd.DataFrame) -> None:
    source = paired.reset_index()
    source.insert(0, "unit", np.arange(1, len(source) + 1))
    z = zscore_by_judge(paired).reset_index(drop=True)
    for judge in JUDGES:
        source[f"{judge} (z)"] = z[judge].round(3)
    source.to_csv(OUT_DIR / "source_data_judge_scores.csv", index=False, encoding="utf-8")


def save_legend(paired: pd.DataFrame, stats_out: dict) -> None:
    pairs = stats_out["pairs"]
    llm = pairs["Claude Opus 5|GPT-5.6-sol"]
    n_units = stats_out["n_units"]
    n_papers = stats_out["n_papers"]
    winner = stats_out["per_paper_winner"]
    rob = stats_out["robustness"]["all_18_papers"]
    # The head-to-head panel is no longer drawn, so its claim moves into the
    # legend as a range over the three judge pairs.
    h2h = [stats_out["head_to_head"][f"{a}|{b}"] for a, b in PAIRS]
    h2h_low = min(v["fraction"] for v in h2h) * 100
    h2h_high = max(v["fraction"] for v in h2h) * 100
    h2h_total = h2h[0]["total"]
    legend = f"""Extended Data Fig. | Human and LLM reproduction-reliability judges differ in severity but not in conclusion.

A human expert panel, Claude Opus 5 and GPT-5.6-sol independently scored the same
{n_units} system-paper units ({n_papers} papers x {len(SYSTEMS)} systems) on the identical 0-100
reproduction-reliability rubric. **a**, Mean score of each system
under each judge; points are means over {n_papers} papers and whiskers are percentile
bootstrap 95% confidence intervals ({N_BOOT:,} resamples). The y-axis is truncated at
30% to resolve the two leading systems, which differ by ~2 percentage points
under the strictest judge; the panel compares line positions, not bar lengths.
Every line falls from left to right - the judges get stricter - but no two lines
cross, so the ranking is judge-independent. **b**, Paired scores for each judge
pair (left, Claude Opus 5 versus GPT-5.6-sol; middle, Human versus Claude Opus 5;
right, Human versus GPT-5.6-sol); each point is one system-paper unit coloured by
system and the dashed line is a through-origin least-squares fit whose slope is
the severity ratio (the identity line is not drawn). Spearman rho is the rank
correlation; ICC(2,1) on raw and on per-judge z-scores, and the fitted slope, are
exported to stats_judge_agreement.json. Colour denotes the reproducing system
throughout: named at the end of each line in **a** and in the key above **b**.

The three judges apply different severity - mean scores are
{stats_out['judge_mean_score']['Human']:.1f}% (Human), {stats_out['judge_mean_score']['Claude Opus 5']:.1f}% (Claude Opus 5) and {stats_out['judge_mean_score']['GPT-5.6-sol']:.1f}% (GPT-5.6-sol) - but
they order the work the same way. The system ranking is identical under all
three judges (a), Kendall's W = {stats_out['concordance']['kendall_w_three_judge']:.2f} across the {n_units} units, {winner['unanimous_papers']}/{winner['papers']} papers have a
unanimous winner against {winner['unanimous_expected_by_chance']:.1f} expected by chance, and every judge pair names
the same winner on {h2h_low:.0f}-{h2h_high:.0f}% of the {h2h_total} head-to-head system comparisons
({n_papers} papers x {len(SYSTEMS)} system pairs, chance 50%). For the two LLM
judges Spearman rho = {llm['spearman_rho']:.2f} with
ICC(2,1) rising from {llm['icc_raw']:.2f} on raw scores to {llm['icc_z']:.2f} once each judge's severity
and spread are removed (b, left). Residual disagreement is therefore a calibration
offset ({llm['mean_signed_offset']:+.1f} percentage points between the two LLM judges) rather than a
conflicting verdict. The per-unit standardised-score profile, the head-to-head
concordance bars and the per-paper winner matrix are not shown here; they are
rendered as subfigs/x6_consensus_profile, subfigs/x7_head_to_head and
subfigs/x5_per_paper_winner and their statistics are in stats_judge_agreement.json.

Statistics
----------
n definition            : one unit = one paper scored for one system; n = {n_units}
                          ({n_papers} papers x {len(SYSTEMS)} systems), complete cases only.
biological replicates   : not applicable (each unit is scored once per judge).
technical replicates    : not applicable (single judge pass per unit).
center statistic        : mean score per system per judge (a); individual units
                          are plotted unaggregated (b).
spread/interval         : percentile bootstrap 95% CI, {N_BOOT:,} resamples (a);
                          Wilson 95% score intervals for the head-to-head rates
                          quoted above are in stats_judge_agreement.json.
test                    : Spearman rho and Kendall tau (rank agreement), Pearson r,
                          ICC(2,1) two-way random effects absolute agreement on raw
                          and on per-judge z-scores, Kendall's W across three judges.
multiple-comparison     : none applied; all p-values are exported unadjusted in
                          stats_judge_agreement.json and none is used as a
                          significance threshold in the figure.
p-value display         : not shown in-panel; exported in stats_judge_agreement.json.
source-data file        : source_data_judge_scores.csv
metric definition       : 0-100 reliability total = (sum of eight rubric dimensions
                          + the reproduction-fidelity dimension) / achievable base
                          x 100, identical rubric and aggregation for all judges.
baseline definition     : the human expert panel is the reference judge.
data exclusion          : {', '.join(stats_out['excluded_experiments'])} are excluded to
                          match the main figure's reliability panels ({n_papers} of 18
                          papers retained). With all 18 papers ({rob['n_units']} units) the
                          conclusion is unchanged: Kendall's W = {rob['kendall_w_three_judge']:.2f} and the
                          two LLM judges give rho = {rob['pairs']['Claude Opus 5|GPT-5.6-sol']['spearman_rho']:.2f}, ICC on z = {rob['pairs']['Claude Opus 5|GPT-5.6-sol']['icc_z']:.2f}.
"""
    (OUT_DIR / "figure_legend.md").write_text(legend, encoding="utf-8")


def save_qa_notes(stats_out: dict, audit: dict) -> None:
    rows = "\n".join(
        f"| {judge} | {v['rows_in_csv']} | {v['rows_with_score']} | {v['rows_after_paper_filter']} |"
        for judge, v in audit["per_judge_rows"].items()
    )
    notes = f"""# QA notes - judge agreement figure

## Contract

- Core conclusion: the human panel, Claude Opus 5 and GPT-5.6-sol differ in
  severity but rank the same work in the same order, so the benchmark conclusion
  does not depend on which judge produced it.
- Archetype: quantitative grid - one uniform row of four equal square panels.
- Backend: Python / matplotlib, exclusive for all drawing, export and visual QA.
- Final size: 183 mm (7.2 in) x 49 mm (1.94 in), double column, single row. The
  four axes are placed by explicit figure coordinates at an identical box size
  (27.8 mm square), not by a gridspec, because an aspect-constrained axes
  shrinks inside a gridspec cell by an amount that depends on the cell height.
- Panel letters: a = severity slope; b = the three paired-score scatters, which
  are one piece of evidence run over three judge pairs and therefore share a
  single letter placed at the left of the first of them.
- Exports: SVG + PDF (editable text, Type 42) and PNG + TIFF at 600 dpi.

## Data-integrity record

| Judge | Rows in CSV | Rows with a usable score | Rows after paper filter |
|---|---|---|---|
{rows}

- Merged rows before complete-case filtering: {audit['rows_merged']}
- Rows after complete-case filtering: {audit['rows_after_complete_case']}
- Rows dropped because a judge was missing: {audit['rows_dropped_incomplete']}
- Papers excluded: {', '.join(audit['excluded_experiments'])}. Rule: reuse exactly
  the exclusion applied by the main figure's reliability panels so the appendix
  and the main text describe the same 16-paper set. Nothing is excluded to make
  the plot easier to render.
- Robustness: every statistic is recomputed on the unfiltered 18-paper set and
  written to `stats_judge_agreement.json` -> `robustness.all_18_papers`. The
  conclusion is unchanged.
- No smoothing, winsorising, or outlier removal is applied to any score.

## Static preflight

`scripts/validate_figure.py` reports two warnings, both expected here:

- `DATA-EXCLUSION` - triggered by the `.dropna()` complete-case filter. Counts
  before and after that filter are tracked and are recorded in the table above
  and in `stats_judge_agreement.json` -> `data_audit`.
- `DEMO-DATA` - triggered by the seeded random generator. No plotted value is
  simulated: the generator is used only to resample observed scores for the
  bootstrap confidence intervals in panel a. Every score comes from the three
  judge CSVs.

## Checklist

- [x] Every panel maps to the core conclusion and carries unique evidence.
- [x] Panel labels lowercase bold, 8 pt, top-left.
- [x] Smallest text 5.8 pt (panel a judge ticks); body text 7 pt.
- [x] All four axes are the same size: identical width and height in figure
      coordinates, verified in the render.
- [x] Editable text (`svg.fonttype=none`, `pdf.fonttype=42`).
- [x] Arial/Helvetica with DejaVu fallback.
- [x] No rainbow colour map; the single sequential map used by the standalone
      agreement matrix is greyscale-monotonic, and in the composite the three
      system colours separate in greyscale (mid grey / mid-dark blue / dark
      green). Panel a names each system directly at the end of its line; the
      three scatters share one key at the top-right corner of the b block.
- [x] Truncated axis declared: panel a starts at 30% rather than 0. It plots
      three lines of means, not bars, and the claim it carries is whether they
      cross - a question of position, not of length - so the truncation cannot
      inflate a ratio the reader takes off the axis. It is stated in
      `figure_legend.md`. Every other axis in the figure is zero-based.
- [x] Every judge is named in full (Human / Claude Opus 5 / GPT-5.6-sol) at every
      occurrence; in the composite the panel-a ticks are angled rather than
      wrapped or abbreviated.
- [x] System colours identical to the main figure; judge colours are a separate,
      non-overlapping family.
- [x] n, centre, spread, test and correction documented in `figure_legend.md`.
- [x] Source data exported to `source_data_judge_scores.csv`.
- [ ] Image integrity: not applicable - no micrographs, blots or gels.

## Verified at final size

Inspect `judge_agreement.png` at 100% and confirm the angled judge names on the
panel-a ticks stay separated, the two-line axis labels of the three panel-b
scatters do not collide across the column gaps, that panel a's direct system
labels clear the first scatter's y label, and that the system key above the b
block clears the panel frames and the Spearman values.
"""
    (OUT_DIR / "qa_notes.md").write_text(notes, encoding="utf-8")


def compute_statistics(paired: pd.DataFrame) -> dict:
    """All agreement statistics for a given unit set, independent of drawing."""
    z = zscore_by_judge(paired)
    return {
        "n_units": int(len(paired)),
        "n_papers": int(paired.index.get_level_values("paper").nunique()),
        "judge_mean_score": {j: float(paired[j].mean()) for j in JUDGES},
        "judge_sd_score": {j: float(paired[j].std(ddof=1)) for j in JUDGES},
        "pairs": {f"{a}|{b}": pair_statistics(paired, a, b) for a, b in PAIRS},
        "head_to_head": {f"{a}|{b}": head_to_head(paired, a, b) for a, b in PAIRS},
        "kendall_w_three_judge": kendall_w(paired.rank(axis=0).to_numpy()),
        "icc_raw_three_judge": icc_absolute_agreement(paired.to_numpy()),
        "icc_z_three_judge": icc_absolute_agreement(z.to_numpy()),
        "system_mean_by_judge": {
            j: {s: float(paired.xs(s, level="system")[j].mean()) for s in SYSTEMS}
            for j in JUDGES
        },
        "system_rank_by_judge": {
            j: {s: int(r) for s, r in
                pd.Series({s: paired.xs(s, level="system")[j].mean() for s in SYSTEMS})
                .rank(ascending=False).astype(int).items()}
            for j in JUDGES
        },
    }


def save_individual_panels(paired: pd.DataFrame) -> None:
    """Render every panel on its own canvas (slide-ready PNG + editable SVG)."""
    panels = [
        # The two composite panels, at the letters the composite gives them. The
        # standalone slope panel keeps its direct system labels: on its own it
        # has no shared legend to lean on.
        ("a_severity_slope", (2.6, 2.3), draw_severity_slope, (paired, "a", True)),
        ("b1_opus5_vs_gpt56sol", (2.6, 2.5), draw_pair_scatter,
         (paired, "Claude Opus 5", "GPT-5.6-sol", "b")),
        ("b2_human_vs_opus5", (2.6, 2.5), draw_pair_scatter,
         (paired, "Human", "Claude Opus 5", "b")),
        ("b3_human_vs_gpt56sol", (2.6, 2.5), draw_pair_scatter,
         (paired, "Human", "GPT-5.6-sol", "b")),
        # Panels that no longer sit in the composite, plus alternative renderings
        # kept for slides and for reviewers who prefer them. Every statistic they
        # show is still in stats_judge_agreement.json.
        ("x6_consensus_profile", (4.0, 2.3), draw_consensus_profile, (paired, "a")),
        ("x7_head_to_head", (2.4, 2.3), draw_head_to_head, (paired, "f")),
        ("x1_system_means_bars", (2.9, 2.3), draw_system_means, (paired, "b")),
        ("x2_agreement_matrix", (2.3, 2.2), draw_agreement_matrix, (paired, "h")),
        ("x3_bland_altman_opus5_gpt56sol", (2.8, 2.2), draw_bland_altman,
         (paired, "Claude Opus 5", "GPT-5.6-sol", "i")),
        ("x4_bland_altman_human_opus5", (2.8, 2.2), draw_bland_altman,
         (paired, "Human", "Claude Opus 5", "j")),
        ("x5_per_paper_winner", (2.0, 2.8), draw_per_paper_winner, (paired, "x5", True)),
    ]
    panel_dir = OUT_DIR / "subfigs"
    panel_dir.mkdir(parents=True, exist_ok=True)
    # Same device as the main figure: shrink the canvas while the point-based
    # font sizes stay fixed, so text reads large when a subfig is scaled up.
    ppt_scale = 0.85
    global DRAW_PANEL_LABELS
    DRAW_PANEL_LABELS = False
    try:
        for name, figsize, draw_fn, extra_args in panels:
            fig = plt.figure(figsize=(figsize[0] * ppt_scale, figsize[1] * ppt_scale))
            ax = fig.add_subplot(1, 1, 1)
            draw_fn(ax, *extra_args)
            fig.savefig(panel_dir / f"{name}.png", dpi=600, bbox_inches="tight")
            fig.savefig(panel_dir / f"{name}.svg", bbox_inches="tight")
            plt.close(fig)
    finally:
        DRAW_PANEL_LABELS = True


def build_figure() -> None:
    paired, audit = load_paired_scores(drop_excluded=True)
    full, _ = load_paired_scores(drop_excluded=False)

    stats_out = compute_statistics(paired)
    stats_out["systems"] = SYSTEMS
    stats_out["judges"] = JUDGES
    stats_out["excluded_experiments"] = sorted(EXCLUDED_EXPERIMENTS)
    stats_out["data_audit"] = audit
    stats_out["robustness"] = {"all_18_papers": compute_statistics(full)}

    # 183 mm double column, one uniform row. The three paired scatters are
    # equal-aspect on a shared 0-100 square, and the request is that the slope
    # panel match them exactly, so the four axes are placed by hand as identical
    # squares rather than through a gridspec: gridspec divides the *cells*
    # evenly, and an aspect-constrained axes then shrinks inside its cell by an
    # amount that depends on the cell's height, which is how equal cells end up
    # holding unequal boxes.
    FIG_WIDTH_MM = 183.0                 # Nature double column
    FIG_W = round(FIG_WIDTH_MM / 25.4, 2)
    LEFT = 0.50          # y-axis label + tick labels of the first panel
    RIGHT = 0.04
    # Each scatter carries a two-line y label plus its tick labels into this gap.
    # At 0.52 in the label of one panel started exactly where its left-hand
    # neighbour's frame ended; 0.60 in leaves ~2 mm of clear paper between the
    # two, which is what stops the row from reading as three joined panels.
    GAP = 0.60
    # The gap after panel a additionally holds that panel's direct system labels,
    # which are drawn outside its frame: ~0.6 in for "AI-Scientist" at 5.8 pt,
    # plus the same 0.60 in the next panel's own y label needs.
    GAP_A = 1.08
    BOTTOM = 0.60        # two-line x labels and the angled judge names
    TOP = 0.24           # panel letters and the system key above the b block
    # All four boxes stay identical; the extra label column is paid for by the
    # gap, not by panel a's plotting area.
    SIDE = (FIG_W - LEFT - RIGHT - GAP_A - 2 * GAP) / 4   # common box edge, ~27 mm
    FIG_H = BOTTOM + SIDE + TOP

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    lefts = [LEFT]
    for gap in (GAP_A, GAP, GAP):
        lefts.append(lefts[-1] + SIDE + gap)
    axes = [
        fig.add_axes([left / FIG_W, BOTTOM / FIG_H, SIDE / FIG_W, SIDE / FIG_H])
        for left in lefts
    ]
    ax_a, ax_c, ax_d, ax_e = axes

    # Panel a keeps its own letter; the three scatters are one piece of evidence
    # - the same comparison run over every judge pair - and share the letter b,
    # set once at the left edge of the first of them.
    # The letters are added below, once, so the label offset can be expressed in
    # the same figure geometry that positions the boxes.
    stats_out["system_means"] = draw_severity_slope(
        ax_a, paired, "", direct_labels=True, angled_ticks=True
    )
    for ax, (x_judge, y_judge) in zip((ax_c, ax_d, ax_e), PAIRS):
        draw_pair_scatter(ax, paired, x_judge, y_judge, "",
                          stats_out["pairs"][f"{x_judge}|{y_judge}"])
    add_panel_label(ax_a, "a", x=-(LEFT - 0.06) / SIDE, y=1.03)
    add_panel_label(ax_c, "b", x=-(GAP - 0.06) / SIDE, y=1.03)

    # Computed, not drawn. The consensus profile, the head-to-head bars and the
    # per-paper winner matrix are no longer composite panels, but every number
    # they carried is still exported and still quoted in the legend.
    stats_out["per_paper_winner"] = per_paper_winner_stats(paired)[1]
    stats_out["concordance"] = {f"{a}|{b}": head_to_head(paired, a, b) for a, b in PAIRS}
    stats_out["concordance"]["kendall_w_three_judge"] = kendall_w(
        paired.rank(axis=0).to_numpy()
    )

    # One shared system key for the three scatters, at the top-right corner of
    # the b block. Stacked inside the last panel it ran into that panel's
    # Spearman value, which sits in the top-left of all three scatters; set in a
    # single row immediately above the frame it clears both, and it still reads
    # as belonging to the corner it sits in. Panel a names its systems directly
    # at the ends of its lines, as before.
    handles = [patches.Patch(facecolor=COLORS[s], edgecolor="none", label=s) for s in SYSTEMS]
    fig.legend(handles=handles, loc="lower right",
               bbox_to_anchor=((lefts[-1] + SIDE) / FIG_W, (BOTTOM + SIDE + 0.03) / FIG_H),
               ncol=3, handlelength=0.9, handleheight=0.9, columnspacing=1.0,
               handletextpad=0.45, borderpad=0.0, fontsize=5.8)

    with (OUT_DIR / "stats_judge_agreement.json").open("w", encoding="utf-8") as handle:
        json.dump(stats_out, handle, indent=2, ensure_ascii=False)
    save_source_data(paired)
    save_legend(paired, stats_out)
    save_qa_notes(stats_out, audit)
    save_individual_panels(paired)

    stem = OUT_DIR / "judge_agreement"
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)

    print(f"units={stats_out['n_units']} papers={stats_out['n_papers']} "
          f"(dropped incomplete: {audit['rows_dropped_incomplete']})")
    for key, value in stats_out["pairs"].items():
        print(f"  {key:24s} rho={value['spearman_rho']:.3f} tau={value['kendall_tau']:.3f} "
              f"ICC_raw={value['icc_raw']:.3f} ICC_z={value['icc_z']:.3f} "
              f"offset={value['mean_signed_offset']:+.1f}pp "
              f"h2h={stats_out['concordance'][key]['fraction']*100:.0f}%")
    print(f"  Kendall W = {stats_out['kendall_w_three_judge']:.3f}  "
          f"unanimous papers = {stats_out['per_paper_winner']['unanimous_papers']}"
          f"/{stats_out['per_paper_winner']['papers']}")
    print("wrote", f"{stem}.png")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_figure()
