#!/usr/bin/env python3
"""Generate the Nature-style benchmark figure for Mechanist.

Panels c/e use real human and LLM judge totals for paper-reproduction
reliability; panels d/f/g use real per-claim LLM-judge scores (novelty, impact,
testability) for all three systems across four domains, 90 hypotheses each.

LLM-judge source: claude-opus-5 re-judge
(llm_judge/_orchestrator/score_stats/llm_scores_opus5.csv). Human-judge
scores are unchanged.
"""

from __future__ import annotations

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


# Mandatory editable-text settings for publication SVG output.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.dpi": 600,
    }
)


SYSTEMS = ["Claude Code", "AI-Scientist", "Mechanist"]
# Big folders in the eval results are the four domains, capitalised.
DOMAINS = ["Knowledge", "Language", "Safety", "Science"]
METRICS = ["Novelty", "Impact", "Testability"]
COLORS = {
    "Claude Code": "#7B7B7B",
    "AI-Scientist": "#4E79B8",
    "Mechanist": "#2E7D5B",
}
LIGHT_COLORS = {
    "Claude Code": "#D9D9D9",
    "AI-Scientist": "#C9D8EE",
    "Mechanist": "#D7E9DD",
}
RNG = np.random.default_rng(20260704)


def clip_score(values: np.ndarray | float) -> np.ndarray | float:
    return np.clip(values, 20, 98)


def bootstrap_ci(values: np.ndarray, n_boot: int = 4000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    boot = RNG.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high)


def icc_absolute_agreement(ratings: np.ndarray) -> float:
    """ICC(2,1), two-way random effects, single rater, absolute agreement.

    `ratings` is an (n_subjects, n_raters) array (here rater columns are the
    Human and LLM judges). Unlike Pearson r, this penalises systematic offsets
    between raters, so it is the standard measure of inter-rater agreement.
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


DRAW_PANEL_LABELS = True


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.08) -> None:
    if not DRAW_PANEL_LABELS:
        return
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        ha="left",
        # baseline anchoring keeps letters with descenders (e.g. "g") level with
        # ascender/x-height letters ("e", "f"), so the b..g labels sit on one line.
        va="baseline",
        color="#222222",
    )


def tidy_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(length=2.2, width=0.6, color="#333333", pad=2)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E7E7E7", linewidth=0.55, zorder=0)


# Real per-claim LLM-judge scores for the hypothesis-quality panels (d/f/g).
# All three systems have completed scoring across all four domains, 90 claims
# each (AI-Scientist=v2-opus4.7, Claude Code=cc, Mechanist=mechanist).
REAL_HYP_ROOTS = {
    "AI-Scientist": Path(os.environ.get("HYPOTHESIS_EVAL_ROOT", "hypothesis_eval")) / "eval_results/v2-opus4.7",
    "Claude Code": Path(os.environ.get("HYPOTHESIS_EVAL_ROOT", "hypothesis_eval")) / "eval_results/cc",
    "Mechanist": Path(os.environ.get("HYPOTHESIS_EVAL_ROOT", "hypothesis_eval")) / "eval_results/mechanist",
}
DIM_KEYS = {
    "novelty": "dimension_1_novelty",
    "impact": "dimension_2_impact",
    "testability": "dimension_3_testability",
}
FOLDER2DOMAIN = {
    "knowledge": "Knowledge",
    "language": "Language",
    "safety": "Safety",
    "science": "Science",
}
SCORE_MAP = 10.0  # 1-10 sub-scores -> 0-100

# Real judge totals (总分 / 0-100) live in the Reproduction repo's score_stats.
# scientist -> system mapping: s1=Claude Code, s2=Mechanist, s3=AI-Scientist.
SCORE_STATS = Path(__file__).resolve().parents[3]
HUMAN_SCORES_CSV = SCORE_STATS / "csv/judge_totals/human_scores.csv"
LLM_SCORES_CSV = SCORE_STATS / "csv/judge_totals/llm_scores_opus5.csv"
SCI2SYS = {"s1": "Claude Code", "s2": "Mechanist", "s3": "AI-Scientist"}
# Experiments excluded from panels c and e (removed for all systems, leaving 16
# papers): outlier reproductions flagged during review.
EXCLUDED_EXPERIMENTS = {"multi_lingual_reasoning", "circuit_breakers"}


def load_real_reliability() -> pd.DataFrame:
    """Load real per-paper reliability totals for the Human and LLM judges.

    One row per (paper=experiment, system, judge). Drives panels c (bars +
    per-paper points) and e (Human-LLM agreement). Panels d/f/g stay synthetic.
    """
    rows = []
    for judge, path in (("Human", HUMAN_SCORES_CSV), ("LLM", LLM_SCORES_CSV)):
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            system = SCI2SYS.get(str(r["sci"]).strip())
            if system is None:
                continue
            if str(r["实验"]).strip() in EXCLUDED_EXPERIMENTS:
                continue
            score = r["总分(0-100)"]
            if pd.isna(score):
                continue
            score = float(score)
            rows.append(
                {
                    "paper_id": str(r["实验"]).strip(),
                    "system": system,
                    "judge": judge,
                    "reliability_score": round(score, 2),
                }
            )
    return pd.DataFrame(rows)


def make_reliability_data() -> pd.DataFrame:
    papers = [f"P{i:02d}" for i in range(1, 16)]
    difficulty = RNG.normal(0, 5.2, len(papers))
    base = {"Claude Code": 53.5, "AI-Scientist": 62.0, "Mechanist": 81.8}
    llm_bias = {"Claude Code": 1.6, "AI-Scientist": 1.0, "Mechanist": -0.8}
    rows = []
    for i, paper in enumerate(papers):
        for system in SYSTEMS:
            human = float(clip_score(base[system] + difficulty[i] + RNG.normal(0, 4.3)))
            llm = float(clip_score(human + llm_bias[system] + RNG.normal(0, 3.7)))
            rows.append(
                {
                    "paper_id": paper,
                    "system": system,
                    "judge": "Human",
                    "reliability_score": round(human, 2),
                }
            )
            rows.append(
                {
                    "paper_id": paper,
                    "system": system,
                    "judge": "LLM",
                    "reliability_score": round(llm, 2),
                }
            )
    return pd.DataFrame(rows)


def load_real_hypotheses() -> pd.DataFrame:
    """Load per-claim LLM-judge scores for every system with completed scoring.

    Each score.json has three big dimensions (novelty/impact/testability), each
    the mean of three 1-10 sub-scores. Per the scoring spec: big-dimension score
    = mean of its 3 sub-scores x10; claim composite = mean of the 3 big-dimension
    means x10. Domain = capitalised big folder.
    """
    rows = []
    for system, root in REAL_HYP_ROOTS.items():
        for score_path in sorted(root.glob("**/score.json")):
            rel = score_path.relative_to(root).parts  # domain/sub/claim/score.json
            folder, claim = rel[0], rel[2]
            domain = FOLDER2DOMAIN.get(folder)
            if domain is None:
                continue
            data = json.loads(score_path.read_text())
            dim = {}
            for short, key in DIM_KEYS.items():
                subs = [v["score"] for v in data[key].values()]
                dim[short] = (sum(subs) / len(subs)) * SCORE_MAP
            composite = float(np.mean([dim["novelty"], dim["impact"], dim["testability"]]))
            rows.append(
                {
                    "domain": domain,
                    "system": system,
                    "hypothesis_id": claim,
                    "novelty": round(dim["novelty"], 2),
                    "impact": round(dim["impact"], 2),
                    "testability": round(dim["testability"], 2),
                    "composite_score": round(composite, 2),
                    "high_value": bool(
                        dim["novelty"] >= 75 and dim["impact"] >= 75 and dim["testability"] >= 70
                    ),
                    "data_source": "real",
                }
            )
    return pd.DataFrame(rows)


def make_hypothesis_data() -> pd.DataFrame:
    """Load real LLM-judge scores for all three systems (90 claims each, four
    domains). Rows are ordered to match SYSTEMS so downstream group-by/plot order
    is stable.
    """
    combined = load_real_hypotheses()
    combined["system"] = pd.Categorical(combined["system"], categories=SYSTEMS, ordered=True)
    combined = combined.sort_values(["system", "domain"]).reset_index(drop=True)
    combined["system"] = combined["system"].astype(str)
    return combined


def summarize_reliability(reliability: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (system, judge), group in reliability.groupby(["system", "judge"], sort=False):
        values = group["reliability_score"].to_numpy()
        ci_low, ci_high = bootstrap_ci(values)
        rows.append(
            {
                "system": system,
                "judge": judge,
                "n_papers": group["paper_id"].nunique(),
                "mean_score": round(values.mean(), 2),
                "ci95_low": round(ci_low, 2),
                "ci95_high": round(ci_high, 2),
            }
        )
    return pd.DataFrame(rows)


def summarize_hypotheses(hypotheses: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (system, metric), group in hypotheses.melt(
        id_vars=["system", "domain", "hypothesis_id"],
        value_vars=["novelty", "impact", "testability"],
        var_name="metric",
        value_name="score",
    ).groupby(["system", "metric"], sort=False):
        values = group["score"].to_numpy()
        ci_low, ci_high = bootstrap_ci(values)
        rows.append(
            {
                "system": system,
                "metric": metric.capitalize(),
                "n_hypotheses": values.size,
                "mean_score": round(values.mean(), 2),
                "ci95_low": round(ci_low, 2),
                "ci95_high": round(ci_high, 2),
            }
        )
    return pd.DataFrame(rows)


def draw_design_panel(ax: plt.Axes) -> None:
    ax.set_axis_off()
    add_panel_label(ax, "b", x=-0.08, y=1.02)
    ax.set_title("Benchmark design", loc="left", pad=6)

    def box(x, y, w, h, text, fc, ec="#555555", lw=0.7, weight="normal", fontsize=5.2):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#222222",
            fontweight=weight,
            linespacing=0.95,
        )

    def arrow(x0, y0, x1, y1):
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#555555", shrinkA=3, shrinkB=6),
            zorder=0,
        )

    ax.text(0.01, 0.88, "Systems", transform=ax.transAxes, fontsize=5.6, color="#555555")
    y_positions = [0.69, 0.50, 0.31]
    compact_systems = ["Claude\nCode", "AI-\nScientist", "Our\nMechanist"]
    for y, system, label in zip(y_positions, SYSTEMS, compact_systems):
        box(0.01, y, 0.30, 0.12, label, LIGHT_COLORS[system], ec=COLORS[system], weight="bold", fontsize=4.8)

    ax.text(0.39, 0.88, "Tasks", transform=ax.transAxes, fontsize=5.6, color="#555555")
    box(0.38, 0.58, 0.24, 0.19, "Repro-\nduction", "#F6F6F6")
    box(0.38, 0.25, 0.24, 0.22, "Hypothesis\ndiscovery", "#F6F6F6", fontsize=5.2)

    ax.text(0.74, 0.88, "Judges", transform=ax.transAxes, fontsize=5.6, color="#555555")
    box(0.73, 0.58, 0.24, 0.19, "Human\n+ LLM", "#EFECE5")
    box(0.73, 0.25, 0.24, 0.22, "LLM:\nnovelty\nimpact\ntestability", "#EFECE5", fontsize=4.9)

    arrow(0.31, 0.56, 0.375, 0.67)
    arrow(0.31, 0.50, 0.375, 0.36)
    arrow(0.62, 0.67, 0.725, 0.67)
    arrow(0.62, 0.36, 0.725, 0.36)

    ax.text(0.01, 0.08, "Real human and LLM judge scores", transform=ax.transAxes, fontsize=5.3, color="#555555")


def draw_reliability_panel(ax: plt.Axes, reliability: pd.DataFrame, summary: pd.DataFrame) -> None:
    add_panel_label(ax, "c")
    ax.set_title("Reliability of paper reproduction", loc="left", pad=6)
    judges = ["Human", "LLM"]
    offsets = np.array([-0.23, 0.0, 0.23])
    width = 0.19
    rng = np.random.default_rng(14)

    for j, judge in enumerate(judges):
        for i, system in enumerate(SYSTEMS):
            x = j + offsets[i]
            row = summary[(summary["system"] == system) & (summary["judge"] == judge)].iloc[0]
            yerr = np.array([[row["mean_score"] - row["ci95_low"]], [row["ci95_high"] - row["mean_score"]]])
            ax.bar(
                x,
                row["mean_score"],
                width=width,
                color=LIGHT_COLORS[system],
                edgecolor=COLORS[system],
                linewidth=0.8,
                zorder=2,
            )
            ax.errorbar(
                x,
                row["mean_score"],
                yerr=yerr,
                fmt="none",
                ecolor=COLORS[system],
                elinewidth=0.75,
                capsize=2.0,
                zorder=3,
            )
            values = reliability[(reliability["system"] == system) & (reliability["judge"] == judge)][
                "reliability_score"
            ].to_numpy()
            jitter = rng.normal(0, 0.026, size=values.size)
            ax.scatter(
                np.full(values.size, x) + jitter,
                values,
                s=8,
                color=COLORS[system],
                alpha=0.56,
                linewidths=0,
                zorder=4,
            )
    ax.set_ylabel("Reliability score (%)")
    ax.set_ylim(30, 100)
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(judges)
    tidy_axis(ax)


def draw_domain_heatmap(ax: plt.Axes, hypotheses: pd.DataFrame) -> None:
    add_panel_label(ax, "d")
    ax.set_title("Overall hypothesis evaluation", loc="left", pad=6)
    heat = (
        hypotheses.groupby(["domain", "system"], sort=False)["composite_score"]
        .mean()
        .unstack("system")
        .loc[DOMAINS, SYSTEMS]
    )
    cmap = LinearSegmentedColormap.from_list(
        "quality", ["#F8F8F6", "#E8EFEA", "#CCDCCE", "#92BFA3", "#2E7D5B"]
    )
    im = ax.imshow(heat.to_numpy(), cmap=cmap, vmin=55, vmax=85, aspect="auto")
    ax.set_xticks(np.arange(len(SYSTEMS)))
    ax.set_xticklabels(["Claude\nCode", "AI-\nScientist", "Our\nMechanist"])
    ax.set_yticks(np.arange(len(DOMAINS)))
    ax.set_yticklabels(DOMAINS)
    for yi, domain in enumerate(DOMAINS):
        for xi, system in enumerate(SYSTEMS):
            val = heat.loc[domain, system]
            color = "white" if val > 78 else "#222222"
            ax.text(xi, yi, f"{val:.0f}", ha="center", va="center", fontsize=6.2, color=color)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(SYSTEMS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DOMAINS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=5.8, length=2, width=0.5)
    cbar.outline.set_linewidth(0.55)
    cbar.set_label("Mean score (%)", fontsize=6)


def draw_agreement_panel(ax: plt.Axes, reliability: pd.DataFrame) -> dict[str, float]:
    add_panel_label(ax, "e")
    ax.set_title("Human–LLM judge agreement", loc="left", pad=6)
    paired = (
        reliability.pivot_table(
            index=["paper_id", "system"], columns="judge", values="reliability_score", aggfunc="mean"
        )
        .reset_index()
        .dropna()
    )
    for system in SYSTEMS:
        group = paired[paired["system"] == system]
        ax.scatter(
            group["Human"],
            group["LLM"],
            s=15,
            color=COLORS[system],
            alpha=0.74,
            linewidth=0.25,
            edgecolor="white",
            label=system,
        )
    lims = [0, 100]
    # Dashed reference line is a through-origin least-squares fit y = kx to the
    # paired points (k = sum(xy) / sum(x^2)), not the identity line.
    x_pts = paired["Human"].to_numpy(dtype=float)
    y_pts = paired["LLM"].to_numpy(dtype=float)
    k_origin = float(np.sum(x_pts * y_pts) / np.sum(x_pts**2))
    ax.plot(
        lims,
        [k_origin * lims[0], k_origin * lims[1]],
        color="#8A8A8A",
        linewidth=0.8,
        linestyle=(0, (3, 2)),
    )
    slope, intercept, r_value, p_value, _ = stats.linregress(paired["Human"], paired["LLM"])
    mae = float(np.mean(np.abs(paired["Human"] - paired["LLM"])))
    rho, rho_p = stats.spearmanr(paired["Human"], paired["LLM"])
    icc = icc_absolute_agreement(paired[["Human", "LLM"]].to_numpy())
    ax.text(
        0.04,
        0.96,
        f"Spearman ρ = {rho:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        color="#222222",
    )
    ax.set_xlabel("Human judge score (%)")
    ax.set_ylabel("LLM judge score (%)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    # Matching 0-100 ticks every 20 on both axes so the scales are identical.
    ticks = np.arange(0, 101, 20)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    # Equal aspect so identical x/y ranges map to identical visual lengths and
    # the identity line reads as a true 45 degrees.
    ax.set_aspect("equal", adjustable="box")
    tidy_axis(ax)
    return {
        "human_llm_r": float(r_value),
        "human_llm_p": float(p_value),
        "human_llm_spearman_rho": float(rho),
        "human_llm_spearman_p": float(rho_p),
        "human_llm_icc": float(icc),
        "human_llm_mae": mae,
        "human_llm_slope_through_origin": k_origin,
    }




def draw_hypothesis_metric_panel(ax: plt.Axes, hypotheses: pd.DataFrame, summary: pd.DataFrame) -> None:
    add_panel_label(ax, "f")
    ax.set_title("Comprehensive hypothesis evaluation", loc="left", pad=6)
    x_base = np.arange(len(METRICS))
    offsets = np.array([-0.23, 0.0, 0.23])
    width = 0.19
    for i, system in enumerate(SYSTEMS):
        means = []
        lows = []
        highs = []
        for metric in METRICS:
            row = summary[(summary["system"] == system) & (summary["metric"] == metric)].iloc[0]
            means.append(row["mean_score"])
            lows.append(row["mean_score"] - row["ci95_low"])
            highs.append(row["ci95_high"] - row["mean_score"])
        ax.bar(
            x_base + offsets[i],
            means,
            width=width,
            color=LIGHT_COLORS[system],
            edgecolor=COLORS[system],
            linewidth=0.8,
            zorder=2,
        )
        ax.errorbar(
            x_base + offsets[i],
            means,
            yerr=np.array([lows, highs]),
            fmt="none",
            ecolor=COLORS[system],
            elinewidth=0.75,
            capsize=2.0,
            zorder=3,
        )
        for x, y in zip(x_base + offsets[i], means):
            ax.text(x, y + 2.2, f"{y:.0f}", ha="center", va="bottom", fontsize=5.8, color=COLORS[system])
    ax.set_ylabel("Score (%)")
    ax.set_ylim(30, 100)
    ax.set_xticks(x_base)
    ax.set_xticklabels(METRICS)
    tidy_axis(ax)


def draw_hypothesis_cloud_panel(ax: plt.Axes, hypotheses: pd.DataFrame) -> None:
    add_panel_label(ax, "g")
    ax.set_title("Joint novelty–impact space", loc="left", pad=6)
    ax.axvspan(75, 100, ymin=(75 - 30) / 70, ymax=1, color="#2E7D5B", alpha=0.06, zorder=0)
    ax.axvline(75, color="#777777", linestyle=(0, (3, 2)), linewidth=0.7)
    ax.axhline(75, color="#777777", linestyle=(0, (3, 2)), linewidth=0.7)
    ax.text(79, 96, "high-novelty/\nhigh-impact", fontsize=5.6, color="#555555", va="top")
    for system in SYSTEMS:
        group = hypotheses[hypotheses["system"] == system]
        sizes = 5 + (group["testability"].to_numpy() - 30) * 0.19
        ax.scatter(
            group["novelty"],
            group["impact"],
            s=sizes,
            color=COLORS[system],
            alpha=0.18 if system != "Mechanist" else 0.22,
            linewidths=0,
        )
        centroid = group[["novelty", "impact"]].mean()
        ax.scatter(
            centroid["novelty"],
            centroid["impact"],
            s=70,
            color=COLORS[system],
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        # Claude Code and Mechanist centroids nearly coincide (both low-novelty),
        # so stagger their labels vertically: Claude Code above, Mechanist below.
        label_offsets = {
            "AI-Scientist": (1.8, 1.8, "left", "bottom"),
            "Claude Code": (2.0, 4.0, "left", "bottom"),
            "Mechanist": (-2.0, -4.0, "right", "top"),
        }
        dx, dy, ha, va = label_offsets[system]
        ax.text(
            centroid["novelty"] + dx,
            centroid["impact"] + dy,
            system,
            fontsize=6.2,
            color=COLORS[system],
            ha=ha,
            va=va,
        )
    ax.set_xlim(30, 100)
    ax.set_ylim(30, 100)
    ax.set_xlabel("Novelty score (%)")
    ax.set_ylabel("Impact score (%)")
    ax.text(0.02, 0.05, "Point size encodes testability", transform=ax.transAxes, fontsize=5.8, color="#555555")
    tidy_axis(ax)


def calculate_stats(reliability: pd.DataFrame, hypotheses: pd.DataFrame) -> dict[str, object]:
    stats_out: dict[str, object] = {}
    human = reliability[reliability["judge"] == "Human"]
    llm = reliability[reliability["judge"] == "LLM"]
    for judge_name, frame in [("human", human), ("llm", llm)]:
        mechanist = frame[frame["system"] == "Mechanist"]["reliability_score"].to_numpy()
        stats_out[f"reliability_{judge_name}"] = {}
        for comparator in ["Claude Code", "AI-Scientist"]:
            other = frame[frame["system"] == comparator]["reliability_score"].to_numpy()
            test = stats.ttest_ind(mechanist, other, equal_var=False)
            stats_out[f"reliability_{judge_name}"][f"Mechanist_vs_{comparator}"] = {
                "mean_difference": float(mechanist.mean() - other.mean()),
                "welch_t": float(test.statistic),
                "p_value": float(test.pvalue),
            }
    stats_out["hypothesis_composite"] = {}
    mechanist_comp = hypotheses[hypotheses["system"] == "Mechanist"]["composite_score"].to_numpy()
    for comparator in ["Claude Code", "AI-Scientist"]:
        other = hypotheses[hypotheses["system"] == comparator]["composite_score"].to_numpy()
        test = stats.ttest_ind(mechanist_comp, other, equal_var=False)
        stats_out["hypothesis_composite"][f"Mechanist_vs_{comparator}"] = {
            "mean_difference": float(mechanist_comp.mean() - other.mean()),
            "welch_t": float(test.statistic),
            "p_value": float(test.pvalue),
        }
    return stats_out


def save_outputs(
    reliability: pd.DataFrame,
    hypotheses: pd.DataFrame,
    reliability_summary: pd.DataFrame,
    hypothesis_summary: pd.DataFrame,
    stats_out: dict[str, object],
) -> None:
    reliability.to_csv(OUT_DIR / "source_data_reliability.csv", index=False)
    hypotheses.to_csv(OUT_DIR / "source_data_hypotheses.csv", index=False)
    reliability_summary.to_csv(OUT_DIR / "summary_reliability.csv", index=False)
    hypothesis_summary.to_csv(OUT_DIR / "summary_hypothesis_quality.csv", index=False)
    stale_high_value = OUT_DIR / "summary_high_value_yield.csv"
    if stale_high_value.exists():
        stale_high_value.unlink()
    with open(OUT_DIR / "stats_summary.json", "w", encoding="utf-8") as handle:
        json.dump(stats_out, handle, indent=2)

    legend = """# Figure legend

Fig. 1 | Benchmark of automated scientific reproduction and hypothesis generation.

b, Benchmark design. Three systems were evaluated on reproduction of 15 papers and on hypothesis generation across four scientific domains. Reliability was scored by human and LLM judges; hypothesis quality was scored by an LLM judge for novelty, impact and testability. c, Reproduction reliability score for each system and judge. Bars show mean; whiskers show bootstrap 95% confidence intervals; points denote individual papers. d, Domain-level composite hypothesis quality, defined as the mean of novelty, impact and testability. e, Human-LLM judge agreement for reproduction reliability across all system-paper pairs. f, Mean LLM-judged hypothesis quality across 90 hypotheses per system. g, Distribution of hypotheses in novelty-impact space; point size encodes testability and large markers show system centroids.

All hypothesis-quality panels (d, f, g) use real per-claim LLM-judge scores for all three systems across all four domains (90 hypotheses each). Reproduction-reliability panels (c, e) use real human and LLM judge totals for all systems.
"""
    with open(OUT_DIR / "figure_legend.md", "w", encoding="utf-8") as handle:
        handle.write(legend)


def save_individual_panels(
    reliability: pd.DataFrame,
    hypotheses: pd.DataFrame,
    reliability_summary: pd.DataFrame,
    hypothesis_summary: pd.DataFrame,
) -> None:
    """Render each subpanel on its own figure and save as a standalone PNG."""
    panels = [
        ("b_design", (2.8, 2.3), draw_design_panel, ()),
        ("c_reliability", (2.8, 2.3), draw_reliability_panel, (reliability, reliability_summary)),
        ("d_domain_heatmap", (2.8, 2.3), draw_domain_heatmap, (hypotheses,)),
        ("e_agreement", (2.8, 2.3), draw_agreement_panel, (reliability,)),
        ("f_hypothesis_quality", (2.8, 2.3), draw_hypothesis_metric_panel, (hypotheses, hypothesis_summary)),
        ("g_novelty_impact", (2.8, 2.3), draw_hypothesis_cloud_panel, (hypotheses,)),
    ]
    panel_dir = OUT_DIR / "subfigs"
    panel_dir.mkdir(parents=True, exist_ok=True)
    # Shrinking the canvas while keeping the point-based font sizes fixed makes
    # every text element (including hard-coded annotations) larger relative to
    # the panel, so the subfigs read clearly when scaled up on a PPT slide.
    ppt_scale = 0.8
    global DRAW_PANEL_LABELS
    DRAW_PANEL_LABELS = False
    try:
        for name, figsize, draw_fn, extra_args in panels:
            figsize = (figsize[0] * ppt_scale, figsize[1] * ppt_scale)
            fig = plt.figure(figsize=figsize, constrained_layout=False)
            ax = fig.add_subplot(1, 1, 1)
            draw_fn(ax, *extra_args)
            # drop the per-panel subtitle for standalone subfigs (panels set it
            # with loc="left", so the left title object must be cleared too).
            for loc in ("center", "left", "right"):
                ax.set_title("", loc=loc)
            # High-res PNG (600 dpi) plus a vector SVG. Insert the SVG in PPT —
            # it stays crisp at any zoom; the PNG is a raster fallback.
            fig.savefig(panel_dir / f"{name}.png", dpi=600, bbox_inches="tight")
            fig.savefig(panel_dir / f"{name}.svg", bbox_inches="tight")
            plt.close(fig)
    finally:
        DRAW_PANEL_LABELS = True


def build_figure() -> None:
    reliability = load_real_reliability()  # panels c & e use real judge totals
    hypotheses = make_hypothesis_data()  # panels d/f/g use real per-claim scores
    reliability_summary = summarize_reliability(reliability)
    hypothesis_summary = summarize_hypotheses(hypotheses)
    stats_out = calculate_stats(reliability, hypotheses)

    fig = plt.figure(figsize=(7.2, 4.85), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        left=0.065,
        right=0.985,
        top=0.94,
        bottom=0.12,
        wspace=0.62,
        hspace=0.48,
        width_ratios=[1.22, 1.34, 1.18],
        height_ratios=[1.0, 1.0],
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    draw_design_panel(ax_a)
    draw_reliability_panel(ax_b, reliability, reliability_summary)
    draw_domain_heatmap(ax_c, hypotheses)
    agreement_stats = draw_agreement_panel(ax_d, reliability)
    draw_hypothesis_metric_panel(ax_e, hypotheses, hypothesis_summary)
    draw_hypothesis_cloud_panel(ax_f, hypotheses)

    stats_out["judge_agreement"] = agreement_stats
    save_outputs(reliability, hypotheses, reliability_summary, hypothesis_summary, stats_out)
    save_individual_panels(reliability, hypotheses, reliability_summary, hypothesis_summary)

    stem = OUT_DIR / "mechanist_synthetic_benchmark"
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    try:
        fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    except TypeError:
        fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_figure()
