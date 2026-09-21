#!/usr/bin/env python3
"""Standalone panel c (reproduction reliability) with Claude Opus 5 as the LLM judge.

Same panel, same geometry, same palette and the same standalone-subfig export
settings as `c_reliability` in make_mechanist_figure.py — only the LLM-judge
score table is swapped:

  this script (c_reliability-claudeopus5) uses the copied Opus 5 score table.

Human-judge scores, the 16-paper set (circuit_breakers and multi_lingual_reasoning
excluded) and the bootstrap CI procedure are unchanged, so the two PNGs are
directly comparable.

Outputs
  ./subfigs/c_reliability-claudeopus5.{png,svg}
  ./source_data_reliability-claudeopus5.csv
  ./summary_reliability-claudeopus5.csv
"""
from __future__ import annotations

from pathlib import Path

import make_mechanist_figure as mech
import matplotlib.pyplot as plt

SCORE_STATS = Path(__file__).resolve().parents[3]
CLAUDE_OPUS5_SCORES_CSV = SCORE_STATS / "csv/judge_totals/llm_scores_opus5.csv"
JUDGE_LABEL = "Claude Opus 5"
SUFFIX = "-claudeopus5"
Y_LIMITS = (30, 100)


def main() -> None:
    mech.LLM_SCORES_CSV = CLAUDE_OPUS5_SCORES_CSV
    reliability = mech.load_real_reliability()
    summary = mech.summarize_reliability(reliability)

    reliability.to_csv(mech.OUT_DIR / f"source_data_reliability{SUFFIX}.csv", index=False)
    summary.to_csv(mech.OUT_DIR / f"summary_reliability{SUFFIX}.csv", index=False)

    # Identical to save_individual_panels(): 0.8x canvas with point-based fonts
    # held fixed, no panel letter, no subtitle, 600 dpi PNG plus a vector SVG.
    mech.DRAW_PANEL_LABELS = False
    try:
        fig = plt.figure(figsize=(2.8 * 0.8, 2.3 * 0.8), constrained_layout=False)
        ax = fig.add_subplot(1, 1, 1)
        mech.draw_reliability_panel(ax, reliability, summary)
        for loc in ("center", "left", "right"):
            ax.set_title("", loc=loc)
        ax.set_ylim(*Y_LIMITS)
        # The judge axis is the only label that changes from the generic panel.
        ax.set_xticklabels(["Human", JUDGE_LABEL])

        panel_dir = mech.OUT_DIR / "subfigs"
        panel_dir.mkdir(parents=True, exist_ok=True)
        stem = panel_dir / f"c_reliability{SUFFIX}"
        fig.savefig(f"{stem}.png", dpi=600, bbox_inches="tight")
        fig.savefig(f"{stem}.svg", bbox_inches="tight")
        plt.close(fig)
    finally:
        mech.DRAW_PANEL_LABELS = True

    print("judge         system         n   mean   95% CI")
    for _, r in summary.iterrows():
        judge = JUDGE_LABEL if r["judge"] == "LLM" else r["judge"]
        print(f"  {judge:<11} {r['system']:<13} {r['n_papers']:>2}  {r['mean_score']:>5}  "
              f"[{r['ci95_low']}, {r['ci95_high']}]")
    print(f"\nwrote {stem}.{{png,svg}}")


if __name__ == "__main__":
    main()
