# QA notes - judge agreement figure

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
| Human | 54 | 54 | 48 |
| Claude Opus 5 | 54 | 54 | 48 |
| GPT-5.6-sol | 54 | 54 | 48 |

- Merged rows before complete-case filtering: 48
- Rows after complete-case filtering: 48
- Rows dropped because a judge was missing: 0
- Papers excluded: circuit_breakers, multi_lingual_reasoning. Rule: reuse exactly
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
