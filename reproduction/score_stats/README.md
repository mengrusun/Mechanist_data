# Score Statistics For Reliability Figures

This directory collects the data tables and scripts used for the four
reliability figures:

1. `fig/appendix_fig/agreement/judge_agreement.png`
2. `fig/appendix_fig/subfigs/reliability-by-area.png`
3. `fig/appendix_fig/subfigs/reliability-by-dimension.png`
4. `fig/main_fig/subfigs/c_reliability.png`

## Directory layout

```text
score_stats/
├── csv/
│   ├── judge_totals/
│   │   ├── human_scores.csv
│   │   ├── llm_scores_opus5.csv
│   │   └── llm_scores_gpt56sol.csv
│   └── figure_inputs/
│       ├── judge_agreement/
│       │   ├── source_data_judge_scores.csv
│       │   └── stats_judge_agreement.json
│       ├── reliability_by_area/
│       │   ├── source_data_category_system.csv
│       │   └── summary_category_system.csv
│       ├── reliability_by_dimension/
│       │   ├── source_data_macro_dims.csv
│       │   └── summary_macro_dims.csv
│       └── reliability/
│           ├── source_data_reliability-claudeopus5.csv
│           ├── summary_reliability-claudeopus5.csv
│           ├── source_data_reliability-gpt56sol.csv
│           └── summary_reliability-gpt56sol.csv
├── scripts/
│   ├── aggregation/
│   │   ├── aggregate_human_scores.py
│   │   ├── aggregate_opus5.py
│   │   └── aggregate_gpt56sol.py
│   └── figures/
│       ├── judge_agreement/make_judge_agreement_figure.py
│       ├── reliability_by_area/make_category_system_figure.py
│       ├── reliability_by_dimension/make_macro_dimension_figure.py
│       └── reliability/
│           ├── make_mechanist_figure.py
│           ├── make_c_reliability_claudeopus5.py
│           └── make_c_reliability_gpt56sol.py
├── figures/
│   ├── reliability_by_area/
│   │   └── reliability-by-area.{png,svg,pdf,tiff}
│   ├── reliability_by_dimension/
│   │   └── reliability-by-dimension.{png,svg,pdf,tiff}
│   └── reliability/
│       ├── c_reliability-claudeopus5.{png,svg}
│       └── c_reliability-gpt56sol.{png,svg}
└── rules/
    └── scoring_rules.md
```

## Figure-to-data mapping

| Figure | Plotting script | Source data | Summary/statistics |
|---|---|---|---|
| Judge agreement | `scripts/figures/judge_agreement/make_judge_agreement_figure.py` | `csv/figure_inputs/judge_agreement/source_data_judge_scores.csv` | `stats_judge_agreement.json` |
| Reliability by area | `scripts/figures/reliability_by_area/make_category_system_figure.py` | `source_data_category_system.csv` | `summary_category_system.csv` |
| Reliability by dimension | `scripts/figures/reliability_by_dimension/make_macro_dimension_figure.py` | `source_data_macro_dims.csv` | `summary_macro_dims.csv` |
| Overall reliability (Claude Opus 5) | `scripts/figures/reliability/make_c_reliability_claudeopus5.py` | `source_data_reliability-claudeopus5.csv` | `summary_reliability-claudeopus5.csv` |
| Overall reliability (GPT-5.6-sol) | `scripts/figures/reliability/make_c_reliability_gpt56sol.py` | `source_data_reliability-gpt56sol.csv` | `summary_reliability-gpt56sol.csv` |

## Statistical flow

```text
raw judge JSON
    └── scripts/aggregation/
        └── csv/judge_totals/*.csv
            └── scripts/figures/
                ├── csv/figure_inputs/source_data_*.csv
                ├── csv/figure_inputs/summary_*.csv
                └── figure-specific statistics JSON
```

The `judge_totals` tables are the upstream per-experiment/per-scientist score
tables. The `figure_inputs` tables are the tidy source and summary tables used
by the four plots. The aggregation scripts apply the rules in
`rules/scoring_rules.md` to raw judge JSON files; the figure scripts then
reshape the totals, calculate means and bootstrap 95% confidence intervals,
and write the figure-ready CSVs.

The GPT-5.6-sol table is included for the three-judge agreement analysis and
the optional GPT-5.6-sol reliability-panel variant. The Claude Opus 5 panel
uses the explicit `claudeopus5` suffix in the copied image and script names so
it is not confused with the GPT-5.6-sol variant.

The aggregation scripts expect the raw judge JSON corpus under
`$REPRODUCTION_ROOT` when run. If that variable is unset, they default to the
workspace's sibling `reproduction/` directory. The figure scripts in this
directory read the copied `csv/judge_totals/` tables by default. The complete
hypothesis-evaluation corpus needed by the full `make_mechanist_figure.py`
script can be supplied with `$HYPOTHESIS_EVAL_ROOT`; the standalone reliability
panel only needs the copied score tables.
