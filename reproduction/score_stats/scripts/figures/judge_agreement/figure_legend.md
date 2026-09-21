Extended Data Fig. | Human and LLM reproduction-reliability judges differ in severity but not in conclusion.

A human expert panel, Claude Opus 5 and GPT-5.6-sol independently scored the same
48 system-paper units (16 papers x 3 systems) on the identical 0-100
reproduction-reliability rubric. **a**, Mean score of each system
under each judge; points are means over 16 papers and whiskers are percentile
bootstrap 95% confidence intervals (4,000 resamples). The y-axis is truncated at
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
74.0% (Human), 57.1% (Claude Opus 5) and 43.3% (GPT-5.6-sol) - but
they order the work the same way. The system ranking is identical under all
three judges (a), Kendall's W = 0.76 across the 48 units, 7/16 papers have a
unanimous winner against 1.8 expected by chance, and every judge pair names
the same winner on 73-83% of the 48 head-to-head system comparisons
(16 papers x 3 system pairs, chance 50%). For the two LLM
judges Spearman rho = 0.78 with
ICC(2,1) rising from 0.46 on raw scores to 0.77 once each judge's severity
and spread are removed (b, left). Residual disagreement is therefore a calibration
offset (-13.8 percentage points between the two LLM judges) rather than a
conflicting verdict. The per-unit standardised-score profile, the head-to-head
concordance bars and the per-paper winner matrix are not shown here; they are
rendered as subfigs/x6_consensus_profile, subfigs/x7_head_to_head and
subfigs/x5_per_paper_winner and their statistics are in stats_judge_agreement.json.

Statistics
----------
n definition            : one unit = one paper scored for one system; n = 48
                          (16 papers x 3 systems), complete cases only.
biological replicates   : not applicable (each unit is scored once per judge).
technical replicates    : not applicable (single judge pass per unit).
center statistic        : mean score per system per judge (a); individual units
                          are plotted unaggregated (b).
spread/interval         : percentile bootstrap 95% CI, 4,000 resamples (a);
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
data exclusion          : circuit_breakers, multi_lingual_reasoning are excluded to
                          match the main figure's reliability panels (16 of 18
                          papers retained). With all 18 papers (54 units) the
                          conclusion is unchanged: Kendall's W = 0.73 and the
                          two LLM judges give rho = 0.77, ICC on z = 0.77.
