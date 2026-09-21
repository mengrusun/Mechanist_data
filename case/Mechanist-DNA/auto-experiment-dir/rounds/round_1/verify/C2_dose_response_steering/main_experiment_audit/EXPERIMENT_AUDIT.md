# Experiment Audit Report — Claim C2

**Date**: 2026-07-18
**Auditor**: external LLM reviewer (cross-model, gpt-5.4 via llm-chat MCP)
**Project**: Feature Steering an α-Helix Knob in Evo2-7B
**Claim**: C2 — Amplifying the C1 α-helix feature set during Evo2-7B autoregressive DNA generation increases the encoded-protein α-helix fraction, monotonically with amplification strength up to an optimum.
**Linked milestones**: M1, M2

## Overall Verdict: WARN
*This is C2's integrity verdict — whether C2's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: WARN
The target metric (helix fraction) is DSSP run on an ESMFold-**predicted** structure of a newly
generated protein — there is necessarily no experimental structure for these sequences (inherent to
a causal-generation experiment, not intrinsically disqualifying). ESMFold and pLDDT gating are named
throughout and not hidden. However, the writeup leans on high pLDDT as reassuring quality evidence
without explicitly caveating that pLDDT is the predictor's own internal confidence, not independent
structural validation.

### B. Score Normalization: PASS
`aggregate()` computes helix/sheet fractions and pLDDT directly from DSSP-on-predicted-structure
output; no normalization against the model's own generation distribution. The steering coefficient
scaling (`s_f` = mean nonzero M0 activation magnitude) affects intervention *units*, not outcome
normalization.

### C. Result File Existence: PASS
`results/m2_dose_response_curve.json` matches `EXPERIMENT_RESULTS.md`'s M2 table exactly: per-dose
helix means, Spearman ρ=0.7592/p=1.696e-5, alpha_star=32.0, effect_size_delta=0.359, n_configs=24.

### D. Dead Code Detection: WARN
`m2_consolidate.py` computes the Spearman trend over 24 per-**seed** dose-mean points, while the
per-dose Mann-Whitney tests use **pooled per-sample** values — a legitimate but inconsistent choice
of aggregation level across the two statistics, underexplained in the prose (which reads as implying
a single consistent trend test). Additionally, `quality_ok(a)` (which gates `alpha_star` selection)
uses a permissive threshold (valid-ORF ≥ 50% of baseline, pLDDT ≥45) — materially looser than the
"quality within M1 tolerance" language implies, and this threshold directly determines the reported
"optimal" α.

### E. Scope Assessment: FAIL
Two compounding overclaims:
1. **Degenerate-sequence-collapse gap.** At the two highest doses (α=16, 32), sheet fraction
   collapses to near-zero (0.0093, 0.0030 from baseline 0.129 — a 97.7% reduction) while helix rises
   to 0.758/0.834 AND pLDDT (the structure predictor's own confidence) simultaneously *rises above
   baseline* (73.0, 79.7 vs 62.1), with a non-monotonic dip-then-recovery in valid-ORF/gated-pass
   rate. This is exactly the signature pattern of a structure predictor over-confidently folding
   simple, low-complexity, repetitive all-helical sequences (e.g. near-homopolymeric or short-period
   repeats) — and `mechanism.py` explicitly discards the actual generated protein sequence text
   after computing metrics (`rec.pop('prot', None)`), so no raw sequence-level output was ever saved
   to any result file. There is currently **no way to inspect** whether the high-α "helix-rich"
   proteins are diverse/plausible or degenerate/repetitive. M3's matched-control result addresses
   specificity-of-perturbation (rules out "any perturbation causes this"), but does not address
   realism/diversity of the sequences generated under the correct feature set specifically.
2. **Edge-of-grid "optimum."** `alpha_star=32` is the maximum dose in the tested grid, and helix is
   still monotonically increasing there — no interior peak/plateau was observed. Declaring this the
   "optimal α*" overclaims what the plan's own H2/pass-criteria describe ("an optimum"); the more
   honest reading is "response continues rising through the tested range; true ceiling not located."
   The separately-reported "conservative clean optimum" at α=8 is better practice, but the headline
   still foregrounds α*=32 as if a true optimum were found.

### F. Evaluation Type: synthetic_proxy
Outcome labels come from a structure predictor's (ESMFold) output on generated sequences, then DSSP
on that predicted structure — not experimental ground truth, and not self-supervised (does not reuse
the generator's own logits/score).

## Action Items
- Save a representative sample of raw generated/translated protein sequences per dose (especially
  α=16, 32) before discarding, and inspect for repetition/low-complexity/homopolymeric collapse; this
  is the single most decisive missing artifact for trusting the high-α headline result.
- Report the Spearman trend test over pooled per-sample data (or explicitly justify per-seed-mean
  aggregation) so the trend and pairwise tests use a consistent unit of analysis.
- Reframe "alpha_star=32" as "best dose within the tested grid (response still rising)" rather than
  "optimum," or extend the grid beyond 32 to actually locate a plateau/peak; tighten `quality_ok`'s
  threshold to match the "within M1 tolerance" language, or explicitly document the looser bar used.
