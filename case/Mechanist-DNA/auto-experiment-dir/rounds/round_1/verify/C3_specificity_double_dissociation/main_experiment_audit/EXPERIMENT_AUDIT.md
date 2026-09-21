# Experiment Audit Report — Claim C3

**Date**: 2026-07-18
**Auditor**: external LLM reviewer (cross-model, gpt-5.4 via llm-chat MCP)
**Project**: Feature Steering an α-Helix Knob in Evo2-7B
**Claim**: C3 — The amplification-induced α-helix rise is specific to the α-helix feature set S (matched-control/off-target β-sheet do not reproduce it, double dissociation), generation quality preserved — S is a causally manipulable, specific knob.
**Linked milestone**: M3

## Overall Verdict: WARN
*This is C3's integrity verdict — whether C3's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: WARN
Same proxy pattern as C2 (shares `mechanism.py`'s ESMFold+DSSP readout on generated sequences) —
no new provenance defect introduced by M3 specifically, but still synthetic-proxy, not experimental
ground truth.

### B. Score Normalization: PASS
Double-dissociation statistics (S vs matched-control, β vs S) are direct Mann-Whitney comparisons of
pooled helix/sheet readouts and raw deltas from the shared α=0 baseline; no self-referential
normalization.

### C. Result File Existence: PASS
`results/m3_specificity_summary.json` matches `EXPERIMENT_RESULTS.md`'s M3 numbers exactly:
helix_delta_S=0.2847≈+0.285, helix_delta_matched=-0.0951≈-0.095, helix_delta_beta=0.0105≈+0.010,
sheet_delta_S=-0.1110≈-0.111, sheet_delta_beta=0.0048≈+0.005, S_vs_matched_helix_p=3.22e-23≈3.2e-23,
beta_vs_S_sheet_p=6.17e-30≈6.2e-30.

### D. Dead Code Detection: WARN
The shared `aggregate()` function (in `mechanism.py`) computes `plddt_mean` over **all** folded
records with a non-null pLDDT value, regardless of whether that record passed the pLDDT≥50 gate —
while the target metrics (helix/sheet means) are correctly restricted to gate-passing records only.
This is a real statistic-definition inconsistency: `matched_control`'s reported `pLDDT=48.57` at
α=16 (below the stated gate threshold) is the raw mean over all attempted foldings, not the gated
population that backs the helix/sheet summary. This makes cross-arm quality comparison ambiguous —
in this instance it likely makes matched-control's α=16 quality look artificially poorer than the
gated population it's implicitly compared against, muddying the "quality guardrail" narrative.

### E. Scope Assessment: FAIL
"SUPPORTED" overstates what M3 actually demonstrated, on two compounding grounds:
1. **The plan's own double-dissociation criterion is only half-met.** The β-sheet off-target arm
   was required to raise sheet-not-helix; it does not — sheet stays essentially flat across the
   entire dose range (0.1144→0.1150→0.1194→0.1192, Δ=+0.0048 at α=16). The reported
   `beta_vs_S_sheet_p=6.2e-30` is significant only because **S suppresses its own sheet content**
   (down to 0.0034), not because β amplifies sheet — this is a real effect but not the planned
   off-target positive control succeeding. EXPERIMENT_RESULTS.md does transparently flag this arm as
   "weak," but still concludes "SUPPORTED" for the full double-dissociation claim, which the data do
   not fully establish.
2. **Single-dose decisive test.** The formal dissociation statistics are computed only at α=16 (the
   grid maximum), not as a trend across {0,4,8,16}. The underlying per-dose table is available and
   broadly consistent with the single-dose story, which mitigates but does not eliminate cherry-pick
   risk.
3. **Matched-control's helix DECREASE (not just "no rise").** At α=16, matched-control helix drops to
   0.398 from baseline 0.493 (Δ=-0.095) — a stronger dissociation from S in one reading, but also
   evidence that the "matched control" perturbation is not fully behaviorally neutral (a
   large-enough unrelated activation perturbation can disrupt, not just fail to replicate, the
   target effect) — a nuance the report does not surface.
4. **Persisting sequence-inspectability gap (carried over from C2).** At α=16, S shows the same
   valid-ORF dip (0.667) + pLDDT rise (73.49) + sheet collapse (0.0034) pattern flagged for C2 as a
   possible degenerate-sequence-collapse signature, and M3's per-sample records again omit raw
   sequence text. The report's own "not degradation-gaming" argument (matched-control has better
   ORF yet no helix rise) rebuts only a coarse "any degradation raises helix" alternative — it does
   not address the sharper concern that S specifically induces a distinctive low-complexity regime
   that a structure predictor over-confidently folds as helical.

### F. Evaluation Type: synthetic_proxy
Same ESMFold+DSSP proxy readout as C2.

## Action Items
- Do not report "C3: SUPPORTED" as a full double dissociation; the β-sheet-specific arm did not
  materialize. Reframe as "S vs matched-control specificity established (p=3.2e-23); β-sheet
  off-target arm inconclusive/weak — full double dissociation not demonstrated" or re-run with a
  more potent β-sheet feature / higher β-sheet doses.
- Fix `aggregate()`'s `plddt_mean` to report the gate-restricted mean (matching the population
  backing helix/sheet means), and separately report the unguarded raw mean if useful for diagnostics.
- Report the dissociation trend across all four grid doses, not only the α=16 snapshot.
- Same as C2: preserve raw generated sequences at α=16 (all three feature kinds) to allow a direct
  check for low-complexity/degenerate collapse specifically under S vs the two controls.
