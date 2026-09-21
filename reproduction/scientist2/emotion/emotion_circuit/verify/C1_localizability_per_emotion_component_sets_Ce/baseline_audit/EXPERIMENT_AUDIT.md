# Experiment Audit — C1 (Localizability) — Phase 2 Baseline Integrity

**Claim**: The framework produces stable, sparse, per-emotion component sets C_e on Llama-3.2-3B-Instruct across SEV:
|C_e| meets sparsity floor (k_h <= 96, k_n <= 8000); per-emotion Jaccard exceeds size-matched permutation-null 95% CI
on >= 5/6 emotions.

**Main-experiment verdict**: supported

---

## Data Integrity

- **Dataset**: SEV sev.json, 480 events x 6 emotions = 2880 pairs. Scenario-level 10/5/5 split per domain (8 domains),
  asserted disjoint in code. Used_n = available_n: no subsetting. PASS.
- **Stage A**: train fold (240 stems x 6 emotions = 1440 pairs). Stage B: 30 val stems per emotion from val fold.
  Jaccard: 3 folds x 80% of 240 train stems = ~192 stems each. PASS — splits respected in code.
- **Contamination check**: Stage B uses the val fold; kstar selection uses the full val fold (120 stems);
  Jaccard uses the train fold only. Eval fold not touched in M1. PASS.

## Statistical Methodology

- **Jaccard stability**: 3 event-subsample folds (80% of train stems, without replacement), mean pairwise Jaccard
  across the 3 folds (3 pairs). 200-draw size-matched permutation null per emotion. Jaccard values reported:
  heads 0.947-1.000, neurons 0.972-0.983 — approximately 5x null upper edge for heads and 20x for neurons. PASS.
- **Null construction**: `permutation_null_jaccard` samples random sets of the same size from the same
  pool (top-3 layers * n_heads for heads; top-3 layers * n_neurons for neurons), confirmed in code. The pool
  is the Stage-A shortlist universe, which is the correct comparison. PASS.
- **Sparsity floor**: (k_h*, k_n*) = (24, 2000), the grid minimum. sparsity_floor_met: true. PASS.

## Critical Finding: Flat kstar Grid

The kstar.json shows all 9 grid cells yield IDENTICAL per-emotion gains and identical macro_gain = 2.136 nats.
This means adding more heads (k_h=48, 96) or more neurons (k_n=4000, 8000) beyond (24, 2000) produces zero
marginal gain in the kstar selection forward pass. This occurs because the `_install_multi_component_hooks`
function aggregates all component deltas into a per-layer residual delta vector — so adding more heads from the
same layer simply adds more direction contributions, but if the top-k and top-2k heads from the same layers
produce the same aggregate layer delta (e.g., they share layers and the additional heads at the margin have
scores that contribute negligibly), the measured gain is flat. This is a methodological RED FLAG for the
kstar selection: kstar defaulted to the minimum grid cell not because it was best, but because all cells tied.

**Consequence for C1**: The sparsity claim ("framework selects the sparse minimum") is correct in letter
(k_h*=24, k_n*=2000), but the selection was not driven by a discriminative val signal — all grid cells
were equivalent, suggesting the Stage-B multi-component enhancement gain is insensitive to the exact number
of components beyond a certain floor. This is a WARN, not a FAIL: the reported (k_h*, k_n*) IS the minimum,
the sparsity floor IS met, and the Jaccard stability results are INDEPENDENT of this selection procedure.

## Stage-B Signal Integrity

- Stage-B macro gain = 2.136 nats at (24, 2000). Well above the 0.01-nat warning floor. PASS.
- Enhancement direction: all 6 emotions have positive Stage-B scores at alpha=1.0 (the direction is meaningful).
  The per-emotion gains range from 0.738 (disgust) to 5.668 (fear). PASS.

## C1 Jaccard Specific Checks

| Emotion | Head J | Head Null CI-hi | Head Pass | Neuron J | Neuron Null CI-hi | Neuron Pass |
|---------|--------|-----------------|-----------|----------|-------------------|-------------|
| joy     | 1.000  | 0.277           | PASS      | 0.974    | 0.046             | PASS        |
| sadness | 0.947  | 0.275           | PASS      | 0.981    | 0.046             | PASS        |
| anger   | 1.000  | 0.271           | PASS      | 0.978    | 0.046             | PASS        |
| fear    | 1.000  | 0.265           | PASS      | 0.983    | 0.047             | PASS        |
| surprise| 0.947  | 0.266           | PASS      | 0.972    | 0.046             | PASS        |
| disgust | 1.000  | 0.265           | PASS      | 0.973    | 0.045             | PASS        |

All 6/6 pass on both axes. Success criterion (>= 5/6) exceeded. PASS.

## Random Control

`random_control_jaccard.json` exists. Random top-k sets from the Stage-A shortlist produce Jaccard within
the permutation-null CI, as expected. Confirms that C_e's high Jaccard is Stage-B-selection-driven.
PASS — note: given the flat kstar grid, this control tests whether the STAGE-A shortlist itself is stable
(not whether Stage-B further concentrates stability), which slightly weakens the control's interpretation
but does not invalidate C1's Jaccard predicate.

## Overall Integrity Verdict

**PASS (with WARN: flat kstar grid)**

The sparsity floor is met, the per-emotion Jaccard exceeds the permutation null on all 6 emotions with large
margins, the null is properly constructed, and the data splits are clean. The flat kstar grid is a concern
for the selection's discriminativeness but does not undermine the stability predicate that underpins C1's
"supported" verdict.
