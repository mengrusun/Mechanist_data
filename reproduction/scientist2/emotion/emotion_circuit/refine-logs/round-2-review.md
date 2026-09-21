# Round 2 Review

**Date**: 2026-07-13
**Overall Score**: 8.4 / 10
**Verdict**: REVISE (but close)
**Drift Warning**: NONE (anchor preserved)

## Parsed Scores

| Dim | Score |
|---|---|
| Problem Fidelity | 9 |
| Method Specificity | 9 |
| Contribution Quality | 8 |
| Frontier Leverage | 8 |
| Feasibility | 8 |
| Validation Focus | 9 |
| Venue Readiness | 8 |
| **Overall** | **8.4** |

## Blocking issues remaining (must fix for READY)

### CRITICAL 1: Stage-B causal-ranking metric not fully pinned
The proposal says Stage B measures "Δ(target-emo score)" but doesn't say *what* the score is. This determines which components enter `C_e`, and thus everything downstream. Must specify: exact prompt format, score definition (judge-based / classifier-based / logit-based), aggregation over items.

### CRITICAL 2: Single-direction steering (Arm C) direction construction not fully locked
The proposal locks the tuning grid (3 layers × 3 α) but not *how the direction is constructed*. Standard single-direction steering critically depends on direction source and injection semantics. Must freeze pre-eval-split.

## Important remaining
- Off-target mean pool for ablation: exact definition (what if a stem has no valid off-target variant?).
- `k_h, k_n` selection: reaffirm globally-once vs. per-emotion (proposal already says shared grid; make it *explicitly* one setting per family across all 6 emotions).
- Explicit success/partial/failure rubric under mixed Claim-2 outcomes.

## Nice-to-have
- Demote Claim 2e (neuron overlap < head overlap) to explicitly secondary.
- Justify or drop the 15%-length threshold as reporting-only.

## Simplifications
- Rank-fusion may be unnecessary; use Stage B as the causal filter, Stage A as pure shortlist.
- Judge audit should read as a QA check, not a subsystem.

See `round-2-review.raw.md` for the verbatim reviewer response.
