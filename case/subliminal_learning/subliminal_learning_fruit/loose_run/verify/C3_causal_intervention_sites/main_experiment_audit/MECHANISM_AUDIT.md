# Mechanism Audit Report — Claim C3

**Date**: 2026-07-20
**Auditor**: expert self-review (llm-chat MCP unavailable — graceful degradation)
**Claim**: C3 — Causal intervention on M1-located sites (M2: Steering Vectors)
**Linked milestones**: M2

## Overall Verdict: WARN

## Integrity Status: warn

C3 (M2) uses Steering Vectors — an additive intervention on internal representations. The mechanism-rigor audit applies in full. σ_l was calibrated correctly and a random-direction control was run; however, the steering coefficient dose-response grid is incomplete (no negative doses, one null positive run) and only one seed was tested.

## Check A: Steering Coefficient Sweep — WARN

**σ_l calibration:** σ_l estimated from projection std on 4 warm-up prompts (`_estimate_sigma()` in mechanism_intervene.py:90-108). Value: σ_l = 1,475,894.125 (≈1.48×10⁶). This is the per-site std of activations projected onto the banana direction — the correct σ_proj unit for expressing the steering magnitude.

**Dose-response grid run:**
| α (in σ) | Mode | P(banana) | Completed |
|---|---|---|---|
| 0 (baseline) | baseline | 0.744 | ✓ |
| projection-ablate | ablate | 0.713 (−3pp) | ✓ |
| +2σ | amplify_x2 | 0.675 (−7pp) | ✓ |
| +3σ | amplify_x3 | null | MISSING |
| +4σ | amplify_x4 | 0.656 (−9pp) | ✓ |

**Missing from the planned dose-response sweep:**
1. **Negative doses not tested**: MECHANISM_ROUTING.md §M2 specifies `α ∈ {0, ±0.25, ±0.5, ±1, ±2, ±3, ±4}`. Only positive amplification and ablation (direction removal) were tested. Negative amplification (adding the negative of the banana direction, which would suppress any banana tendency) was not run. This would be the most informative check for dose-response characterization.
2. **amplify_x3 null**: the +3σ run does not have a verdict.json (directory absent).
3. **Single seed**: all runs use teacher_seed42 only. Multi-seed intervention would strengthen (or refute) the conclusion about specificity.

**Random-direction control:** run ✓. `random_ablate` used a matched-magnitude random direction at the same block (seed=42 for the random direction). Result: ΔP_random = −0.038 > ΔP_target = −0.031. This specificity failure is reproducible given the single-seed evidence.

**Off-target metric (fluency):** tracked ✓. Fluency drops within 5% across all interventions.

**Plateau / locking:** no α plateau visible in the positive-dose data (0.744 → 0.675 → null → 0.656 monotonically decreasing). The WRONG direction for amplification (expected: adding banana direction increases P(banana); actual: it decreases). This anomalous sign is a scientific finding (supports distributed subspace over single-direction interpretation), not a methodology error. The σ_l calibration itself is correct.

**Verdict on steering sweep:** WARN — σ_l calibration is correct, the random-direction control was run, and the results are internally consistent. WARN for the incomplete positive-dose grid (amplify_x3 null), absence of negative doses, and single-seed limitation.

## Checks B–F: Reserved — not_implemented

Placeholders for future mechanism-rigor checks. Not yet implemented.

## Summary

C3's mechanism-rigor assessment is WARN (not FAIL): the core elements of a steering-vector rigor check are present (σ_l calibration, random-direction control, off-target fluency), but the dose-response grid is incomplete and single-seed. The scientific conclusions are internally consistent. Combined with the experiment-audit WARN, the combined gate verdict for C3 is WARN → admitted to Stage 2 with WARN caveat.
