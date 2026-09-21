# Mechanism Audit — C4 (Compositional steering)

**Claim scope**: Milestones M9_C4_second_pair, M10_C4_combo_sweep, M11_C4_holdout  
**Mechanism family**: Representation and Parameter Analysis / activation-steering (RFM concept-vector, supervised)  
**Auditor**: verify Stage 1, Phase 2  
**Date**: 2026-07-15  

## Check A — Steering Coefficient Sweep

**Status: WARN**

**Dev sweep**: 3×3 alpha grid (alpha1, alpha2 ∈ {1, 2, 3} — 9 cells per combo on 5 dev prompts). This is a coarse grid covering the moderate range. No negative alpha tested (appropriate for compositional positive-direction steering). The grid is smaller than C1's 7-point signed sweep, but the 3×3 joint design is appropriate for a 2-vector interaction.

**Alpha_star selection**: alpha_star=(1.0, 1.0) selected for both combos — this is the joint argmax of the sum rubric on dev prompts. With only 5 dev prompts, the alpha selection is extremely noisy.

**Block vectors**: B5_pin_block14 extracts refusal_neg, formal_tone, cpp_python, technical_persona all at pinned block=14. Honesty at block=15 (from B1). Mixed blocks across v1/v2 in combos (combo1: block 15 for honesty, block 14 for refusal_neg; combo2: both block 14). This is acceptable since the vectors are extracted at each concept's chosen block and added together. The blocks are close enough (14 vs 15) that interference between the two additive interventions is plausible but is part of the experimental design.

**Random-direction control**: Required by the EXPERIMENT_PLAN.md cross-claim controls ("matched-random-direction control (C1, C2, C4)"). However, the random-direction control is NOT present in the C4b_pin_block14 or C4_compositional run outputs. The plan required it but it was not executed for C4.

This is a WARN: the matched-random-direction control for C4 is missing. The success predicate comparison (sum > single-v baseline) partially substitutes for the random control (comparing sum to single-vector conditions is a form of compositional specificity test), but it does not address the magnitude-alone confound.

## Checks B–F — Reserved

**Status: not_implemented**

## Overall Verdict

**overall_verdict: WARN**

Missing random-direction control (required by plan's cross-claim controls for C4). The 3×3 alpha sweep on 5 dev prompts is coarse and noisy. Mixed-block vectors (block 14 vs 15) are acceptable but introduce complexity. The ceiling effect noted in EXPERIMENT_AUDIT.md is also relevant here — the intervention at (alpha1, alpha2)=(1,1) is modest, and the ceiling at rubric=5.0 precludes any observable compositional gain.

The WARN is driven by the missing random-direction control. The not-supported verdict is still credible (ceiling effect is the dominant explanation, not a mechanism failure), but the missing control is a methodological gap.
