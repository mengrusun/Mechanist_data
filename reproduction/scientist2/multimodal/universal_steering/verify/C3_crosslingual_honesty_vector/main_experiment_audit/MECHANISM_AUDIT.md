# Mechanism Audit — C3 (Cross-lingual transferability)

**Claim scope**: Milestones M7_C3_translate, M8_C3_steer_and_judge  
**Mechanism family**: Representation and Parameter Analysis / activation-steering (RFM concept-vector, supervised)  
**Auditor**: verify Stage 1, Phase 2  
**Date**: 2026-07-15  

## Check A — Steering Coefficient Sweep

**Status: PASS**

C3 reuses the v_honesty vector and alpha_star=+3.0 locked from C1. The alpha is not re-swept for C3 (by design — C3 tests cross-lingual transfer of the same fixed intervention). The plan explicitly states "at the C1-chosen alpha" (no new sweep). This is appropriate: the claim is about transfer, not about finding a new optimal alpha.

Alpha=+3 is the correct direction for "more honest" (from C1 summary.json: honesty at alpha=+3 mean=4.40 > baseline 4.26 = +0.14 shift, positive direction). C3 correctly uses alpha=+3 (alpha_star=3.0 in C3 config), independent of the C1 alpha_star labelling bug (which stored alpha_star=-3 due to argmax(|delta|) picking the negative direction). The C3 script hardcoded the intended positive alpha (+3), not the stored alpha_star.

The random-direction control is not used for C3 (not required by plan — C3 is a transfer test, not a new steering efficacy test).

The per-block screen is not re-run for C3 (reuses block=15 from C1). This is correct: the claim is about the same English-extracted vector working on foreign-language prompts.

## Checks B–F — Reserved

**Status: not_implemented**

## Overall Verdict

**overall_verdict: pass**

C3's mechanism setup is correct: fixed alpha=+3 from C1 (correctly hardcoded despite C1's alpha_star labelling bug), block=15 from C1's honesty probe, same v_honesty vector. No mechanism-rigor failure. The null result (no p<0.05) is a valid scientific outcome reflecting the weakness of the underlying honesty signal.
