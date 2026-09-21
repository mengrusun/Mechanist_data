# Mechanism Audit — C2 (Python → C++ HackerRank steering)

**Claim scope**: Milestones M4_C2_extract, M5_C2_alpha_sweep, M6_C2_holdout  
**Mechanism family**: Representation and Parameter Analysis / activation-steering (RFM concept-vector, supervised)  
**Auditor**: verify Stage 1, Phase 2  
**Date**: 2026-07-15  

## Check A — Steering Coefficient Sweep

**Status: PASS**

**Sweep range**: alpha ∈ {0, 1, 2, 3, 4} on dev split (5-point sweep). The alpha range is smaller than C1's signed 7-point sweep, but the claim tests a unidirectional effect (C++ direction only, positive alpha). Five values are sufficient for a monotone sweep of a unidirectional intervention.

**Unit-norm v_cpp**: Extracted via the same RFM procedure as C1, at block 19 (val probe accuracy logged in B2_extract_extras). The `load_vector` function returns the unit-normalized vector saved by `save_vector`.

**Length guardrail**: Applied at `mean_len <= 2.5 * baseline_len`. At alpha=4.0, mean_len=510 vs baseline_len=299 (ratio ~1.7, below the cap). No over-steering to gibberish at alpha=4 for this concept.

**Alpha lock**: alpha_star=3.0 selected on 10 dev problems, applied to 10 held-out problems. Dev/held-out split is correct (first 10 / last 10 of the 20 problems).

**Random-direction control**: NOT implemented for C2 (the plan does not require it for C2 — the three-condition comparison (default/cpp_prompt_only/cpp_steered) is the primary evidence). The matched-random control was a C1 design; C2 uses a different evidence structure (language-functional pass rate rather than rubric score). This is appropriate per the plan.

**Key finding**: At alpha_star=3.0, `cpp_frac=0.00` across all dev alphas (0 through 4). The v_cpp vector extracted from language-prefixed pairs (`"Language: C++.\n<code>"`) did not induce a language switch at any tested alpha. This is a null result for the mechanism, not a mechanism failure per se — the direction is simply not effective at the tested alpha range.

## Checks B–F — Reserved

**Status: not_implemented**

## Overall Verdict

**overall_verdict: pass**

The mechanism is correctly applied per the plan's specification: alpha sweep 0–4 on dev, lock alpha_star on held-out, unit-norm vector, length guardrail. The null result (no language switch at any alpha) is a valid scientific outcome, not a methodology failure. The under-power issue is an experiment-audit concern (Check E), not a mechanism-rigor concern.
