# Experiment Audit — C4 (Compositionality of concept vectors)

**Claim scope**: Milestones M9_C4_second_pair, M10_C4_combo_sweep, M11_C4_holdout  
**Run IDs**: C4b_pin_block14, B5_pin_block14  
**Auditor**: verify Stage 1, Phase 2  
**Date**: 2026-07-15  

## Check A — Ground-Truth Provenance

**Status: PASS**

GPT-4o-2024-11-20 as rubric judge — dual rubric per output (concept-1 and concept-2 independently), preventing cross-contamination between target effects. This is the appropriate design for a compositional evaluation.

## Check B — Score Normalization

**Status: WARN**

Both combo rubrics are on a 5-point scale. The success predicate requires sum > v2_only on rubric-1 AND sum > v1_only on rubric-2. When single-vector conditions already saturate at rubric ≈ 5.0 (the maximum), the predicate is structurally unachievable regardless of compositional gain. 

This is a measurement ceiling problem: the rubric ceiling at 5.0 means the test is uninformative when single vectors saturate it. The task.md notes this: "single-vector ceiling ~5.0 on both combos — this is a measurement-ceiling failure. Was the rubric prompt appropriate? Would a harder held-out set have avoided ceiling?" The experimental design is methodologically sound but the chosen prompts and rubric are too easy, producing a ceiling effect that prevents the compositional test from being informative. This is a WARN on scope/evaluation design.

## Check C — Result File Existence (claim-scoped)

**Status: PASS**

- `runs/C4b_pin_block14/summary.json` — exists with complete per-combo results (dev_scored, held_agg, alpha_star, v1_meta, v2_meta, verdict).
- Combo1 held_agg: v1_only (n=15), v2_only (n=15), sum (n=15) — all populated.
- Combo2 held_agg: all at ceiling — all populated.

The original C4_compositional run also exists (for block-0 vectors, which showed gibberish). C4b used pinned block-14 vectors (B5_pin_block14), fixing the gibberish problem.

## Check D — Dead Code

**Status: PASS**

The composite rubric evaluation (judging each output twice — once for concept-1, once for concept-2) is implemented in `c4_compositional.py`. The dual-rubric design is in the active execution path.

## Check E — Scope (claim-scoped)

**Status: WARN**

The claim requires compositional gain on "≥2 concept vectors." Two combos tested, both fail the strict predicate due to ceiling effect. The held-out size is 15 prompts per combo (plan says 20; n_dev=5 subtracted from 20-prompt file). 

Task.md notes: "single-vector ceiling ~5.0 on both combos — measurement-ceiling failure." The test was run correctly per the plan's design, but the design itself is underpowered for detecting compositional effects on easy prompts. This is a limitation of the experimental design, not a methodology flaw in execution.

## Check F — Evaluation Type

**Status: PASS**

synthetic_proxy (GPT-4o dual rubric). Appropriate for open-ended compositional steering evaluation.

## Overall Verdict

**overall_verdict: WARN**

Experimental execution is correct. The ceiling effect is a known limitation of the test design, properly documented in EXPERIMENT_RESULTS.md and claims_ledger. Not a red flag for the integrity of the negative result — the result is "not-supported due to ceiling, not due to composition failure." The distinction matters for interpretation.
