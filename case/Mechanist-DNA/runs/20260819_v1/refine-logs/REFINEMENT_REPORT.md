# Refinement Report

**Date**: 2026-08-19 | **Mode**: reproduction (behavior=given, mechanism=given, resource_fidelity=strict)

## Problem Anchor (frozen)
Evo2-7B steered via amplification of Layer-26 Mixed SAE α-helix features → generated DNA with higher predicted α-helical content. The behavior, mechanism, model, SAE, and the 3 claims are given and were NOT modified; refinement targeted only the testing method.

## What moved vs. what stayed
- **Stayed (anchor):** the 3 claims (C1/C2/C3), the SAE mechanism, Evo2-7B, the Layer-26 Mixed SAE, strict full-scale resources.
- **Moved (testing method only):** added preregistration discipline — frozen numeric criteria, homology-clustered leakage-free splits, a single exact TopK-SAE steering definition with a portable α scale, treatment-independent QC with ITT+conditional reporting, dev/held-out seed separation for α*, cluster-level statistics, and specificity controls.

## Dominant contribution (retained)
First dose-controlled, quality-controlled, specificity-checked demonstration that an interpretable SAE feature set causally raises α-helix (a structural phenotype) in a DNA LM's generations, on Evo2-7B via its released SAE, with a reported optimal α*.

## Complexity intentionally rejected
No SAE retraining, no fine-tuning/weight editing, no optimized/gradient steering vector, no smaller model or data subset. Simplest adequate mechanism = fixed residual-add of selected features' decoder directions.

## Risks still open (documented, mitigated)
1. Predicted-vs-experimental readout (ESMFold→DSSP) — mitigated by dual-confidence reporting + controls; phrased as "predicted".
2. ESMFold weights offline availability — mitigated by local SS-predictor fallback (eval-tool choice; strict fidelity preserved).
3. Compute headroom on folding throughput — mitigated by budget guard that trims generation N to floors, never model/SAE scale.

## Gate check (Phase 2)
- Final thesis: crisp (see FINAL_PROPOSAL thesis).
- Dominant contribution: crisp.
- Rejected complexity: explicit.
- Reviewer concerns for validation: enumerated in REVIEW_SUMMARY and mapped into PLAN.
- Frontier primitive: the SAE is central and given (not optional); no additional frontier primitive forced.

**Verdict: READY** to proceed to implementation.
