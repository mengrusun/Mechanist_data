# Robustness Report — C1 (Localizability per-emotion component sets C_e)

**Terminal state**: INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap)
**Main-experiment verdict**: supported
**Baseline integrity (Phase 2)**: PASS with WARN

---

## Claim Statement

The framework produces stable, sparse, per-emotion component sets C_e on Llama-3.2-3B-Instruct across SEV:
|C_e| meets sparsity floor (k_h <= 96, k_n <= 8000); per-emotion Jaccard exceeds size-matched
permutation-null 95% CI on >= 5/6 emotions.

## Main Experiment Evidence

- (k_h*, k_n*) = (24, 2000) — the grid minimum. sparsity_floor_met: true.
- Stage-B macro gain: 2.136 nats (well above 0.01-nat warning floor).
- Per-emotion Jaccard: heads 0.947-1.000 vs. null CI-hi 0.265-0.277 (ratio ~3.5-4x); neurons 0.972-0.983
  vs. null CI-hi 0.045-0.047 (ratio ~21x). All 6/6 emotions pass on both axes.
- Random top-k control: within null CI as expected.

## Stage 2 Status

**INTEGRITY_ONLY — stage2_skip_reason: max_verify_claims_cap**

C1 was admitted to the Stage-1 pool (baseline integrity PASS) but was not selected for Stage 2 (swap variants)
because max_verify_claims=1 and C3 was selected as the top-importance claim. C1 can be independently
verified via `/auto-verify C1 --resume=true` to run a model-swap variant (e.g., Qwen2.5-7B-Instruct).

## Robustness Score

**Not computed** (Stage 2 not run for this claim).

## Integrity Warnings (WARN, not FAIL)

1. **Flat kstar grid**: All 9 (k_h, k_n) grid cells yield identical macro_gain = 2.136 nats. The kstar
   selection was not discriminative — (k_h*, k_n*) = (24, 2000) by grid-order tie-breaking, not by
   a superior val signal. The sparsity floor claim is technically satisfied, but the selection narrative
   (smallest grid cell yielded best gain) is weakened by the tie. Root cause: the multi-component hook
   at k_h=24 and k_h=48 from the same layers produces near-identical aggregate layer deltas.

2. **Head Stage-B enhancement is layer-level not head-level**: The head-enhancement in Stage B adds
   alpha * d_{e,L} to the LAYER residual, making all heads at the same layer appear equally valuable
   to the causal ranker. The Stage-A (probe AUC) selection IS head-specific; only the Stage-B reranking
   is degenerate at head level.

## Upgrade Command

To run Stage 2 swap variants for C1:
```
/auto-verify C1 --resume=true --dimensions=model --gpu_id=1,2,3,5,6
```
