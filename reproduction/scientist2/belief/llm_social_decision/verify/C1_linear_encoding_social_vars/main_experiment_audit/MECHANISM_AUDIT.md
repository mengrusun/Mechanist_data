# Mechanism Audit Report — Claim C1

**Date**: 2026-07-13
**Auditor**: executor (mechanism-audit skill; early N/A exit — no reviewer call needed)
**Project**: Steerable Social-Variable Directions in an LLM Dictator
**Claim**: C1 — Linear encoding of each social/contextual variable (G, A, I, M)
**Linked milestones**: M2

## Overall Verdict: N/A
*This is C1's mechanism-rigor verdict. N/A means no catalogue check was triggered for C1's scope — C1's milestone (M2) uses no additive activation intervention with a scalar coefficient. M2 extracts directions via paired difference-of-means and evaluates them via linear probing and OLS regression. The CAA-style intervention (α · ṽ_V injection) is applied only in M4 (C3's milestone), not in M2 (C1's milestone). Downstream combination treats N/A as PASS.*

## Triggered checks (this run): (none)

## Checks

### A. Steering Coefficient Sweep: N/A
- Triggered: no — no catalogue trigger matched in C1's scope (M2)
- Reason: M2 (`scripts/m2_extract_and_probe.py`) performs:
  1. Forward passes on 800 train paired-partner prompts → extract residual activations
  2. Compute `v_hat_V = mean(h(p') - h(p))` per variable per layer (direction extraction)
  3. 5-fold logistic regression probe on train activations → cv_acc + held_acc
  4. Projection-transfer OLS: regress tau onto projection of held-out activations onto direction
  5. Baseline transfer decode (greedy, NO hooks/steering applied)
- The `greedy_transfer` function at line 405 is called WITHOUT `apply_hooks` (no scalar multiplication onto activations). No `alpha * direction` pattern exists in m2_extract_and_probe.py (confirmed by grep: no match).
- Check A triggers only when an additive activation intervention with scalar coefficient α is used. M2's evaluation is purely observational (decoding + regression), not interventional.

### B–F. Reserved (not_implemented)
Status: not yet implemented. Future checks may cover direction-extraction quality, site / layer selection, n_effective sufficiency, probe-vs-causal disentanglement, intervention scope.

## Action Items
- None required (N/A verdict; no mechanism intervention in C1's scope).
