# Mechanism Audit Report — Claim C2

**Date**: 2026-07-13
**Auditor**: executor (early N/A exit — no reviewer call needed)
**Project**: Steerable Social-Variable Directions in an LLM Dictator
**Claim**: C2 — Purity via decorrelation (GS and LEACE produce cross-leakage-free pure directions)
**Linked milestones**: M3

## Overall Verdict: N/A

*C2's milestone M3 performs direction purification via closed-form linear operations (Gram-Schmidt orthogonalization, LEACE projection matrix), followed by probe evaluation on held-out activations. No additive activation intervention with a scalar coefficient α is applied in M3 — it is a Location milestone (fitting and evaluating pure directions), not a Causal Intervention milestone. The steering coefficient sweep check (Check A) is not triggered. Downstream combination treats N/A as PASS.*

## Triggered checks (this run): (none)

## Checks

### A. Steering Coefficient Sweep: N/A
- Triggered: no — no catalogue trigger matched in C2's scope (M3)
- Reason: M3 (`scripts/m3_decorrelate.py`) performs:
  1. Gram-Schmidt orthogonalization of raw directions against each other (pure-algebraic, no model forward pass, no alpha)
  2. LEACE closed-form projection to obtain ṽ_V^LEACE (fit on train activations, applied to v̂_V — purely linear)
  3. 4×4 cross-leakage evaluation via 1-D scalar-projection logistic probe on held-out activations
  4. Preservation-of-self check and norm audit
- There is no `h ← h + α · ṽ_V` pattern anywhere in m3_decorrelate.py (confirmed by code inspection). The LEACE purification is a projection operation (`v_erased = eraser(v.unsqueeze(0)).squeeze(0)`), not a model-level steering intervention.
- Check A triggers only when an additive activation intervention with scalar coefficient α is used. M3's evaluation is purely representational (direction-fitting + probe), not interventional.

### B–F. Reserved (not_implemented)
Status: not yet implemented. Future checks may cover direction-extraction quality, site/layer choice, n_effective sufficiency, probe-vs-causal disentanglement, intervention scope.

## Action Items
- None required (N/A verdict; no mechanism intervention in C2's scope).
