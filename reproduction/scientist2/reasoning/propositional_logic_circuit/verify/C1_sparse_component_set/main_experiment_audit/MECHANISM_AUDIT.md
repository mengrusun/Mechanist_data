# Mechanism Audit Report — Claim C1

**Date**: 2026-07-15
**Auditor**: executor trigger detection (no reviewer call — early N/A exit)
**Project**: Sparse Modular Circuit for Propositional-Logic Reasoning
**Claim**: C1 — "A sparse subset of specific attention heads and MLP components jointly implements the minimal propositional-logic reasoning task — the circuit is small relative to the full model (target |shortlist|/|total| ≤ ~15% with completeness ≥ 0.9 and single-removal minimality drop ≥ 0.05 on Mistral-7B)."
**Linked milestones**: M1

## Overall Verdict: N/A
*This is C1's mechanism-rigor verdict. N/A means no catalogue check was triggered for C1's scope — the claim uses activation replacement patching (no scalar α multiplier), which is outside the current catalogue coverage for Check A (steering coefficient sweep).*

## Triggered checks (this run): (none)

## Checks

### A. Steering Coefficient Sweep: N/A
- Triggered: no — C1/M1 uses activation replacement patching (splice clean activations into corrupt run at shortlisted components: `z_new[:, :, head_idxs, :] = clean_z_tensor[...]`). There is NO scalar coefficient α multiplying an additive contribution to a hidden state. Grep of `scripts/attribution_screen.py` and `scripts/prop_circuit_lib.py` for `alpha`, `coeff`, `scale_factor` returns empty. Check A's trigger requires `activations += alpha * v` or equivalent scalar-multiplicative additive pattern — this is absent from C1's intervention code.

### B–F. Reserved (not_implemented)
Status: not yet implemented. Future checks may cover direction-extraction quality, site / layer selection, n_effective sufficiency, probe-vs-causal disentanglement, intervention scope.

## Action Items
None — N/A overall verdict indicates the mechanism lies outside the current catalogue coverage, not a rigor failure.
