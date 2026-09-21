# Mechanism Audit Report — Claim C2

**Date**: 2026-07-15
**Auditor**: executor trigger detection (early N/A exit — no LLM reviewer call needed)
**Project**: Sparse Modular Circuit for Propositional-Logic Reasoning
**Claim**: C2 — modular decomposition (fact / rule / answer)
**Linked milestones**: M4, M4.stab

## Overall Verdict: N/A

*C2/M4/M4.stab use replacement activation patching — single-component clean-into-corrupted patch, `z_new[:, :, head_idxs, :] = clean_z_tensor[...]`. No additive scalar steering coefficient `h += alpha * direction` pattern found in `scripts/role_dissociation.py` or `scripts/prop_circuit_lib.py`. Check A (steering coefficient sweep) does not apply to replacement activation patching. No scalar alpha is selected or tuned.*

## Integrity Status: n/a

## Checks

### A. Steering Coefficient Sweep: N/A
**Trigger check**: mechanism type = replacement activation patching (clean activations patched onto corrupted-prompt run at each shortlisted component, one at a time). No scalar alpha coefficient exists in the M4/M4.stab pipeline. The S-matrix entries S[c, r] are patch recovery scores — a function of the patched activations' effect on the answer logits — not a function of any tuned scalar. Check A is not applicable to this intervention type.

### B–F. Reserved Checks: not_implemented
Placeholders for future mechanism-rigor checks. Not applicable to this pipeline.
