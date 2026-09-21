# Mechanism Audit Report — Claim C3

**Date**: 2026-07-15
**Auditor**: executor trigger detection (early N/A exit — no LLM reviewer call needed)
**Project**: Sparse Modular Circuit for Propositional-Logic Reasoning
**Claim**: C3 — necessity + sufficiency (path patching + reinsertion)
**Linked milestones**: M2, M3

## Overall Verdict: N/A

*C3/M2/M3 use replacement activation patching — clean activations are patched directly onto corrupted-prompt runs (M2, necessity) or clean-prompt runs with all other components resample-ablated (M3, sufficiency). No additive scalar steering coefficient `h += alpha * direction` pattern is found in `scripts/path_patch_necessity.py`, `scripts/reinsertion_sufficiency.py`, or `scripts/prop_circuit_lib.py`. Check A (steering coefficient sweep) does not apply to replacement activation patching.*

## Integrity Status: n/a

## Checks

### A. Steering Coefficient Sweep: N/A
**Trigger check**: M2 (path patching) and M3 (reinsertion sufficiency) both use direct activation replacement — `clean_z_tensor[...]` values are written to the hooked positions on the corrupted (or ablated) run. No scalar alpha is tuned, swept, or applied. The intervention is binary: either the clean activation is inserted or it is not. Check A requires additive intervention of the form `h += alpha * direction` with alpha swept across ≥ 3 orders of magnitude — this pattern is not present.

### B–F. Reserved Checks: not_implemented
Placeholders for future mechanism-rigor checks. Not applicable to this pipeline.
