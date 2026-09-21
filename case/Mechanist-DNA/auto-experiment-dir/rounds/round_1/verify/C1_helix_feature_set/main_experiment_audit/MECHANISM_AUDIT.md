# Mechanism Audit Report — Claim C1

**Date**: 2026-07-18
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP) — not invoked (early N/A exit)
**Project**: Feature Steering an α-Helix Knob in Evo2-7B
**Claim**: C1 — In Evo2-7B's Layer-26 SAE there is a non-empty set of features whose activation selectively marks α-helix codons, beyond confounds and multiple-testing chance.
**Linked milestones**: M0

## Overall Verdict: N/A
*C1's linked milestone (M0) is a pure feature-discovery/discrimination gate (per-codon AUROC/F1
scoring of SAE features against real DSSP labels) — no additive activation intervention is applied
to Evo2-7B within M0's scope. The catalogue's only implemented check (A: steering coefficient sweep)
requires an additive intervention with a scalar coefficient; M0 performs none. C1's mechanism-rigor
gate is therefore not applicable and, per this skill's combination rule, N/A is treated as PASS.*

## Triggered checks (this run): (none)

## Checks

### A. Steering Coefficient Sweep: N/A
- Triggered: no. Grep of C1's scoped scripts (`code/m0_data.py`, `code/m0_scoring.py`,
  `code/m0_feature_selectivity.py`, `code/m0_consolidate.py`, `code/m0_build_dataset.py`,
  `code/m0_cache_acts.py`) for the catalogue's trigger regex found only comment/metadata matches on
  the word "steer" (e.g. `code/m0_consolidate.py:4,47,88,156,175` — "steering scales", "primary
  steer organism", `"steer_organism"`, `"used_relaxed_set_for_steering"`). These are M0
  precomputing per-feature scale values (`s_f`) and organism/config choices **for later use** by
  M1–M3's steering hook — no `act + alpha*s_f` additive update, and no scalar coefficient sweep,
  is executed anywhere inside M0's own code. The actual steering intervention (which IS subject to
  Check A) lives in `code/mechanism.py` / M2's dose-response sweep, which backs C2, not C1.

### B–F. Reserved (not_implemented)
Status: not yet implemented. Future checks may cover direction-extraction quality, site/layer
selection, n_effective sufficiency, probe-vs-causal disentanglement, intervention scope.

## Action Items
None — this claim's evidence (M0) is not mechanism-intervention based; its rigor is instead
evaluated by `/experiment-audit` (methodology honesty) and `/result-to-claim` (semantic support).
The mechanism sweep that DOES back this project (M2's α sweep) is properly in scope for C2's
mechanism-audit, not C1's.
