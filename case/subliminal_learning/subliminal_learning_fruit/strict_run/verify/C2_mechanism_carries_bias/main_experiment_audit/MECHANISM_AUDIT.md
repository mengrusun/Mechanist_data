# Mechanism Audit Report — Claim C2

**Date**: 2026-07-19
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP)
**Project**: Subliminal Learning in Diffusion Image Models (Qwen-Image)
**Claim**: C2 — Some identifiable internal component-kind of the DiT / MMDiT causally carries the transferred bias
**Linked milestones**: M1, M2

## Overall Verdict: WARN

*C2 uses a forward-hook steering intervention on `transformer_blocks[50]` in `src/mechanism/intervene_and_eval.py`. This is a catalogue-triggering additive activation intervention with a scalar coefficient. The mechanism audit Check A (Steering Coefficient Sweep) applies. The implementation correctly calibrates σ_proj per site, uses σ_proj-scaled amplification (`k × sigma × v̂`), includes a matched-random specificity control, and preserves sign consistency. However, the dose-response curve is a single point (amplify_x3 only), not the ≥3 orders-of-magnitude sweep required for a complete steering coefficient sweep. This is a documented sweep-cardinality gap — the planned grid included amplify_x{2,3,4} but only amplify_x3 was executed. The gap is WARN, not FAIL: the σ_proj calibration and matched-random control are implemented correctly and mitigate the most dangerous forms of mis-specification; the missing amplify_x{2,4} points reduce confidence in the dose-response monotonicity claim but do not invalidate the ablation and specificity results.*

## Integrity Status: warn

## Triggered checks (this run): Check A (steering_coefficient_sweep)

## Checks

### A. Steering Coefficient Sweep: WARN

**Trigger**: `src/mechanism/intervene_and_eval.py` registers a forward hook on `transformer_blocks[50]` that scales `img_h` by additive term `k × sigma × v̂` (amplify modes) or subtracts a projection (ablate/random_ablate) — this is an additive activation intervention with a scalar coefficient.

**Rubric assessment**:

1. **σ_proj scaling** ✅ — IMPLEMENTED. `_estimate_sigma()` runs the model on `sigma_calib_n=4` prompts and computes `std(projection_onto_direction)`. The amplification formula `k × sigma × v̂` is exactly the σ_proj-scaled convention that makes coefficients comparable across layers and models. This is the most important safeguard — correctly implemented.

2. **Sweep cardinality ≥3** ❌ — GAP. The plan committed to `amplify_x{1,3,5}` at minimum (per-plan grid: amplify_x2, amplify_x3, amplify_x4). The realized runs include only `amplify_x3`. A single amplification point is a 1-point dose curve — it shows direction (whether amplifying increases P(banana)) but cannot support the planned Spearman-rho ≥ 0.9 monotonicity test. This is the primary gap driving the WARN verdict.

3. **Capability/coherence metric logged alongside** ✅ — `fluency = fruit_n / n` is logged in every verdict.json. Fluency stays 0.80–0.98 across all M2 runs, confirming the model remains in-distribution at the tested amplification level.

4. **α locked mid-plateau** — NOT EVALUABLE. With a single amplification point there is no plateau to identify. The plan's intent was to sweep and lock at the plateau; this requirement cannot be assessed from a 1-point curve. Not-evaluable is treated as WARN contribution (the sweep cardinality gap is the root cause).

5. **Matched-random direction baseline** ✅ — IMPLEMENTED. `random_ablate` runs a `matched-magnitude Gaussian rank-1 direction drawn from the same-norm Gaussian in the ΔW subspace` via `seed_rand=42`. The result (Δ_random_mean = -0.019pp, indistinguishable from Δ_ablate_mean = -0.015pp) is a valid specificity control — its output happens to show the effect is non-directional, which is a scientifically correct finding (delocalized), not a methodology failure.

6. **Sign pattern preserved** ✅ — amplify_x3 adds `+k × sigma × v̂` (positive direction push); ablate projects out the same direction. Signs are consistent with the direction extraction at M1 (top-1 left singular vector of mean-teacher-arm ΔW).

**Summary**: 4/6 sub-criteria pass; 1 gap (cardinality); 1 not-evaluable due to cardinality gap. The σ_proj calibration and matched-random control are the most scientifically critical checks and both pass. The sweep-cardinality gap is a known limitation documented in EXPERIMENT_RESULTS.md as a scope reduction. The finding itself (delocalized carrier) is consistent across the 4 tested seeds and is not contradicted by missing amplification points.

**Grade: WARN** (not FAIL) — the 1-point dose curve reduces confidence in the monotonicity claim but the mechanistic interpretation (block-50 intervention too narrow, carrier delocalized) is supported by the ablation and specificity data together. A FAIL grade would apply if: (a) the σ_proj calibration were absent (it is present), (b) the matched-random control were absent (it is present), or (c) the amplification were in raw-activation units without any σ normalization (it is σ-normalized). None of these apply.

### B–F. Reserved (not_implemented)
Status: not yet implemented. Future checks may cover direction-extraction quality, site / layer choice, n_effective sufficiency, probe-vs-causal disentanglement, intervention scope.

## Action Items
- **Sweep cardinality gap (WARN)**: Run amplify_x2 and amplify_x4 alongside amplify_x3 on the full 8 seeds in a follow-up round. This will enable the Spearman-rho ≥ 0.9 monotonicity check. If the result remains near-zero movement at all three amplification points (consistent with the delocalized finding), it further supports the multi-block-intervention follow-up as the correct next step.
- **Plateau locking**: Not actionable until sweep cardinality ≥ 3 is achieved.
