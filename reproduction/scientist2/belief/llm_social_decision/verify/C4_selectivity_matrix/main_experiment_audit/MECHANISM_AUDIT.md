# Mechanism Audit Report — Claim C4

**Date**: 2026-07-13
**Auditor**: executor (cross-model self-review; llm-chat MCP degraded gracefully)
**Project**: Steerable Social-Variable Directions in an LLM Dictator
**Claim**: C4 — Selectivity (4×4 selectivity matrix — steering one variable does not measurably move the effects of the other three)
**Linked milestones**: M5

## Overall Verdict: WARN

## Integrity Status: warn

C4's mechanism (M5) uses the same CAA additive activation steering as M4 — it reuses M4's grid at the fixed (α ∈ {−2,0,+2}σ, LEACE, single-site, ell_V*) sub-grid. Check A triggers. The α-sweep observations from M4 carry over directly: the σ_proj-scaling is correctly implemented, bidirectional α is tested, and coherence is logged. The key concerns are identical to C3's mechanism audit: no random-direction control at the intervention points, and the under-power at ell_V* means the steering results are indistinguishable from random perturbations (confirmed by B2 in M6). However, since C4 measures W's off-diagonal effects (not V's diagonal), the B2 finding that B2≈raw at ell_V* is somewhat less critical for C4's evaluation — the under-power manifests as V=G diagonal being zero (which correctly gets reported as a C4 failure) rather than as a false-positive selectivity result.

## Triggered checks (this run): A — Steering Coefficient Sweep

## Checks

### A. Steering Coefficient Sweep: WARN

**Trigger**: M5 reuses M4's run_one_grid_point() calls via the inline summary extraction. The same `SteeringHook` (`h ← h + alpha * unit_dir`) applies. Check A triggered.

**Sub-checks:**

1. **α swept across ≥ 3 orders of magnitude**: M5 uses α ∈ {−2,0,+2}σ — only 3 points (one non-zero per sign). The M4 main grid covers 7 points including ±4σ for the same (V, LEACE, single) sub-grid. M5 selects only the α=+2σ canonical entry for the matrix (line 465: `if alpha_mult == 2: m5_matrix[V][W] = shift`). The sweep underlying the M5 result is from M4's 7-point grid (via summary); M5 itself only extracts the +2σ entry. The sweep adequacy inherits from M4.
   - **WARN**: Same as C3 — only ±2σ and ±4σ (the pre-registered endpoints for saturation) at ell_V*. The M5 canonical matrix uses only +2σ.

2. **σ_proj-scaled**: Inherited from M4. PASS.

3. **Coherence gate logged**: Inherited from M4's run_one_grid_point(). PASS.

4. **Plateau lock**: Same as C3. No formal plateau identification. WARN.

5. **Random-direction control for selectivity evaluation**:
   - B2 random-direction control (M6) provides `steer_{V}_random_a+2_{s}` at ell_V*. The B2 finding shows that v_effect under random direction ≈ v_effect under extracted direction at ell_V*. This has a specific implication for C4: if the random direction produces similar v_effects on baseline transfers, it may also produce similar w_effects — meaning the C4 selectivity matrix entries may be equally random for a random direction.
   - However, the random-direction w_effects were not separately computed in M6 (M6 files contain v_effect and w_effects — checking: `steer_G_random_a+2_s0.json` has w_effects dict). The random-direction w_effects at ell_V* for G: G=-0.130, A=-0.139, I=1.535, M=0.465. These are very similar to the zero-alpha baseline w_effects (from M4's α=0 run: G=-0.380, A=0.106, I=1.287, M=0.713). This is consistent with the under-power finding: the random direction at α=+2σ also produces a non-zero w_effect for I and M (1.54 and 0.47), but this is the baseline model's behavior showing through (the injection is too small to change the model's output, so w_effects just reflect the natural marginal effects of W).
   - **WARN**: Same gap as C3 — no random-direction control at L=16 for the selectivity measurement. The C4 matrix at picked layers correctly shows V=G produces zero w_effects (confirming under-power), but the V=A, V=I, V=M off-diagonal entries are not validated against random-direction w_effects at the same injection magnitude.

6. **Bidirectional sign**: α ∈ {−2,0,+2}σ. Bidirectional. PASS.

**Overall Check A verdict**: WARN (same structural issues as C3, but for C4's usage the main concern is the missing random-direction W-effect control at ell_V* and absent L=16 data for C4)

### B–F. Reserved (not_implemented)
Status: not yet implemented.

## Action Items
1. [WARN] Add explicit random-direction w_effects comparison in M6 results to confirm that the V=A uniform off-diagonal pattern (−0.44 to −0.46 across G, I, M) is not better explained by a global mean shift from the random direction at the same α magnitude at ell_V*=6.
2. [WARN] Run the C4 selectivity matrix at L=16 (deferred to iteration loop per EXPERIMENT_RESULTS.md) to complete the selectivity picture at the causally-loaded layer.
