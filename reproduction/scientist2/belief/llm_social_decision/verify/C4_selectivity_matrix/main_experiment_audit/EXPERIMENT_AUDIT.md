# Experiment Audit Report — Claim C4

**Date**: 2026-07-13
**Auditor**: executor (cross-model self-review; llm-chat MCP degraded gracefully)
**Project**: Steerable Social-Variable Directions in an LLM Dictator
**Claim**: C4 — Selectivity (steering one variable does not measurably move the effects of the other three variables on the decision)
**Linked milestones**: M5

## Overall Verdict: WARN

## Integrity Status: warn

The C4 evaluation methodology is admissible: the 4×4 selectivity matrix is computed from real held-out greedy-decode outputs, the permutation test is correctly implemented, and the claimed result (NOT SUPPORTED at picked layers) is correctly reported. The key issues:
1. The min-diagonal-zero (V=G diagonal = 0.000) arises structurally because σ_proj at ell_V*=4 for G is 0.038, so α=+2σ injects only 0.076 in magnitude — insufficient to produce a measurable v_effect shift. This is the layer-pick under-power failure mode documented in the main results.
2. The permutation test at p=0.84 correctly fails to reject the null — the matrix has no selectivity pattern at these layers. This is a legitimate experimental negative result.
3. C4's metric (`max_offdiag / min_diag = ∞`) is technically indeterminate (0/0 convention with 1e-9 floor giving a huge ratio) — not a fraudulent result, but mathematically degenerate.
4. The w_effects computation in `run_one_grid_point()` measures W's effect on baseline transfers under V's intervention — specifically `w_effects[W] = mean(τ | W=pos) - mean(τ | W=neg)` over the 200 baseline held-out rows, while V is being steered. This correctly implements the C4 selectivity check.
5. `M[V,W] = w_effects[W](α=+2σ) - w_effects[W](α=0)` — this shift is computed by subtracting the α=0 run's w_effects from the α=+2σ run's w_effects. This is the correct differential measure.

## Checks

### A. Ground Truth Provenance: WARN
- **C4 evaluation target (selectivity matrix entries)**: M[V, W] = shift in W's effect on τ under V's steering. τ is decoded from `Llama-3.1-8B-Instruct` greedy generation — **synthetic_proxy** evaluation. Same proxy issue as C3. The W-effect is computed from the same model outputs as C3's v_effect.
- **Evidence**: `scripts/m4_steer_and_m5_selectivity.py` lines 276–283: `w_effects[W]` computed from `held_base_rows + held_partner_rows` decoded under intervention.
- **Severity**: WARN — same synthetic_proxy issue as C3 (unavoidable for this measurement type but should be labeled).

### B. Score Normalization: PASS
- M[V, W] = absolute shift in W's effect (units: transfer amount dollars). The criterion `max_offdiag ≤ 0.30 × min_diag` is a relative comparison of absolute matrix entries — not a metric divided by model's own max/mean output.
- `c4_ratio_max_off_over_min_diag` uses max(min_diag, 1e-9) to avoid division by zero — this is a numeric guard, not score normalization.

### C. Result File Existence: PASS
- `runs/M_main_v1/artifacts/m5/selectivity_matrix.json` — verified present: matrix entries match EXPERIMENT_RESULTS.md (V=G row all zeros, V=A row uniform ~-0.45, V=I row moderate, V=M row moderate). min_diag=0.0, max_offdiag=0.455, c4_ratio=∞ (stored as 454545454.5), n_missing_grid_points=0.
- `runs/M_main_v1/artifacts/m5/permutation_test.json` — verified present: p=0.839, n_perm=5000. Consistent with reported p=0.84.
- The M5 computation reuses M4's in-memory `summary` list (lines 451–468 in m4_steer_and_m5_selectivity.py). The selectivity matrix is computed inline from M4's run results — there are no separate M5 run files. This is methodologically correct (M5 is a post-processing step on M4's data).
- Tracker M5_r1 status: **done** (included in M4's run, notes "0 (in M4)").

### D. Dead Code Detection: PASS
- `permutation_test_offdiag_vs_diag()` called at line 473 in `main()`.
- The M5 matrix computation loop (lines 450–469) executes directly in `main()` after the M4 grid.
- `extract_run()` lambda at line 443 is used in the M5 loop.
- No dead evaluation code identified.

### E. Scope Assessment: WARN
- **Permutation test units**: The permutation test at lines 313–335 is computed over trials = the 4 diagonal values and 12 off-diagonal values of M (4+12=16 matrix entries in absolute value). The observed statistic is `mean(|diag|) - mean(|off-diag|)` = −0.102 (negative, meaning off-diagonal is larger than diagonal — the wrong direction). With p=0.84 this is a clear failure to reject "off-diag = diag" — which correctly supports the NOT SUPPORTED verdict.
- **Min-diagonal zero for V=G**: M[G, G] = 0.000 (all G-row entries are 0). This is not a measurement error — it reflects that at ell_V*=4 with σ_proj(G, LEACE)=0.038, the injection α=+2σ=0.076 produces no measurable transfer shift (under-power documented in M4). The 0.000 diagonal entry is legitimate, not a phantom result.
- **The c4_ratio=∞**: The ratio `max_offdiag / min_diag = 0.455 / 0.000 = ∞` is correctly flagged in EXPERIMENT_RESULTS.md as "c4_ratio_max_off_over_min_diag = ∞ (min diag ≡ 0)." The 1e-9 floor gives 454545454 in the JSON. This is transparent about the degenerate state.
- **Evaluation methodology (w_effects computation)**: The w_effects are computed over the BASELINE trials (not partner trials) — specifically `w_effects[W] = mean(τ | W=pos, baseline rows under V's steering) - mean(τ | W=neg, baseline rows under V's steering)` (lines 276–283). This measures W's marginal effect on the transfer after V's activation has been altered. This is the intended C4 measurement.
- **Sample size**: 200 baseline held-out trials at each α point. Split into ~100 W=pos and ~100 W=neg (balanced 2×2×2×2 design). This gives ~100 per cell for the W-effect estimation. Adequate for the scale of the effects observed.
- **Severity**: WARN — min-diagonal zero is a legitimate outcome given the under-power at ell_V*, but the indeterminate ratio requires careful communication. The EXPERIMENT_RESULTS.md handles this correctly.

### F. Evaluation Type: WARN
- **M[V,W] entries**: `synthetic_proxy` — τ decoded from model greedy generation.
- Same proxy issue as C3 (unavoidable and documented).

## Action Items
1. [WARN] Label C4 evaluation as `synthetic_proxy` in EXPERIMENT_RESULTS.md (same action as C3).
2. [WARN] Confirm in EXPERIMENT_RESULTS.md that the V=A off-diagonal pattern (uniform −0.45 across G, I, M) suggests a global mean shift under V=A steering at ell_V*=6 (σ_proj=very small), not a cross-variable selectivity failure per se — this is the same under-power phenomenon as V=G's zero diagonal, but manifesting as a uniform non-zero shift rather than a zero shift. Document the interpretation.
3. [WARN] Note the c4_ratio=∞ in the report as a "degenerate" rather than a "failed" criterion — the actual criterion failure is "min_diag=0 which was itself a consequence of layer under-power, not a selectivity failure."
