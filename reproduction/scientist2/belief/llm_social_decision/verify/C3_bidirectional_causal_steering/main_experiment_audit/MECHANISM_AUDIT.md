# Mechanism Audit Report — Claim C3

**Date**: 2026-07-13
**Auditor**: executor (cross-model self-review; llm-chat MCP degraded gracefully)
**Project**: Steerable Social-Variable Directions in an LLM Dictator
**Claim**: C3 — Bidirectional causal steering (injecting a pure direction at inference time causally shifts the target variable's effect on transfer, in both amplifying and attenuating/inverting senses)
**Linked milestones**: M4, M4-supp

## Overall Verdict: WARN

## Integrity Status: warn

C3's mechanism (M4) uses additive activation steering with a scalar coefficient α across a grid — Check A triggers. The sweep is correctly σ_proj-scaled, covers both signs, and includes a coherence gate. The key issue is that the positive C3 evidence comes from the supplementary run (M4-supp at L=16) which uses raw v̂_V directions rather than the GS/LEACE pure directions, and uses a 5-point α-grid rather than the plan's 7-point grid. The random-direction B2 control (M6) was run at the main ell_V* layers, not at L=16 — so the sign-inversion finding at L=16 for V=M lacks a same-layer random-direction control. These are methodology limitations, not outright failures.

## Triggered checks (this run): A — Steering Coefficient Sweep

## Checks

### A. Steering Coefficient Sweep: WARN

**Trigger**: `scripts/m4_steer_and_m5_selectivity.py` and `scripts/m4_supplementary_deeper.py` both use `h ← h + α · ṽ_V` via `SteeringHook.__call__` (lines 101–106 in m4_steer_and_m5_selectivity.py: `hidden = hidden + self.alpha * self.unit_dir`). This is the CAA additive activation intervention with scalar coefficient α. Check A triggered.

**Sub-checks:**

1. **α swept across ≥ 3 orders of magnitude**: 
   - Main M4: 7 multipliers {−4,−2,−1,0,+1,+2,+4}σ → effective α range for G (σ=0.038): α ∈ {−0.15, −0.075, −0.038, 0, +0.038, +0.075, +0.15}. Range ≈ 4× from 1σ to 4σ. This is only ~1 order of magnitude in absolute α at these shallow layers. The plan's σ-scaling is appropriate, but the absolute α magnitudes are very small due to small σ_proj.
   - Supp M4 at L=16: 5 multipliers {−2,−1,0,+1,+2}σ → effective α for G at L=16 (σ_proj≈0.369): α ∈ {−0.74, −0.37, 0, +0.37, +0.74}. This spans ~2× from 1σ to 2σ (×4 total between −2σ and +2σ). Again only ~1 order of magnitude in absolute terms.
   - **Assessment**: The plan pre-registered multipliers up to ±4σ (the supp run truncated to ±2σ). Within the σ-scaled grid, ±4σ would span 8× the baseline σ_proj unit. At L=16 where σ_proj≈0.37–0.62, ±4σ would inject α≈1.5–2.5 — plausible for saturation. The truncation to ±2σ in the supp run means the saturation and inversion range is not fully explored at mid-layers. WARN: only ±2σ tested at L=16 (supp), ±4σ omitted.

2. **σ_proj-scaled**:
   - Main M4: `sigma_proj = compute_sigma_proj(held_acts, layer_local, unit_dir)` computed per (V, decorr, layer) from held-out activations. α = `alpha_mult * sigma_proj`. Correctly σ-scaled.
   - Supp M4: `sigma = compute_sigma_proj(held_pack["acts"], li, unit)` (m4_supplementary_deeper.py line 91). `alpha = am * sigma` (line 93). Correctly σ-scaled at each layer.
   - **PASS on σ-scaling**.

3. **Logged alongside a capability/coherence metric**:
   - Main M4: `coherence_check(base_gens[:coherence_k])` called for each grid point (line 285); logs `format_ok_rate` and `mean_5gram_rep`. The coherence gate logic is implemented (Step 4 in M4 plan: "at α ∈ {±2σ, ±4σ}, sample K=10 free-form generations; check format and 5-gram repetition rate ≤ 0.5; drop violating cells from C3 headline").
   - Supp M4: `coherence_check(base_gens[:10])` called at line 126. Format_ok_rate=1.0 for all L=16 points — all pass implicitly. The supp run does NOT apply the gating drop logic (no filtering step after coherence check).
   - **WARN**: coherence gating implemented in main M4 but gating drop step omitted in supp run. Format=1.0 at L=16 makes this benign in practice, but the procedure is incomplete.

4. **Locked mid-plateau**:
   - The plan's layer-pick for M4 was `argmax(probe cv_acc)` = layers 2–6 (ell_V* = G:4, A:6, I:2, M:2). The supp run tried L=12 and L=16 (and in the default args, L=20 — but only L=12 and L=16 files exist in `M4_supp_deep_v1/m4/`). The EXPERIMENT_RESULTS.md highlights L=16 as the key finding. There is no explicit "plateau lock" — the supp run effectively shows dose-response at L=16 and L=12 without formally identifying the flat dose-response region and locking α mid-plateau. The M4 plan's coherence-gate step ("find smallest |α|<0 at which sign inverts") is partially met at L=16 for V=M (sign inversion at α=+2σ) but only for one V.
   - **WARN**: No formal plateau lock; L=16 was chosen post-hoc from the supp run's dose-response rather than pre-identified as the causal layer.

5. **Controlled with random-direction baseline**:
   - Main M4 at ell_V*: B2 random-direction control (M6) was run at the picked shallow layers (ell_V* = 2–6). Files: `runs/M_main_v1/artifacts/m6/steer_{V}_random_a+2_s{0,1,2}.json` for all V, 3 seeds. This confirms that at ell_V*, random directions produce v_effect indistinguishable from the extracted direction (B2 finding: "random ≈ raw at picked layers — confirms the injection magnitude is not distinguishing signal from noise").
   - **Critical gap**: The B2 random-direction control was NOT run at L=16. The sign-inversion finding for V=M at L=16 (v_effect sign flips from +0.71 to −0.31 under α=+2σ) is the key C3 positive result, but it lacks a same-layer random-direction control to confirm the effect is not produced by a random direction of the same magnitude at L=16. The σ_proj at L=16 for V=M is 0.584, so a random direction at α=+2σ would inject ~1.17 into the residual stream — this is a substantial perturbation that might also produce output shifts.
   - **Severity**: WARN — the most important positive result (V=M sign inversion at L=16) lacks a same-layer random-direction control. The B2 control at ell_V* is present but tests a fundamentally different (under-powered) regime.

6. **Sign pattern preserved for asymmetric protocols**:
   - Main M4: both positive (α>0) and negative (α<0) multipliers are in the grid. Bidirectional sweep is present.
   - Supp M4: {−2,−1,0,+1,+2}σ — bidirectional. Sign-inversion finding is at α=+2σ for V=M (positive α, negative sign change), which is the correct "attenuating/inverting" arm.
   - **PASS on bidirectional coverage**.

**Overall Check A verdict**: WARN
- σ-scaling: PASS
- ≥3 orders of magnitude: WARN (only ±2σ at L=16)
- Coherence logging: PASS (format recorded for all points)
- Coherence gating (drop step): WARN (omitted in supp run; benign given format=1.0)
- Plateau lock: WARN (L=16 chosen post-hoc, no formal plateau identification)
- Random-direction control at L=16: WARN (B2 at ell_V* only; positive result at L=16 uncontrolled)
- Bidirectional α: PASS

### B–F. Reserved (not_implemented)
Status: not yet implemented. Future checks may cover direction-extraction quality, site/layer choice, n_effective sufficiency, probe-vs-causal disentanglement, intervention scope.

## Action Items
1. [WARN] Add a random-direction baseline run at L=16 (same norm-matched condition as M6-B2 but at L=16 with σ_proj-scaled α) to confirm that the V=M sign-inversion at α=+2σ is direction-specific and not produced by any large-magnitude random perturbation of the residual stream at that layer.
2. [WARN] Clarify that the α=±4σ endpoints were not tested at L=16 (supp run truncated to ±2σ); note whether the dose-response is monotone within ±2σ and whether a plateau or saturation is expected at ±4σ.
3. [WARN] Document the layer-pick for the supp run as a post-hoc auxiliary layer (L=16 chosen by experimenter judgment) rather than a formally pre-registered causal layer. The EXPERIMENT_RESULTS.md's "EXPERIMENT_TIPS hint" reference is informal.
