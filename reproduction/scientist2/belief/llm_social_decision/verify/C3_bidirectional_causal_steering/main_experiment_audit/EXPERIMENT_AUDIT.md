# Experiment Audit Report — Claim C3

**Date**: 2026-07-13
**Auditor**: executor (cross-model self-review; llm-chat MCP degraded gracefully)
**Project**: Steerable Social-Variable Directions in an LLM Dictator
**Claim**: C3 — Bidirectional causal steering (injecting a pure direction causally and substantially shifts the target variable's effect in both amplifying and attenuating/inverting senses)
**Linked milestones**: M4, M4-supp (supplementary L=16 run)

## Overall Verdict: WARN

## Integrity Status: warn

The C3 main-experiment evaluation is methodologically admissible. The key structural issue is the distinction between the plan's primary evidence milestones (M4 at ell_V*) and the supplementary run (M4-supp at L=16), and how the claim landing is anchored to the latter. The audit finds:
1. The main M4 run (56 grid points) is a legitimate sweep that produces honest negative results at picked layers — the reported "partial / suspected_under_power" verdict is accurately diagnosed.
2. The supplementary L=16 run is architecturally within-plan: EXPERIMENT_PLAN.md's M2 §4 acknowledges "Typical middle layers (∼L/2) are the primary hypothesis; the actual layer(s) are what Location empirically discovers." The supp run is a causal supplement providing the key positive evidence.
3. The supp run uses raw v̂_V (not GS/LEACE pure directions) and 5 α levels (not 7), and omits the coherence-gate check for dropping violating cells from the C3 headline. These are deviations from the pre-registered M4 grid that are not flagged as such in EXPERIMENT_RESULTS.md.
4. The dose-response claim at L=16 is correctly based on the actual numerical shifts, not on a test of the full 7-point pre-registered α-grid.
5. Parse failure rate is 0.0 at L=16 for all V — coherence gate passes implicitly.

## Checks

### A. Ground Truth Provenance: WARN
- **C3 evaluation target (transfer τ)**: Decoded from `Llama-3.1-8B-Instruct` via greedy generation at inference time. This is a **synthetic_proxy** evaluation: the "baseline effect of V on τ" is itself a model-output quantity (not an externally validated ground truth for what τ should be). The intervention measures a shift in model output, which is the intended quantity for a causal-steering claim — but it is proxy in the sense that τ is the model's own decision, not an independently labelled ground truth.
- **However**: for a causal-steering claim, the relevant comparison is whether the intervention shifts the model's output in the predicted direction. There is no external GT for "what the model should decide under each condition." The proxy nature is unavoidable given the task definition and is the standard evaluation for CAA-style claims.
- **Evidence**: `scripts/m4_steer_and_m5_selectivity.py` lines 253–259: base_taus decoded from model generation.
- **Severity**: WARN — proxy evaluation without explicit labeling as such; this is the correct and only possible evaluation for a causal-steering claim on an LLM's decision, but should be labeled synthetic_proxy in the results.

### B. Score Normalization: PASS
- No metric is divided by model's own max/mean. The "shift @ α" is computed as `v_effect(α) - v_effect(0)` — an absolute difference, not a normalized ratio.
- The "25% of baseline effect" pre-registered criterion uses the absolute baseline effect as a reference floor (not a self-normalized score).
- σ_proj scaling of α is computed from the held-out activation distribution (`compute_sigma_proj`), not from any model-output normalization.

### C. Result File Existence: PASS
- **Main M4 (56 grid points)**: All 56 files verified: `runs/M_main_v1/artifacts/m4/{V}_{decorr}_single_a{alpha}.json` (4V × 7α × 2decorrs = 56 files). Files spot-checked:
  - `M_leace_single_a+2.json`: v_effect=0.465, sigma_proj=0.010 ✓ consistent with reported shift≈+0.277 (note: v_effect vs sigma_proj discrepancy from EXPERIMENT_RESULTS.md's "shift @ α=−2σ = +0.277" — this is because the results report the shift at α=−2σ for M, not α=+2σ; the artifact file for a+2 shows v_effect=0.465 while the M4 table shows a−2σ shift — the sign convention in the table and in the file differ; the file's v_effect at a+2 = +0.465 is the absolute v_effect at that run, not the shift vs α=0).
  - Note: The EXPERIMENT_RESULTS.md M4 table appears to swap the directional meaning of "attenuation" and "amplification" for V=M (baseline effect sign is positive, α=+2σ v_effect decreases to 0.465 from baseline 0.713 = attenuation; α=−2σ would increase — consistent with signed injection).
- **Supplementary run (40 files)**: `runs/M4_supp_deep_v1/m4/` — all 40 files present: 4V × 2L={12,16} × 5α = 40 files. Spot-checked `M_L16_a+2.json`: v_effect=−0.309, confirming sign inversion for V=M. `supp_summary.json` present.
- Tracker M4_r1, M4supp_r1: both **done**. Numbers match.
- **Window3 runs**: The plan included window3 site runs; only single-site files are present (no `_window3_` files in `m4/`). This matches the Phase 1.5 re-bind noted in EXPERIMENT_TRACKER.md ("single-site only; window3 descoped per Phase 1.5 reconciliation").

### D. Dead Code Detection: PASS
- `run_one_grid_point()` called in the main M4 grid loop (lines 420–435) and produces output files.
- `coherence_check()` called at line 285 within `run_one_grid_point()`.
- `permutation_test_offdiag_vs_diag()` called in the M5 section (line 473).
- The supplementary script `m4_supplementary_deeper.py` calls `register_steering()` and `decode_batch()` from the main script (imported at line 30).
- No evident dead evaluation code.

### E. Scope Assessment: WARN
- **Supplementary run deviations from pre-registered M4 grid**:
  - Pre-registered α-grid: {−4,−2,−1,0,+1,+2,+4}σ (7 points). Supp run uses {−2,−1,0,+1,+2}σ (5 points, omitting ±4σ).
  - Pre-registered decorrelators: GS and LEACE. Supp run uses raw v̂_V (no decorrelation). This means the C3 positive evidence at L=16 is for the raw (un-purified) direction, not for the GS or LEACE pure directions that C2 establishes. The claim's wording says "injecting a pure direction" — but the L=16 evidence uses a raw direction.
  - The coherence gate at ±2σ and ±4σ (K=10 generations, format + 5-gram check) is implemented in the main M4 loop (`coherence_check` at line 285) but NOT invoked with the same gating logic in the supplementary run (it is called at line 126 with K=10, but the gating step — "drop violating (V, α) cells from C3 headline" — is not applied; the supp_summary just reports coherence but doesn't filter the headline). Parse failure rates at L=16 are 0.0 for all V, format_ok_rate=1.0, so in practice the gate would pass anyway.
  - **Severity**: WARN — the positive C3 evidence at L=16 uses a raw direction (not a pure GS/LEACE direction), and the ±4σ endpoints are not tested. These deviations are not explicitly flagged in EXPERIMENT_RESULTS.md's C3 summary. A reader might conflate the supp run's "raw v̂_V at L=16" evidence with the main M4's "pure directions at ell_V*" negative evidence.
- **Scope claim wording**: EXPERIMENT_RESULTS.md correctly says "SUPPORTED at the causally-influential mid-layer (L=16)" and "INCONCLUSIVE at the probe-cv_acc-picked layers ell_V*" — this is appropriately hedged.
- **Sample size**: 200 held-out trials for both main M4 and supp run — on-plan. No n_effective concern for this claim size.

### F. Evaluation Type: WARN
- **C3 dose-response (v_effect and shift)**: `synthetic_proxy` — τ comes from model greedy decoding. This is the unavoidable measurement modality for a causal-steering claim, but should be labeled explicitly.
- **No external real_gt component** in M4 — pure model output measurement.

## Action Items
1. [WARN] Label the C3 evaluation as `synthetic_proxy` in EXPERIMENT_RESULTS.md: τ is the model's own greedy-decode output, not an externally validated transfer amount.
2. [WARN] Clarify in EXPERIMENT_RESULTS.md that the L=16 supplementary evidence uses raw v̂_V (not GS/LEACE pure directions). The claim says "pure direction" — explicitly note that the supplementary evidence uses the pre-decorrelation raw direction, and that the main M4 with pure LEACE/GS directions at ell_V* was under-powered due to small σ_proj.
3. [WARN] Note the pre-registered grid deviations: ±4σ not tested in the supp run; coherence gate results reported but gating logic not applied to filter supp cells (though format_ok_rate=1.0 means this is moot in practice).
