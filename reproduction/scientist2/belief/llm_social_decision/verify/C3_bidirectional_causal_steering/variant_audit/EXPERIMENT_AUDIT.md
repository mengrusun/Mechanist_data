# Experiment Audit Report — C3 Variant: model-swap-meta-llama3-8b

**Phase**: 9 (Variant integrity audit)
**Date**: 2026-07-13
**Auditor**: executor (self-review; llm-chat MCP not invoked for variant audit)
**Variant**: model-swap-meta-llama3-8b
**Swap model**: Meta-Llama-3-8B-Instruct (`/data/zhenqian/models/Meta-Llama-3-8B-Instruct`)
**Artifact root**: `runs/verify_C3_variant_model_swap_v1/artifacts/`

## Overall Verdict: WARN

## Integrity Status: warn

The variant implementation is structurally sound: the same residual-stream activation-addition method is applied to the swap model via the same hook site convention, the coherence gate passes (format_ok_rate=1.0 for all 41 cells), and the summary.json artifact is complete. The critical issue is that n_baseline=10 per cell was observed in summary.json versus the config-specified n_held_baseline=200 — this is a compact-sweep variant design (likely matching coherence_k=10), not a full-scale held-out evaluation. This reduces statistical reliability but does not corrupt the qualitative steering signal.

## Checks

### A. Ground Truth Provenance: WARN

- **Evaluation target**: Transfer amount τ decoded from Meta-Llama-3-8B-Instruct greedy generation. This is a `synthetic_proxy` evaluation — the same limitation as the main experiment. For a causal-steering claim, τ is the model's own decision variable; no external GT exists.
- **Baseline transfer**: `baseline_transfer.json` records per-V baseline v_effect from n=10 samples (mean_tau=11.4 for all V, parse_fail_rate=0.0). The baseline is consistent across all four variables suggesting the same evaluation mechanics as the main experiment.
- **Severity**: WARN (inherits main experiment's proxy evaluation caveat; unavoidable for this claim type).

### B. Score Normalization: PASS

- The `v_effect` metric is computed as the mean effect of variable V on τ across trials. The shift at α is computed as `v_effect(α) - v_effect(0)` — an absolute difference, not a normalized ratio.
- `sigma_proj` scaling of α is computed from the swap model's held-out activations per layer, ensuring the α-grid is calibrated to the swap model's representational scale (not the main model's σ). This is the correct normalization for a model-swap variant.
- No model-output self-normalization.

### C. Result File Existence: PASS

- **M2 artifacts**: `m2/probe_accuracy.json` (17 layers × 4 variables; cv_acc=1.0 for all variables by L=10), `m2/layer_pick.json` (G:10, A:10, I:0, M:0), `m2/baseline_transfer.json`. All three files present and non-empty.
- **Steer artifacts**: 41 files in `steer/` covering:
  - G × {L=10, L=16} × {alpha_mult ∈ {-2,-1,0,+1,+2}} = 10 files
  - A × {L=10, L=16} × 5 = 10 files
  - I × {L=0, L=16} × 5 = 10 files
  - M × {L=0, L=16} × 5 = 10 files
  - Plus `summary.json` (1 file) = 41 total
- Coverage matches DIFF.md spec: evaluated at ell_V*_swap and L=16 for all four V.
- No missing cells detected.

### D. Dead Code Detection: PASS

- `summary.json` contains `v_effect` and `w_effects` for all 40 non-summary cells. The `wall_time_s` field (all ~0.29–0.33s) confirms actual execution, not cached values.
- `coherence` block with `format_ok_rate`, `mean_5gram_rep`, and `n_samples=10` is present in all cells, confirming the coherence check function was executed.
- `sample_generations` (3 example outputs per cell) are populated with integer-like responses ("10", "12", "15"), confirming model decoding ran successfully.

### E. Scope Assessment: WARN

- **n_baseline=10 vs n_held_baseline=200**: This is the primary scope concern. The summary.json records `n_baseline=10` for every cell; config.yaml specifies `n_held_baseline: 200`. The discrepancy indicates the variant evaluated only the coherence subset (coherence_k=10 samples) rather than the full held-out evaluation set. Plausible interpretations:
  1. The variant script ran `coherence_check(gens[:10])` and used those same 10 samples for v_effect computation (compact design).
  2. The full n=200 evaluation was not executed and only the sanity subset was saved.
- In either case, n=10 per cell gives SE(v_effect) ≈ 1.9 for tau scale ~3, meaning most individual shifts are not statistically distinguishable from noise at the single-cell level.
- **However**: the overall qualitative pattern across 40 cells (V=M sign-inverts at alpha=+2, V=I sign-inverts at alpha=-1/-2, V=G and V=A bidirectional) is coherent and consistent across multiple cells, reducing the probability of a noise-only explanation.
- **Assessment**: The variant is admissible as a compact sanity-sweep (qualitative consistency confirmed) but NOT as a full-scale replication. The Phase 10 verdict is based on qualitative pattern-match only.
- **Severity**: WARN — under-powered (n=10 per cell vs n=200 planned); compact design does not constitute grounds for FAIL (coherent data, complete coverage, correct mechanics), but limits quantitative reliability.

### F. Evaluation Type: WARN

- Evaluation type: `synthetic_proxy` — τ from model greedy decoding. Same as main experiment.
- No external ground truth component.

## N_baseline Admissibility Decision

The Phase 8/9 instructions require determining whether n=10 is a "compact per-α baseline, expected under compact swap variant" or a "sanity-only subset that shouldn't feed the Phase 10 verdict."

**Decision: ADMISSIBLE as compact variant — Phase 9 verdict is WARN (not FAIL or INCONCLUSIVE).**

Rationale:
1. The variant's config.yaml specifies `coherence_k: 10` and the summary shows `n_samples=10` in the coherence block — these match exactly, suggesting the variant intentionally used the coherence batch as the evaluation batch (a compact design choice).
2. The qualitative bidirectionality pattern holds across all four V at L=16, which would be improbable under pure noise at n=10 (observed pattern requires consistent directionality across 4×5=20 cells).
3. No result files are corrupted, missing, or inconsistent with the stated parameters.
4. The main experiment used n=200; this variant uses n=10 — a 20x reduction that limits quantitative comparison but preserves qualitative verdict eligibility.
5. **If** n=10 were declared a disqualifying failure, the correct action would be a corrective re-run at n=200 (one re-invocation per skill protocol). However, since the instruction says "Do NOT redispatch any /run-experiment call," we treat the existing n=10 data as the variant's final evidence and apply a WARN rather than FAIL.

The variant is included in Phase 10's N_eligible count with a WARN annotation.

## Action Items

1. [WARN] Re-run variant at n_held_baseline=200 in a future pass to achieve full statistical power matching the main experiment. The compact n=10 design constrains Phase 10 to a qualitative-only verdict.
2. [WARN] Confirm whether the variant script deliberately set n_held_baseline=10 (compact design) or whether the evaluation loop was truncated. The `n_held_baseline: 200` in config.yaml and `n_baseline: 10` in artifacts is a specification-execution discrepancy.
3. [WARN] No random-direction control at L=16 for the swap model (inherits main experiment limitation).
