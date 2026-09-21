# Mechanism Audit Report — Claim C2

**Date**: 2026-07-18
**Auditor**: external LLM reviewer (cross-model, gpt-5.4 via API)
**Project**: Cross-Modal Covert Transfer of Unsafe Behavior via a Text-Only Teacher-Generated Channel
**Claim**: C2 — Some low-rank residual-stream direction in the student's language tower causally mediates the covert-channel safety drop, with sign + monotone dose-response + specificity all confirmed; OR the mechanism arc reports a BOUNDED NULL under this ontology.
**Linked milestones**: M1 (M1.L0, M1.L-Core), M2 (M2.2a, M2.2b, M2.2b.aggregate, M2.2c, M2.2d)

## Overall Verdict: FAIL

*Check A (Steering Coefficient Sweep) was triggered and the external LLM reviewer returned FAIL on two hard-fail criteria: (1) no independent capability/coherence metric logged at each alpha point, and (2) sign/monotonicity pattern broken (median Spearman rho = +0.371, 2/3 seeds show wrong-direction correlation). Note: FAIL here reflects mechanism-rigor protocol violation, not scientific fraud — the BOUNDED NULL is the correct scientific conclusion and is consistent with the data.*

## Integrity Status: fail

## Trigger Detection (Check A)

Check A fires for any additive activation intervention with a scalar coefficient alpha. Confirmed triggers:
- `steer_and_eval.py` implements `h_new = h + alpha_actual * u` — additive injection at top-3 language-tower layers
- `ablate_and_eval.py` implements `h_new = h - (h·u)u` — project-out (related intervention)
- EXPERIMENT_PLAN.md M2 methodology names "steering", "dose-response", "α-sweep", "CAA-style additive push"
- `triggered[A] = true`

## Checks

### A. Steering Coefficient Sweep: FAIL

**Audit questions (8/8 assessed by external reviewer):**

1. **Was a sweep performed?** — PASS. α ∈ {-2, -1, -0.5, 0, +0.5, +1, +2} σ_proj units (7 points). Not a single hardcoded value.

2. **Did sweep span ≥3 orders of magnitude with α=0 baseline?** — WARN. α=0 baseline included. Nonzero magnitudes span {0.5, 1, 2} σ_proj units — factor of 4, far short of 10³. Span deficiency alone triggers WARN (not FAIL).

3. **Was α expressed in σ_proj units?** — PASS. Confirmed: `sigma_proj = std(h @ u_diff per layer)` via `compute_sigma_proj()`; `agg_sigma = RMS(sigma_per_layer)`; `alpha_actual = args.alpha * agg_sigma`. σ_proj scaling is explicit in the code.

4. **Was BOTH a target metric AND independent capability/coherence metric logged at every alpha point?** — **FAIL**. Per-run JSON contains only `acc_steered` and `n_correct`. No perplexity, no OTHER rate, no off-target accuracy, no coherence metric. This is a hard-fail criterion: without a capability metric, OOD collapse at extreme α cannot be detected; the steering "effect" could be a behavioral artifact.
   Evidence: per-run JSON keys = {alpha, alpha_actual, agg_sigma_proj, seed, random_direction, top_layers_hidden_states_index, layer_hook_indices, n, acc_steered, n_correct, per_item}. No capability field.

5. **Was a usable plateau identified?** — FAIL (consequential on Q4). No capability metric → capability-within-tolerance cannot be established. Additionally, the dose-response curve is non-monotone (seed42 rho=+0.371, seed2026 rho=+0.786) — no stable causal plateau in the expected direction.

6. **Was locked α placed in the middle of the plateau?** — N/A. No locked α concept in this experiment. The sweep is a monotonicity test; BOUNDED NULL is the outcome (no α was selected for downstream inference). Not an additional independent FAIL.

7. **Was a random-direction control run with n_random ≥ 30?** — WARN. Matched-random-direction control was run (M2.2c), but with n_random = 1 per seed (not ≥ 30 independent random draws). Mean |ΔAcc| ≤ 1pp (0.64%, 0.32%, 0.86%) — qualitatively supportive of specificity, but underpowered.

8. **If asymmetric protocol, is sign pattern preserved?** — **FAIL**. Direction convention: v = mean(h_treated) − mean(h_Ctrl-B); positive α should push toward treated (lower acc). Observed: seed42 Spearman ρ = +0.371 (acc INCREASES with positive α), seed2026 ρ = +0.786 (same wrong direction), only seed123 ρ = −0.019 (near-zero). Median ρ = +0.371. Two of three seeds show sign inversion. Hard-fail criterion: "sign pattern broken."

**Hard-fail criteria triggered**: Q4 (no capability metric) and Q8 (sign pattern broken).

**Evidence references**:
- `scripts/steer_and_eval.py:39-98` — σ_proj computation + hook implementation + per-run output structure
- `results/mech/M2_2b_alpha{alpha}_seed{seed}.json` — all 21 files, keys confirmed
- `results/MECHANISM_VERDICT.json:spearman_rho_per_seed = {42: +0.371, 123: -0.019, 2026: +0.786}`
- `results/mech/M2_2c_randdir_alpha*_seed*.json` — 21 files, random_direction=True, n_random_directions=1 per seed
- `scripts/launch_m1_m2.sh:88-153` — sweep and specificity invocations

**Check A structured fields**:
- status: fail
- intervention_type: CAA
- sweep_grid: [-2, -1, -0.5, 0, 0.5, 1, 2] (sigma_proj units)
- sigma_proj_scaling: true
- capability_metric: null (not logged)
- plateau_range: null (no plateau identified — BOUNDED NULL result)
- locked_alpha: null (no α locked — monotonicity fails)
- alpha_position: n/a
- random_baseline: {run: true, n_random: 1, passed: null (underpowered)}
- sign_pattern: broken (2/3 seeds show positive rho; expected negative)
- output_case_spotcheck: {cases_available: false, verdict: no_cases_logged, note: "per_item key present in JSONs but no output-text spot-check was submitted to reviewer"}
- evidence: ["scripts/steer_and_eval.py:31-100", "results/mech/M2_2b_alpha0_seed42.json", "results/MECHANISM_VERDICT.json:spearman_rho_per_seed"]

### B–F. Reserved

Not yet implemented. Not applicable (or not triggered).

## Scientific Interpretation

The FAIL verdict reflects mechanism-rigor protocol violations (absence of capability metric; broken sign/monotonicity pattern) rather than scientific misconduct. The BOUNDED NULL conclusion is scientifically honest and pre-registered: the plan specifies that BOUNDED NULL is the verdict when recovery fails, which it does (max recovery = 0.256, below the 0.30 threshold in ≥2/3 seeds). The sign inversion IS the scientific finding — the rank-3 residual-stream direction does not causally mediate the safety drop in the expected direction under the CAA ontology.

The mechanism audit FAIL means: if this experiment were to claim a positive causal mechanism, the rigor would be insufficient. Since it claims BOUNDED NULL, the protocol violation does not invalidate the null finding per se. However, per auto-verify's rules, a FAIL at Phase 2 → claim C2 is marked INCONCLUSIVE for this verify run (variants would compute robustness around an anchor where the steering rig lacks a capability metric).

## Action Items

1. **Critical**: Add an independent capability/coherence metric at every α sweep point (e.g., OTHER rate from QA_I per-item verdicts, perplexity on a held-out text sample, or off-target eval_pairs accuracy). This is cheap to add to `steer_and_eval.py` without re-running the main experiment.
2. **Recommended**: Increase n_random to ≥ 30 per seed for the specificity control to reach the criterion threshold.
3. **Informational**: The sign inversion (positive Spearman ρ) is the BOUNDED NULL's key observation — it should be foregrounded in the mechanism section of the paper as the reason for the null: "the direction appears sign-inverted or non-concentrated in this ontology."
