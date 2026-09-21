# Robustness Report — Claim C1

**Claim**: C1 — Harmful-output behaviour in an instruction-tuned LLM corresponds to identifiable internal representations that can be rerouted to an orthogonal, non-harmful subspace using only paired benign/harmful data, without exposure to any attack prompts.
**Date**: 2026-07-15
**verdict**: INCONCLUSIVE
**stage2_skip_reason**: —
**robustness**: — (Stage 2 never ran — Phase 2 gate FAIL)
**n_eligible**: 0
**n_run**: 0
**n_pass**: 0
**n_fail**: 0
**integrity_clean**: N/A (no variants)

## Verdict: INCONCLUSIVE

### Phase 2 combined verdict: FAIL

- Experiment audit: WARN
  - Probe direction used instead of plan-specified mean-difference direction (m1_locate.py:195 saves LR coef as directions.pt)
  - sweep_notes in training_summary.json: misleading static string ("L_rr decreasing significantly") inconsistent with final EMA=0.0016
  - Alpha=10.0 used vs plan alpha=1.0; loss function changed to cos_sq_plus_signed vs plan cos_sq
  - Scope: single model, single seed — within tolerance for the experiment plan
  - GT: real HarmBench+Alpaca pairs, AUC vs true labels — PASS
  - Files: all exist and match reported numbers — PASS

- Mechanism audit: FAIL (Check A — Steering Coefficient Sweep)
  - No alpha sweep: single hardcoded alpha=10.0 (plan specified 1.0)
  - No sigma_proj scaling
  - No capability metric logged at multiple alpha values
  - Random-direction control: n=1 (required n>=30)
  - Target reroute effect not demonstrated: delta_cos_harmful=-0.0197 vs required <=-0.30 (off by ~15x)
  - Training instability: grad_norm spikes to 852 (step 330), 159 (step 480), 137 (step 490)
  - L_rr EMA near-zero from step 0 (final 0.0016)

### inconclusive_reason

main-experiment mechanism rigor broken — see verify/C1_identifiability_reroute/main_experiment_audit/MECHANISM_AUDIT.md

### Iteration guidance

To convert INCONCLUSIVE to a testable state:
1. Fix the M3 RR fine-tune: run an alpha sweep [0.1, 0.5, 1.0, 5.0, 10.0, 50.0] with MMLU/MT-Bench capability metric at each point; compute sigma_proj and express alpha in those units; lock to mid-plateau alpha; increase random-direction control to n>=30.
2. Fix training instability: investigate grad-norm spikes (may require grad clipping < 1.0, lower LR, or higher warmup fraction for the RR objective).
3. Re-run Phase 2 audits after fixing M3; if combined verdict upgrades to PASS or WARN, C1 can proceed to Stage 2 with the Mistral-7B-Instruct-v0.2 model-swap variant (M6 artifacts already on disk).

### Stage 2 plan (if C1 were admitted)

If C1 were admitted: the Mistral-7B-Instruct-v0.2 model-swap variant would reuse artifacts/m6/{m1_mistral, m3_mistral, m4_mistral} without re-running. Expected consistency with main experiment (Llama-3): same reroute failure (Mistral delta_cos_harmful=-0.012 vs Llama-3 delta_cos_harmful=-0.020) → consistent_with_main_experiment=yes → robustness would be 1/1=1.0 → PASS (if Phase 2 had admitted C1).
