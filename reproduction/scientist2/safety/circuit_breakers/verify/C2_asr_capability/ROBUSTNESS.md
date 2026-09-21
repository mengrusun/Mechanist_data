# Robustness Report — Claim C2

**Claim**: C2 — After representation-rerouting fine-tune, the model's Attack Success Rate (ASR) on HarmBench is materially lower than both a no-defence baseline (B0) and a competitive adversarial-training baseline (B1), while general capability (MT-Bench) is preserved.
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

- Experiment audit: FAIL
  - B1 baseline uses R2D2-lite (12 adversarial framing templates) instead of plan-specified full 512-GCG suffix optimisation
  - HarmBench eval uses "gcg-lite" (fixed generic suffix, not real GCG per-prompt optimisation)
  - MT-Bench n=40 vs plan-specified n=80 (undisclosed substitution)
  - These substitutions substantively weaken the competitive comparison: R2D2-lite over-refuses relative to a properly adversarial-trained B1, masking whether RR beats a real adversarial baseline
  - Negative result (RR_ASR=0.356 > B0_ASR=0.333) is honestly disclosed

- Mechanism audit: FAIL (Check A — Steering Coefficient Sweep)
  - M3 RR fine-tune is in C2 scope (same adapter used for ASR evaluation)
  - Same three FAIL violations as C1: single alpha=10.0 (no sweep), no capability metric at sweep points, reroute not demonstrated (delta_cos_harmful=-0.020 vs required ≤-0.30)
  - No sigma_proj scaling; random-direction control n=1 (requires ≥30)

### inconclusive_reason

main-experiment integrity broken (experiment + mechanism) — see verify/C2_asr_capability/main_experiment_audit/{EXPERIMENT,MECHANISM}_AUDIT.md

### Iteration guidance

To convert INCONCLUSIVE to a testable state:
1. Fix experiment integrity: replace R2D2-lite B1 with a genuine adversarial-training baseline (or clearly label R2D2-lite as a proxy and adjust the claim scope); run real GCG optimisation per HarmBench protocol; increase MT-Bench to n=80.
2. Fix M3 RR mechanism (see C1 iteration guidance): run alpha sweep, add capability metric, increase random-direction control to n≥30.
3. Re-run Phase 2 audits after fixing both; if combined verdict upgrades to PASS or WARN, C2 can proceed to Stage 2.

### Stage 2 plan (if C2 were admitted)

C2 was not in the top-1 pick pool because C1 (the mechanism-level parent claim) has higher importance per FINAL_PROPOSAL.md §6. C2 would only be picked if C1 was already covered. No swap-variant plan exists for C2 independently.
