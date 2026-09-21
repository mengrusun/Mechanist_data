# Integrity Audit

**Overall**: FAIL
**Main-experiment integrity (Phase 2)**: FAIL
**Variant integrity (Phase 9)**: [skipped — all main-experiment audits FAIL]

## Baseline integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1 | WARN | FAIL | FAIL | INCONCLUSIVE (main-experiment mechanism rigor broken) | verify/C1_identifiability_reroute/main_experiment_audit/ |
| C2 | FAIL | FAIL | FAIL | INCONCLUSIVE (main-experiment integrity broken — experiment + mechanism) | verify/C2_asr_capability/main_experiment_audit/ |
| C3 | WARN | FAIL | FAIL | INCONCLUSIVE (main-experiment mechanism rigor broken) | verify/C3_vlm_transfer/main_experiment_audit/ |
| C4 | WARN | FAIL | FAIL | INCONCLUSIVE (main-experiment mechanism rigor broken) | verify/C4_agent_transfer/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim.
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. `N/A` when the claim uses no mechanism intervention.
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a` and `n/a` treated as `pass`.
> - **Gate decision** — what Phase 2 does with this claim.

### C1 — Identifiability + reroute (M1, M3, M4)
- Experiment audit: WARN — probe direction used instead of mean-diff (undisclosed deviation); sweep_notes string misleading (says "L_rr decreasing significantly" but EMA=0.0016); scope limited (single model, single seed). All files exist, GT is real data, no normalization fraud.
- Mechanism audit: FAIL — Check A (Steering Coefficient Sweep): single hardcoded alpha=10.0 (no sweep), no sigma_proj scaling, no capability metric at multiple alpha values, n_random=1 for random-direction control (requires >=30), target reroute effect not demonstrated (delta_cos_harmful=-0.020 vs required <=-0.30), training instability (grad_norm spikes 852, 159, 137).
- inconclusive_reason: main-experiment mechanism rigor broken — see verify/C1_identifiability_reroute/main_experiment_audit/MECHANISM_AUDIT.md

### C2 — HarmBench ASR + capability (M2, M3, M5)
- Experiment audit: FAIL — B1 baseline uses R2D2-lite (12 template adversarial framings) in place of full 512-GCG suffixes; HarmBench "gcg-lite" is a fixed generic suffix, not real GCG optimization; MT-Bench n=40 vs plan n=80. These are substantive substitutions that affect the competitive comparison (B1 over-refuses because of the R2D2-lite weakness, masking whether RR beats a real adversarial baseline). Negative result (RR_ASR > B0_ASR) is honestly disclosed.
- Mechanism audit: FAIL — same Check A failure as C1 (M3 is in C2's scope; same alpha=10.0, no sweep, reroute not demonstrated).
- inconclusive_reason: main-experiment integrity broken (experiment + mechanism) — see verify/C2_asr_capability/main_experiment_audit/{EXPERIMENT,MECHANISM}_AUDIT.md

### C3 — VLM transfer (M6)
- Experiment audit: WARN — partial run correctly disclosed; full PGD + VLM assembly skipped (budget-gated). M6 substep files exist with correct values. Scope FAIL internally but overall WARN because skipping was documented and the partial evidence (Mistral mechanism level) is honestly reported.
- Mechanism audit: FAIL — Check A on M6T (Mistral RR fine-tune): same violations as M3 — alpha=10.0 single value (no sweep), no capability metric, n_random=1, reroute not demonstrated on Mistral (delta_cos_harmful=-0.012, L_rr_ema=0.000222).
- inconclusive_reason: main-experiment mechanism rigor broken — see verify/C3_vlm_transfer/main_experiment_audit/MECHANISM_AUDIT.md

### C4 — Agent transfer (M7)
- Experiment audit: WARN — BFCL uses real GT (AST name match). Harmful tool-use uses authored prompts + LLM judge (two-stage). Floor effect disclosed. BFCL 50-item substitute documented. Negative result honestly disclosed.
- Mechanism audit: FAIL — Check A on M3 (M7 reuses M3 adapter): same violations as C1/C2 — single alpha, no sweep, reroute not demonstrated.
- inconclusive_reason: main-experiment mechanism rigor broken — see verify/C4_agent_transfer/main_experiment_audit/MECHANISM_AUDIT.md

## Variant integrity (Phase 9)

[skipped — all main-experiment audits FAIL]

All 4 target claims received INCONCLUSIVE at Phase 2 — no claim advanced to Stage 2. Therefore no variants were run, and no Phase 9 variant integrity audit was performed.
