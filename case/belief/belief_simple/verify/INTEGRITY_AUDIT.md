# Integrity Audit

**Overall**: WARN  ← max severity across Phase 2 (main experiment) + Phase 9 (variants)
**Main-experiment integrity (Phase 2)**: WARN  ← max severity across per-claim combined verdicts
**Variant integrity (Phase 9)**: WARN  ← C2 model-swap-olmo-1b: EXPERIMENT=pass, MECHANISM=warn (cross-vocab PPL)

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1 | PASS | N/A | PASS | continue | verify/C1_scale_dependent_emergence/main_experiment_audit/ |
| C2 | PASS | N/A | PASS | continue | verify/C2_belief_heads_localization/main_experiment_audit/ |
| C3 | WARN | N/A | WARN | continue-with-warn (experiment: scope 24/154 checkpoints) | verify/C3_formation_window/main_experiment_audit/ |
| C4 | PASS | N/A | PASS | continue | verify/C4_dynamic_controllability/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim.
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. `N/A` when no implemented check triggered (zero-ablation and head-output amplification do not match Check A's additive-direction-steering trigger; B-F reserved).
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a`.
> - **Gate decision** — what Phase 2 does with this claim.
>
> Auditor note: llm-chat MCP unavailable (empty .mcp.json config); self-review substituted per CLAIMS_LEDGER.md precedent established during experiment stage.

## Variant integrity (Phase 9)

Scope: C2 only (MAX_VERIFY_CLAIMS=1; C1/C3/C4 deferred to INTEGRITY_ONLY).

| Variant | Exp. audit | Mech. audit | Combined | Eligible | Judgment | Detail |
|---------|------------|-------------|----------|----------|----------|--------|
| C2 model-swap-olmo-1b | pass | warn | **warn** | yes | FAIL (n_localized=0) | verify/C2_belief_heads_localization/variant_audit/ |

**Notes:**
- Mechanism WARN: cross-vocabulary PPL — Pythia-tokenized eval tokens (max ID 50276) applied to OLMo-1B vocab (50304). No OOB errors (max < vocab size). C2d criterion uses ratio `ablated_ppl / clean_ppl` (internally consistent); absolute PPL not cross-model comparable. Ratio remains a valid ablation-effect measure.
- WARN = eligible: the variant's findings count toward robustness computation.
- N_eligible = 1 ≥ MIN_VARIANTS_FOR_VERDICT (1) → PASS/FAIL verdict issued.
- robustness = 0/1 = 0.00 < 0.50 (threshold) → C2 = FAIL
