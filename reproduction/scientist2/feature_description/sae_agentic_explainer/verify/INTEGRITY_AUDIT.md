# Integrity Audit

**Overall**: WARN (driven by Phase 2 baseline WARNs on C1/C3/C4; Phase 9 variant audit PASS)
**Main-experiment integrity (Phase 2)**: WARN — max_severity across all per-claim combined verdicts
**Variant integrity (Phase 9)**: PASS — 0 integrity failures, 1/1 variant clean (C4: model-swap-gpt-oss-20b)

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1 | WARN | N/A | WARN | continue-with-warn (experiment: stat resolution) | verify/C1_gen_acc_main_pair/main_experiment_audit/ |
| C2 | PASS | N/A | PASS | continue | verify/C2_pred_acc_main_pair/main_experiment_audit/ |
| C3 | WARN | N/A | WARN | continue-with-warn (experiment: Bonferroni framing, small n) | verify/C3_layer_depth_generalization/main_experiment_audit/ |
| C4 | WARN | N/A | WARN | continue-with-warn (experiment: L28 missing, scope narrowed) | verify/C4_cross_pair_generalization/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim.
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. N/A = no mechanism intervention (SAGE is a Unit Interpretation pipeline with no additive steering; Slot A returns N/A for all four claims).
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a`. N/A contributes severity 0; combined is driven purely by exp_verdict for all claims here.
> - **Gate decision** — PASS/WARN admits the claim to Stages 2-3; FAIL would short-circuit to INCONCLUSIVE. All four claims are ADMITTED.

**Gate outcome**: All 4 claims ADMITTED to Stage 2 (PASS or WARN). No claim is INCONCLUSIVE.

**Warning tags summary**:
- C1 WARN: gen_acc has only 5 probes/feature (coarse discrete metric); Wilcoxon n_eff=4 (90.9% zero-differences); statistical significance is real but fragile.
- C3 WARN: Bonferroni correction described in EXPERIMENT_RESULTS.md but not explicitly implemented in aggregate script; per-layer n=14-15; L20 gen_acc all-zero differences.
- C4 WARN: M2 data covers only 2/3 planned depths (L28 absent); scope narrowed to predictive-only + SAGE-lite.

## Variant integrity (Phase 9)

| Claim | Variant | Exp. audit | Mech. audit | Combined | Integrity status | Detail |
|-------|---------|------------|-------------|----------|-----------------|--------|
| C4 | model-swap-gpt-oss-20b | PASS | N/A | PASS | clean | verify/C4_cross_pair_generalization/variant_audit/ |

> **Notes**: 1 variant run for C4 (DIMENSIONS=model, MAX_VERIFY_CLAIMS=1). Variant integrity PASS on all 7 checks (no fake GT, correct normalization, no phantom results, live metric code, correct scope, no test leakage, valid statistics). Variant excluded from robustness denominator: 0. Integrity-clean variants for C4: N_eligible=1.

**Phase 9 overall**: PASS — 0 integrity failures across 1 variant run.
