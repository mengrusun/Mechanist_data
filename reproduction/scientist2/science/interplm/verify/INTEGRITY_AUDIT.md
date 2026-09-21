# Integrity Audit

**Overall**: WARN (Phase 2 all-WARN; Phase 9 PASS)
**Main-experiment integrity (Phase 2)**: WARN  — all 6 claims WARN; no FAIL
**Variant integrity (Phase 9)**: PASS — 1 variant (c2 model-swap-esm2-8m) integrity CLEAN; 0 findings

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| c1 | WARN | N/A | WARN | continue-with-warn (experiment: auto-interp proxy GT + scope under-power) | verify/c1_sae_interpretable_feature_count/main_experiment_audit/ |
| c2 | WARN | N/A | WARN | continue-with-warn (experiment: scope under-power vs. target ~143) | verify/c2_sae_concept_alignment_gap/main_experiment_audit/ |
| c3 | WARN | N/A | WARN | continue-with-warn (experiment: strict margin not met; direction supported) | verify/c3_sae_superposition_specificity/main_experiment_audit/ |
| c4 | WARN | N/A | WARN | continue-with-warn (experiment: synonym-check GT proxy + scope + criterion non-discriminative) | verify/c4_novel_concept_discovery/main_experiment_audit/ |
| c5a | WARN | N/A | WARN | continue-with-warn (experiment: dead code in per_concept_pr_auc + scope + SGD substitution) | verify/c5a_annotation_filling_probes/main_experiment_audit/ |
| c5b | WARN | WARN | WARN | continue-with-warn (experiment: scope; mechanism: sweep <3 OOM / <5 grid points) | verify/c5b_feature_clamp_steering/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim.
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. `N/A` when the claim uses no mechanism intervention.
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a`; `n/a` treated as `pass`.
> - **Gate decision** — what Phase 2 does with this claim.

## Variant integrity (Phase 9)

**Scope:** claim c2 (only picked claim; c1, c3, c4, c5a, c5b deferred by MAX_VERIFY_CLAIMS cap)

| Claim | Variant | Exp. audit | Mech. audit | Combined | Integrity status | N_eligible contribution |
|-------|---------|------------|-------------|----------|-----------------|------------------------|
| c2 | model-swap-esm2-8m | PASS | N/A | PASS | CLEAN | +1 (eligible) |

**Phase 9 finding count:** 0 warnings, 0 failures
**N_eligible for c2:** 1 (all 1 variants passed integrity)
**Overall Phase 9:** PASS (no integrity failures; c2 variant is clean)
