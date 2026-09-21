# Integrity Audit

**Overall**: WARN    ← max severity across Phase 2 + Phase 9
**Main-experiment integrity (Phase 2)**: WARN    ← max severity across main-experiment combined verdicts
**Variant integrity (Phase 9)**: WARN    ← 1 variant, combined=WARN (exp=WARN, mech=N/A), 0 FAILs, 1 eligible

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1    | WARN       | N/A         | WARN     | continue-with-warn (experiment — scope overstatement; P3 skipped caveat not narrowed in claim header; P1b docstring mismatch) | verify/C1_concept_faithful_summary/main_experiment_audit/ |
| C2    | WARN       | N/A         | WARN     | continue-with-warn (experiment — fc P2c fails under heuristic grouping, H2d partially falsified on fc; P3 cross-model SKIPPED; scope overstates universality) | verify/C2_clip_joint_semantic_space/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim.
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. N/A means the claim uses no mechanism intervention (SemanticLens/CLIP-Dissect is purely observational — no steering, no CAA, no activation patching).
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a`. N/A treated as pass → combined = exp-side verdict.
> - **Gate decision** — what Phase 2 does with this claim. Both admitted.

## Variant integrity (Phase 9)

Only C2 entered Stage 2 (picked by Phase 3 step 0; C1 is INTEGRITY_ONLY, max_verify_claims_cap). One variant run: method-swap-crp-compose.

| Claim | Variant | Exp. audit | Mech. audit | Combined | Eligible (Y/N) | Detail |
|-------|---------|------------|-------------|----------|----------------|--------|
| C2    | method-swap-crp-compose | WARN | N/A | WARN | YES | verify/C2_clip_joint_semantic_space/variant_audit/ |

- WARN source (C2 / exp): scope narrowed to fc components under CRP-crop preprocessing — expected for method-swap; does not disqualify variant from robustness numerator/denominator
- N/A (mech): CRP-compose is observational; treated as pass for gate purposes
- N_eligible for C2: 1 (WARN is not FAIL; variant counts toward robustness)
- N_integrity_fail: 0

**Phase 9 overall**: WARN (1 variant, WARN, no FAILs)
