# Integrity Audit

**Overall**: WARN    (max severity across Phase 2 + Phase 9)
**Main-experiment integrity (Phase 2)**: WARN    (max severity: C1=WARN, C2=PASS, C3=FAIL → WARN from admitted claims; FAIL from C3 marks it INCONCLUSIVE)
**Variant integrity (Phase 9)**: PASS    (1 variant, 0 integrity failures; C1 model-swap-judge-gpt4o eligible)

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1 | WARN | N/A | WARN | continue-with-warn (experiment scope) | verify/C1_cross_modal_subliminal_transfer/main_experiment_audit/ |
| C2 | PASS | N/A | PASS | continue | verify/C2_data_purity_precondition/main_experiment_audit/ |
| C3 | WARN | FAIL | FAIL | INCONCLUSIVE (mechanism broken) | verify/C3_low_dim_safety_substrate/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim.
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. `N/A` when the claim uses no mechanism intervention.
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a` and `n/a` treated as pass.
> - **Gate decision** — what Phase 2 does with this claim.

### C1 Phase 2 notes
- Exp. audit WARN: Check E — claim text requires per-seed unanimity (>= 3 seeds ALL pass >= 3 pp) but main experiment reports conditional (2/3 seeds). Correctly disclosed but creates scope tension.
- Mech. audit N/A: C1 is a behavioral phenomenon claim (M0). No mechanism intervention. severity=0.
- Combined = WARN → admitted to Stages 2-3.

### C2 Phase 2 notes
- Exp. audit PASS: All six checks clear. GT from dataset, counts correct, rescan correctly scoped to scrubbed training set.
- Mech. audit N/A: C2 is a data-purity precondition claim. No mechanism intervention. severity=0.
- Combined = PASS → admitted to Stages 2-3.

### C3 Phase 2 notes
- Exp. audit WARN: Check E — n=27 held-out under-powered; MMLU sub-predicate of C3b never tested. Correctly disclosed.
- Mech. audit FAIL: Check A.3 (MMLU capability metric not logged at any alpha point) + Check A.4 (no plateau in 5-point steering sweep, mid-plateau lock not possible). These are hard rigor failures for the CAA family.
- Combined = max_severity(WARN, FAIL) = FAIL → INCONCLUSIVE. Phases 3-10 skip for C3.
- ROBUSTNESS.md pre-written with verdict=INCONCLUSIVE, inconclusive_reason: main-experiment mechanism rigor broken.

## Variant integrity (Phase 9)

> Only C1 ran Stage-2 variants (MAX_VERIFY_CLAIMS=1 cap; C2 was deferred with stage2_skip_reason=max_verify_claims_cap).

| Claim | Variant tag | Exp. audit | Mech. audit | Combined | Eligible | Detail |
|-------|-------------|------------|-------------|----------|----------|--------|
| C1 | model-swap-judge-gpt4o | PASS | N/A | **PASS** | yes | verify/C1_cross_modal_subliminal_transfer/variant_audit/ |

**N findings**: 0 integrity failures, 1 eligible variant.

### C1 Phase 9 notes
- Exp. audit PASS: All six checks clear on the variant. Judge-model swap is methodology-clean: same gold GT, same metric, same scripts, same result files on disk. Separate cache prevents contamination.
- Mech. audit N/A: C1 variant makes no mechanism intervention (behavioral eval only). severity=0.
- Combined = PASS → variant is integrity-eligible.
- Robustness: 1/1 = 1.000 ≥ 0.5 → **C1 PASS**.
