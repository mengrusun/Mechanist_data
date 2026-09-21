# Integrity Audit

**Overall**: WARN  (max severity across Phase 2 main-experiment + Phase 9 variants)
**Main-experiment integrity (Phase 2)**: WARN  (max severity across per-claim combined verdicts)
**Variant integrity (Phase 9)**: PASS  (all 1 variant(s) eligible; 0 integrity-fail)

---

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1a | PASS | N/A | PASS | continue | verify/C1a_teacher_beats_baselines/main_experiment_audit/ |
| C1b | WARN | N/A | WARN | continue-with-warn (experiment — numeric citation mismatch; fine-level n=0 in JSON vs 0.817 in prose) | verify/C1b_teacher_hierarchy_monotonic/main_experiment_audit/ |
| C2a | PASS | N/A | PASS | continue | verify/C2a_aligned_spearman_gain/main_experiment_audit/ |
| C2b | WARN | N/A | WARN | continue-with-warn (experiment — fine-level 0 pairs makes per-level predicate partially unverifiable) | verify/C2b_perlevel_spearman_gain/main_experiment_audit/ |
| C2c | PASS | N/A | PASS | continue | verify/C2c_gain_alignment_specific/main_experiment_audit/ |
| C3 | PASS | N/A | PASS | continue | verify/C3_behavioural_uncertainty_match/main_experiment_audit/ |
| C4a | WARN | N/A | WARN | continue-with-warn (experiment — dataset substitution: plan panel Birds/UC-Merced/Colon/Aircraft not tested; scope mismatch) | verify/C4a_downstream_noninferior/main_experiment_audit/ |
| C4b | WARN | N/A | WARN | continue-with-warn (experiment — dataset substitution: BREEDS entity13/living17/non-living26/entity30 + ImageNet-A not tested; 3 of 5 substitute splits are not genuine OOD) | verify/C4b_ood_strict_improve/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim.
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. N/A = the claim's experiment uses no additive mechanism intervention; severity ordering gives n/a severity 0 (does not worsen combined verdict).
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a` and `n/a` treated as 0 severity.
> - **Gate decision** — all 8 claims admitted (PASS or WARN); no FAIL claims → no INCONCLUSIVE claims. MAX_VERIFY_CLAIMS=1 gate applies at Phase 3 Step 0 only.

**All 8 claims ADMITTED. Admitted pool → Phase 3 Step 0 picks top-1 by importance.**

---

## Variant integrity (Phase 9)

Variant integrity audit for C2a (model-swap-dinov2-vits) — the one Stage-2-picked claim.

| Claim | Variant tag | Exp. audit | Mech. audit | Combined | Integrity status | Detail |
|-------|-------------|------------|-------------|----------|-----------------|--------|
| C2a | model-swap-dinov2-vits | PASS | N/A | PASS | eligible | verify/C2a_aligned_spearman_gain/variant_audit/ |

> All checks passed. The variant uses identical evaluation code (spearmanr on THINGS heldout) with correct ViT-S architecture loading; confound isolation confirmed (single variable: student architecture). No findings that would exclude this variant from robustness computation.

**N_eligible = 1 / N_run = 1. No variants excluded from robustness denominator.**
