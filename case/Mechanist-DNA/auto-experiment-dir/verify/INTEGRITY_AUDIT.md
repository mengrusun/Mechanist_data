# Integrity Audit

**Overall**: PASS
**Main-experiment integrity (Phase 2)**: PASS
**Variant integrity (Phase 9)**: PASS

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1 | PASS | N/A | PASS | continue | verify/C1_helix_feature_set_exists/main_experiment_audit/ |
| C2 | PASS | PASS | PASS | continue | verify/C2_dose_response_helix/main_experiment_audit/ |
| C3 | PASS | PASS | PASS | continue | verify/C3_specificity_causal_knob/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — overall_verdict of /experiment-audit on this claim.
> - **Mech. audit** — overall_verdict of /mechanism-audit on this claim. N/A when no mechanism intervention.
> - **Combined** — max_severity(exp, mech) with fail > warn > pass > n/a and n/a treated as pass.
> - **Gate decision** — what Phase 2 does with this claim.

## Variant integrity (Phase 9)

**Variants audited**: 3  (3 pass, 0 warn, 0 fail)

### Per-claim audit verdicts

| Claim | Exp. audit (variants) | Mech. audit (variants) | Combined |
|-------|-----------------------|------------------------|----------|
| C1 | PASS | N/A | PASS |
| C2 | PASS | PASS | PASS |
| C3 | PASS | PASS | PASS |

### Findings (per variant)
- C1 / model-swap-h-only-helix-def: [INTEGRITY: PASS] — clean. H-only labels same experimental DSSP pipeline; AUROC computation unchanged; single swap.
- C2 / model-swap-h-only-helix-frac: [INTEGRITY: PASS] — clean. helix_h_w_mean from same structure prediction pipeline; Spearman rho on same held-out seeds; single endpoint swap. Minor caveat: ESMFold pLDDT-threshold sweep was null in main experiment but this variant does not claim to populate it.
- C3 / model-swap-h-only-specificity: [INTEGRITY: PASS] — clean. Same 48 null directions; same arm data; single endpoint swap. Noted limitation: no cluster-bootstrap CI for H-only (aggregate only) documented in DIFF.md; does not constitute a FAIL since empirical p-value test is valid and primary.
