# Integrity Audit

**Overall**: FAIL    ← max severity across main experiment + variants (C1 main-experiment FAIL)
**Main-experiment integrity (Phase 2)**: FAIL    ← max severity across per-claim combined verdicts (C1=FAIL)
**Variant integrity (Phase 9)**: WARN    ← C2 variants audited (see below)

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1 (RE-AUDIT after ② fix) | FAIL (Check-E claim-scope only; Check F repaired fail→warn) | WARN | FAIL | INCONCLUSIVE→**FAIL (claim-support)** — narrow claim (③) | verify/C1_causal_helix_steering/main_experiment_audit/ |
| C2    | WARN       | N/A         | WARN     | continue-with-warn (experiment)  | verify/C2_eval_harness_fidelity/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on the claim's main-experiment evidence.
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit`. `N/A` when no mechanism intervention (C2 is a pure eval-harness study).
> - **Combined** — `max_severity(exp, mech)`, `n/a` treated as pass.
> - **Gate decision** — C1 combined=FAIL → short-circuits to INCONCLUSIVE (variants never run); C2 combined=WARN → admitted to Stage 2 with a warn caveat.

**C1 status (superseded — RE-AUDIT after iteration-loop type-② fix):** was INCONCLUSIVE (endpoint untrustworthy). The type-② fix (milestone M5, `runs/iteration_round_1/M5_structural_gc_control.json`) REPAIRED the endpoint — Check F fail→warn (uplift triangulated across two ESM2-independent predictors; composition confound + non-monotonicity disclosed; labeled non-structural). A main-experiment verdict is now computable and is **not-supported**: the +%H is a composition-level metric effect (Lys/low-complexity/GC-collapse), not surviving (underpowered) GC control, not structurally validated. Residual overall FAIL is now **Check-E claim-scope only** (the frozen claim overclaims). New state: **FAIL (claim-support)** → route to claim narrowing (③). See verify/C1_causal_helix_steering/{ROBUSTNESS.md, main_experiment_audit/EXPERIMENT_AUDIT.md}.

## Variant integrity (Phase 9)

**Variants audited**: 2  (C2 method-swap + dataset-swap)  — 0 fail, 2 warn, 0 pass. All eligible (integrity WARN counts normally).

### Per-claim audit verdicts

| Claim | Exp. audit (variants) | Mech. audit (variants) | Combined |
|-------|-----------------------|------------------------|----------|
| C2    | WARN (scope)          | N/A                    | WARN     |

### Findings (per variant)
- C2 / method-swap-gor-windowed: [INTEGRITY: WARN — experiment] — scope-only WARN (finite 200-protein set, one run). GT=experimental DSSP; r=0.8505 verified on disk; eligible, counted.
- C2 / dataset-swap-heldout-proteins: [INTEGRITY: WARN — experiment] — scope-only WARN (finite 310-protein set, one run). GT=experimental DSSP; r=0.9792 verified on disk; eligible, counted.

> Note: an initial Phase 9 audit pass returned FAIL on Check C (result existence) purely because the on-disk result.json files had not been shown to the external reviewer; re-audited with the artifacts provided, Check C = PASS and the combined verdict is WARN (scope only). No integrity-FAIL variants; N_eligible = 2/2.
