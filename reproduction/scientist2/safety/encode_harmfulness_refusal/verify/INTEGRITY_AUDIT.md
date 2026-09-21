# Integrity Audit

**Overall**: FAIL
**Main-experiment integrity (Phase 2)**: FAIL (C3 mechanism broken; C5 experiment scope broken)
**Variant integrity (Phase 9)**: WARN (1 variant audited; 1 warn, 0 fail — NaN auroc_r expected)

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1 | WARN | N/A | WARN | continue-with-warn (experiment: scope/proxy-GT) | verify/C1_h_r_directions_distinct/main_experiment_audit/ |
| C2 | WARN | N/A | WARN | continue-with-warn (experiment: scope/unmeasurable-crossover) | verify/C2_position_dissociation/main_experiment_audit/ |
| C3 | WARN | FAIL | FAIL | INCONCLUSIVE (mechanism broken — steering coefficient sweep fails) | verify/C3_causal_steering_dissociation/main_experiment_audit/ |
| C4 | WARN | N/A | WARN | continue-with-warn (experiment: claim untestable at ASR=0) | verify/C4_jailbreak_signature/main_experiment_audit/ |
| C5 | FAIL | N/A | FAIL | INCONCLUSIVE (experiment broken — scope mismatch: jailbreak claim tested on bare-harmful data) | verify/C5_probe_beats_llamaguard/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim.
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. `N/A` when the claim uses no mechanism intervention.
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a` and `n/a` treated as `pass`.
> - **Gate decision** — what Phase 2 does with this claim. The reason tag in parentheses points to which sub-audit drove a WARN/FAIL.

**Summary of Phase 2:**
- ADMITTED (WARN): C1, C2, C4 — proceed to Phases 3–10
- INCONCLUSIVE (FAIL): C3 (mechanism audit FAIL — steering sweep rigor), C5 (experiment audit FAIL — scope mismatch: 0 jailbreaks in test set)

## Variant integrity (Phase 9)

**Variants audited**: 1 (1 pass/warn, 0 fail)

### Per-claim audit verdicts

| Claim | Exp. audit (variants) | Mech. audit (variants) | Combined |
|-------|-----------------------|------------------------|----------|
| C1 | WARN (1 variant — auroc_r=NaN expected) | N/A (no steering) | WARN |

### Findings (per variant)
- C1 / variant model-swap-qwen2-instruct-7b: [INTEGRITY: WARN — experiment] — auroc_r=NaN in result file (expected: Qwen2 also refuses nearly all bare-harmful prompts; refusal sub-test val set has near-zero positive class). Variant counted in eligible set (WARN is not FAIL).
