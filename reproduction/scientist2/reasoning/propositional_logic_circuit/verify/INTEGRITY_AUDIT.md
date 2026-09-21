# Integrity Audit

**Overall**: WARN    ← max severity across main experiment + variants
**Main-experiment integrity (Phase 2)**: WARN    ← max severity across main-experiment combined verdicts
**Variant integrity (Phase 9)**: PASS

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision                        | Detail |
|-------|------------|-------------|----------|--------------------------------------|--------|
| C1    | WARN       | N/A         | WARN     | continue-with-warn (experiment)      | verify/C1_sparse_component_set/main_experiment_audit/ |
| C2    | WARN       | N/A         | WARN     | continue-with-warn (experiment)      | verify/C2_modular_decomposition/main_experiment_audit/ |
| C3    | WARN       | N/A         | WARN     | continue-with-warn (experiment)      | verify/C3_necessity_sufficiency/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim.
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. `N/A` when the claim uses no mechanism intervention (all three claims here use replacement activation patching, which does not trigger the steering-coefficient sweep check).
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a` and `n/a` treated as `pass`.
> - **Gate decision** — what Phase 2 does with this claim. The reason tag in parentheses points to which sub-audit drove a WARN.

**All three claims admitted (WARN). WARNED issues per claim:**
- **C1**: undisclosed minimality sampling (20/158 components sampled; approximation not surfaced in per-claim verdict text).
- **C2**: shortlist scope reduction (top-40 of 158 run; explicitly disclosed in Notes but creates mismatch with claim predicate which references "the shortlisted components" i.e. all 158).
- **C3**: KL recovery scaling artifact (baseline KL=0.043 causes recovery_KL=-5.618 in M2; documented in Notes but not elevated to per-claim verdict box; plan criterion requires all three metrics).

## Variant integrity (Phase 9)

**Variants audited**: 1  (1 pass, 0 warn, 0 fail)

### Per-claim audit verdicts

| Claim | Exp. audit (variants) | Mech. audit (variants) | Combined |
|-------|-----------------------|------------------------|----------|
| C3    | PASS                  | N/A                    | PASS     |

### Findings (per variant)
- C3 / model-swap-gemma2-9b: [INTEGRITY: PASS] — GT from dataset JSONL, normalization correct, result file exists (results/M5_gemma9b.json), no dead code, full-shortlist necessity/sufficiency metrics correct, no scope issues, no steering coefficient (replacement patching). Integrity clean.
