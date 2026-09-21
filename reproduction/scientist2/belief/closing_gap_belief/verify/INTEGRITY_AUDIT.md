# Integrity Audit

**Overall**: WARN
**Main-experiment integrity (Phase 2)**: WARN
**Variant integrity (Phase 9)**: PASS

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1 | WARN | N/A | WARN | continue-with-warn (experiment) | verify/C1_correctness_linear_accessible/main_experiment_audit/ |
| C2 | WARN | N/A | WARN | continue-with-warn (experiment) | verify/C2_verbalized_conf_linear/main_experiment_audit/ |
| C3a | PASS | N/A | PASS | continue | verify/C3a_near_orthogonality/main_experiment_audit/ |
| C3b | WARN | WARN | WARN | continue-with-warn (experiment+mechanism) | verify/C3b_causal_separability/main_experiment_audit/ |
| C3c | WARN | N/A | WARN | continue-with-warn (experiment) | verify/C3c_dissociation_when_disagree/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim.
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. `N/A` when the claim uses no mechanism intervention.
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a` and `n/a` treated as pass.
> - **Gate decision** — what Phase 2 does with this claim. The reason tag in parentheses points to which sub-audit drove a WARN.

### Per-claim WARN reasons

**C1 WARN (experiment)**: Bootstrap CI [0.818, 0.835] does not contain point estimate 0.840 — normal statistical artifact (bootstrap mean vs full-data) but flagged for consistency. Bootstrap n=200 instead of planned 1000 (disclosed). No FAIL criteria triggered.

**C2 WARN (experiment)**: Probe target is model's own verbalized confidence c (by design, disclosed — not fraud, but strict audit flags proxy target). Bootstrap CI [0.913, 0.940] does not contain point estimate 0.948. Paraphrase Delta AUC P1=-0.11 exceeds planned <=0.10 tolerance by 0.01 (AUC still above 0.70 floor, honestly reported).

**C3a PASS**: Clean across all checks. Derived analytical quantity (cosine of probe vectors); real GT for probe training; all artifact files exist and numbers match.

**C3b WARN (experiment+mechanism)**: Experiment WARN — primary metric is internal probe readout (pre-registered but weaker than behavioral GT); sparse steering controls. Mechanism WARN — sparse alpha grid (3 values, 1 order of magnitude; < 3 orders of magnitude WARN criterion); single random-direction control (n=1 << 30 recommended).

**C3c WARN (experiment)**: Load-bearing cell (probe_low, verbal_low) has n=4/2000. Test under-powered. Honestly reported as [suspected under-power] in EXPERIMENT_RESULTS.md. Not fabrication — a genuine null.

## Variant integrity (Phase 9)

**Variants audited**: 1  (1 pass, 0 warn, 0 fail)

### Per-claim audit verdicts

| Claim | Exp. audit (variants) | Mech. audit (variants) | Combined |
|-------|-----------------------|------------------------|----------|
| C3a   | PASS                  | N/A                    | PASS     |

### Findings (per variant)

- C3a / variant model-swap-qwen25-7b-instruct: [INTEGRITY: PASS] — all experiment checks pass; mechanism check N/A (no steering intervention in C3a). Variant included in robustness numerator and denominator.

**Audit log:**
```
[variant-audit] claim=C3a variant=model-swap-qwen25-7b-instruct exp_verdict=pass mech_verdict=n/a combined=pass
[variant-audit] claim=C3a variant=model-swap-qwen25-7b-instruct exp_source=verify/C3a_near_orthogonality/variant_audit/EXPERIMENT_AUDIT.json
[variant-audit] claim=C3a variant=model-swap-qwen25-7b-instruct mech_source=verify/C3a_near_orthogonality/variant_audit/MECHANISM_AUDIT.json
[variant-audit] claim=C3a variant=model-swap-qwen25-7b-instruct integrity_status=pass action=include
```
