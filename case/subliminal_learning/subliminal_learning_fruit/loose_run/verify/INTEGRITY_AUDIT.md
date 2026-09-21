# Integrity Audit

**Overall**: WARN
**Main-experiment integrity (Phase 2)**: WARN
**Variant integrity (Phase 9)**: PASS

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1 | WARN | N/A | WARN | continue-with-warn (experiment) | verify/C1_subliminal_transfer_phenomenon/main_experiment_audit/ |
| C2 | PASS | N/A | PASS | continue | verify/C2_banana_signal_location/main_experiment_audit/ |
| C3 | WARN | WARN | WARN | continue-with-warn (experiment + mechanism) | verify/C3_causal_intervention_sites/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim.
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. `N/A` when the claim uses no mechanism intervention.
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a` and `n/a` treated as pass.
> - **Gate decision** — what Phase 2 does with this claim.

### C1 — WARN detail

Main WARN flag: `runs/M0_verdict.json` contains `original_algorithmic_verdict: "inconclusive"` but `phenomenon_status: "conditional"` — an agent verdict override was applied post-hoc. The override is documented and scientifically defensible (judge stochasticity, 2/154=1.3% residue, 5-pass filter converged to 0, effect is 13× the 5 pp bar), but represents a deviation from the pre-registered §M0 decision rule in `compute_verdict.py`. `task.md` explicitly flags this for verify to independently audit.

### C2 — PASS

All numbers verified in `runs/m1_location/shortlist.json`. Weight-space analysis has no circular GT. No normalization artifacts. Full scope (16 LoRA adapters, 60 blocks). No dead code.

### C3 — WARN detail

Two WARN flags:
1. Experiment-audit WARN: single seed (42 only), amplify_x3 null, no negative-dose amplification.
2. Mechanism-audit WARN: σ_l calibrated correctly; random-direction control run; incomplete dose grid (no negative α, one null positive run); single-seed limitation weakens specificity conclusion.

Both WARNs are honestly documented in EXPERIMENT_RESULTS.md and CLAIMS_LEDGER.md.

## Variant integrity (Phase 9)

Scope: C1 only (the one claim admitted to Stage 2 per MAX_VERIFY_CLAIMS=1 cap).

| Claim | Variant | Exp. audit | Mech. audit | Combined | Eligible? |
|-------|---------|------------|-------------|----------|-----------|
| C1 | method-swap-binary-judge | PASS | N/A | PASS | yes |

**N_run = 1, N_eligible = 1, integrity_failures = 0**

### C1 / method-swap-binary-judge — PASS detail

- Check A (GT provenance): PASS — no external GT; P(binary_yes) over existing PNGs; same proxy-eval design as main experiment
- Check B (score normalization): PASS — denominator is sample size N=160 (fixed), not model output statistics
- Check C (result existence): PASS — result.json and cost.json exist non-empty; numbers match run.log verbatim
- Check D (dead code): PASS — all functions called in execution chain
- Check E (scope): PASS — full 2720 eval PNGs + 154 teacher channel, no subsetting
- Check F (eval type): synthetic_proxy (same as main experiment)
- Mechanism: N/A — behavioral claim, no mechanism intervention in variant

Combined = max_severity(PASS, n/a) = PASS. Variant is integrity-clean and enters robustness denominator.
