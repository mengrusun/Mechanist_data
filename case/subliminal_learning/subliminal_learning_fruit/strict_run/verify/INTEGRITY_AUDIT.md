# Integrity Audit

**Overall**: WARN    <- max severity across main-experiment audits (Phase 2); Phase 9 not run
**Main-experiment integrity (Phase 2)**: WARN    <- max severity across C1 combined=WARN, C2 combined=WARN
**Variant integrity (Phase 9)**: n/a (skipped)

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision                       | Detail |
|-------|------------|-------------|----------|-------------------------------------|--------|
| C1    | WARN       | N/A         | WARN     | continue-with-warn (experiment)     | verify/C1_subliminal_transfer_diffusion/main_experiment_audit/ |
| C2    | WARN       | WARN        | WARN     | continue-with-warn (experiment+mechanism) | verify/C2_mechanism_carries_bias/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim.
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. `N/A` when the claim uses no mechanism intervention.
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a` and `n/a` treated as pass.
> - **Gate decision** — what Phase 2 does with this claim. The reason tag in parentheses points to which sub-audit drove the WARN.

### C1 Phase 2 Detail

**Experiment audit (WARN)**:
- Check A (GT provenance): PASS — gpt-5.4 judge at T=0.0, task-specified, used symmetrically for filtering and evaluation.
- Check B (score normalization): PASS — p_banana = banana_count / 160 (fixed denominator, not model-dependent).
- Check C (result file existence): WARN — all 17 p_banana.json files exist and numbers are consistent, BUT verdict.json reports `conditional` while judge_and_verdict.py code returns `inconclusive` when residue>0. Documented protocol deviation with scientific justification (judge-noise-floor residue).
- Check D (dead code): PASS — all metric functions on active execution path.
- Check E (scope): PASS — 8 seeds × 160 prompts × 3 arms = 2720 samples; no overclaim.
- Check F (eval type): task_specified_proxy (designated measurement instrument from task.md).

**Mechanism audit (N/A)**:
- C1 is a phenomenon-validation claim (M0 only); no additive activation intervention or scalar coefficient anywhere in the M0 code path. Mechanism audit does not apply.

### C2 Phase 2 Detail

**Experiment audit (WARN)**:
- Check A (GT provenance): PASS — M1 uses LoRA ΔW SVD structure (internal); M2 uses task-specified gpt-5.4 judge, same as M0.
- Check B (score normalization): PASS — p_banana = banana_n / 160 (fixed denominator).
- Check C (result file existence): WARN — 12/12 scoped runs have verdict.json; numbers consistent with EXPERIMENT_RESULTS.md. WARN: planned 8×5=40 runs reduced to 4×3=12 runs (budget-motivated, documented). Dose-response curve is single-point (amplify_x3 only); Spearman-rho monotonicity check not computable.
- Check D (dead code): PASS — intervene_and_eval.py and location_screen.py fully live; artifacts on disk confirm execution.
- Check E (scope): WARN — plan success criterion requires >=6/8 seeds; realized 4 seeds. Claim correctly reported as not-supported [provisional — under-power]; no overclaim.
- Check F (eval type): task_specified_proxy_and_internal_svd (M1 internal ΔW analysis; M2 task-designated judge).

**Mechanism audit (WARN)**:
- Check A (steering coefficient sweep): WARN
  - sigma_proj scaling: PASS (implemented correctly in _estimate_sigma())
  - Sweep cardinality >=3: FAIL (amplify_x3 only — 1-point dose curve; planned amplify_x{2,3,4} not executed)
  - Capability metric logged: PASS (fluency in every verdict.json; 0.80-0.98)
  - Alpha locked mid-plateau: NOT_EVALUABLE (root cause: cardinality gap)
  - Matched-random baseline: PASS (random_ablate with same-norm Gaussian direction, seed_rand=42)
  - Sign pattern preserved: PASS (consistent with M1 direction extraction)
  - Grade: WARN (not FAIL) — sigma_proj calibration and matched-random control (most critical safeguards) both pass; gap is cardinality only.
- Checks B-F: not_implemented (reserved).

## Variant integrity (Phase 9)
[skipped — Stage-2 variant not run for budget reasons; see VERIFY_REPORT.md]
