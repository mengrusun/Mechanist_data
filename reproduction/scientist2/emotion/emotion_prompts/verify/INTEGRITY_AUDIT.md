# Integrity Audit

**Overall**: WARN
**Main-experiment integrity (Phase 2)**: WARN
**Variant integrity (Phase 9)**: PASS (1/1 variants clean)

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1 | WARN | N/A | WARN | continue-with-warn (experiment) | verify/C1_small_input_dependent_shift/main_experiment_audit/ |
| C2 | WARN | N/A | WARN | continue-with-warn (experiment) | verify/C2_task_family_spread_ordering/main_experiment_audit/ |
| C3a | PASS | N/A | PASS | continue | verify/C3a_no_consistent_winner/main_experiment_audit/ |
| C3b | WARN | N/A | WARN | continue-with-warn (experiment) | verify/C3b_no_monotone_intensity/main_experiment_audit/ |
| C4 | FAIL | N/A | FAIL | INCONCLUSIVE (experiment broken — M7 descoped, zero evidence) | verify/C4_adaptive_policy_beats_fixed/main_experiment_audit/ |
| CM | WARN | WARN | WARN | continue-with-warn (experiment+mechanism) | verify/CM_residual_direction_causal/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim.
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. `N/A` when the claim uses no mechanism intervention.
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a` and `n/a` treated as pass.
> - **Gate decision** — what Phase 2 does with this claim. The reason tag in parentheses points to which sub-audit drove a WARN/FAIL.

### Warn details by claim

- **C1 (experiment WARN)**: sign-consistency predicate of C1 was not computed — EXPERIMENT_RESULTS.md explicitly skipped this half of C1's required evidence. Noise-floor threshold half is clean.
- **C2 (experiment WARN)**: cross-task spread comparison confounds evaluation modes (CoT for GSM8K vs MCQ log-likelihood for SocialIQA/MedQA). The larger GSM8K spread is partly attributable to CoT generation variance, not purely emotional framing.
- **C3b (experiment WARN)**: passes at exactly threshold (3/6 emotions); no per-emotion CI bounds reported in EXPERIMENT_RESULTS to assess marginality of the 3rd qualifying emotion.
- **C4 (experiment FAIL)**: M7 descoped entirely — no result files, no evaluation run, no GT used. All M7 scripts are dead code relative to actual execution. This is the correct FAIL for "experiment not carried out at all."
- **CM (experiment WARN + mechanism WARN)**: M6 scope gaps (9/13 runs; 1/12 emotional prefixes; 0/3 specificity controls; 50 items vs 200 planned). Mechanism WARN: steering sweep spans only ~2 orders of magnitude; no random-direction control; parse_rate is task-internal fluency proxy, not an independent capability metric.

## Variant integrity (Phase 9)

Phase 9 audits variants of the one claim selected for Stage 2 (CM, picked by Phase 3 step 0).

| Claim | Variant tag | Exp. audit | Mech. audit | Combined | integrity_status | Detail |
|-------|-------------|------------|-------------|----------|------------------|--------|
| CM | model-swap-qwen3-4b | PASS | N/A | PASS | clean | verify/CM_residual_direction_causal/variant_audit/ |

**N_run = 1, N_eligible = 1** (0 excluded by integrity FAIL)

### Variant integrity details

**CM / model-swap-qwen3-4b**:
- **Exp. audit PASS**: GT from HF GSM8K (same source as main experiment); accuracy = n_correct/n_items exact match; all 7 result files present, 50/50 items each; no dead code; Location arm scope stated correctly; eval mode consistent (cot, temperature=0.0). Note: solver changed from lbfgs to liblinear in probe step (equivalent results; liblinear is standard for this scale and was used to avoid sklearn convergence timeout with 2560-dim features).
- **Mech. audit N/A**: Variant runs probing only — no steering, no patching, no causal intervention. Check A does not apply.
- **Combined**: max_severity(pass, n/a) = pass → integrity_status = clean
- **Included in robustness computation**: YES (N_eligible += 1)
