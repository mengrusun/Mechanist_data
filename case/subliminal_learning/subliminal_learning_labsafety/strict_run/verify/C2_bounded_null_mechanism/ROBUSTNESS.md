## C2: Phase 2 FAIL — main-experiment mechanism rigor broken  →  INCONCLUSIVE

- verdict: INCONCLUSIVE
- inconclusive_reason: main-experiment mechanism rigor broken — see verify/C2_bounded_null_mechanism/main_experiment_audit/MECHANISM_AUDIT.md
- swap_variants_run: false
- Main-experiment verdict on C2: not-supported (BOUNDED_NULL)
- Main-experiment integrity (Phase 2): FAIL (exp=WARN, mech=FAIL, combined=FAIL)
- Variants: none (Stage 2 skipped — INCONCLUSIVE at Phase 2 gate)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C2's mechanism experiment (M1+M2 — CAA steering arc) triggered Check A of the mechanism-audit and returned FAIL on two hard-fail criteria: (1) no independent capability/coherence metric was logged at any of the 7 alpha sweep points (only QA_I accuracy), making OOD collapse undetectable; (2) the sign/monotonicity pattern is broken — median Spearman rho = +0.371 (2/3 seeds show positive rho, the wrong direction under the v = mean(h_treated) - mean(h_Ctrl-B) convention). The experiment-audit returned WARN (external reviewer flagged incomplete artifact traceability in the audit prompt; executor verification confirmed all files and call chains are sound, but WARN is preserved per reviewer-independence protocol). Combined Phase 2 verdict = FAIL (driven by mechanism-audit FAIL). 

Iteration instruction: fix the mechanism-rigor issues before re-verifying C2:
1. Add a capability/coherence metric (e.g., OTHER rate from per-item verdicts, or perplexity on a holdout) to `steer_and_eval.py` and re-run M2.2b with it logged.
2. Increase n_random >= 30 for M2.2c specificity control.
3. Document the sign inversion as the primary BOUNDED NULL evidence in the paper.
Do NOT change the claim (C2 explicitly allows BOUNDED NULL). Do NOT re-run the main M0 experiment. Only the mechanism rigor controls need updating.
