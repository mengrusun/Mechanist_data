## C1: audit-only (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C1: not-supported (AUC >= 0.75 passes; first-PC |cos| >= 0.70 fails for all four behaviours — joint predicate fails)
- Main-experiment integrity: WARN (exp=WARN — LLM-judge proxy annotation; mech=N/A — M1 uses probe only, no additive steering)
- warn_source: experiment (LLM-judge proxy GT, not human-annotated or dataset GT)
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology (evaluation and mechanism) was audited. The evaluation uses LLM-judge proxy labels which is appropriate for this type of study but carries a WARN for non-oracle GT. No steering intervention is used in M1 (mechanism audit = N/A). The main-experiment verdict on C1 (not-supported at the joint predicate: first-PC alignment fails for all four behaviours while AUC passes) stands as-is with this integrity caveat. No swap stress test was attempted this pass — C4 was selected as the top-1 claim under MAX_VERIFY_CLAIMS=1. To upgrade C1 to full verify without re-auditing:

`/auto-verify C1 — resume: true` (single-claim mode; Phase 2 audit is reused via RESUME; Stages 2-3 execute for this one claim)
