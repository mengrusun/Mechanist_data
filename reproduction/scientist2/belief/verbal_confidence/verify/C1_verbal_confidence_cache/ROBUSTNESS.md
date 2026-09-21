## C1: INCONCLUSIVE — main-experiment mechanism rigor broken (Phase 2)

- verdict: INCONCLUSIVE
- swap_variants_run: false
- Main-experiment verdict on C1: not-supported
- Main-experiment integrity: FAIL (exp=WARN, mech=FAIL → combined=FAIL)
- inconclusive_reason: main-experiment mechanism rigor broken — see verify/C1_verbal_confidence_cache/main_experiment_audit/MECHANISM_AUDIT.md
- Variants: none (Stage 2 skipped — combined Phase 2 audit = FAIL)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: The experiment-audit (WARN) flagged P2 ratio aggregation ambiguity, P5 cross-seed inconsistency (M6c seed42=3.57 vs seed123=0.10), and the trivially-preserved answer_acc metric. The mechanism-audit (FAIL) found that the M5 steering sweep has no independent capability metric (only off_digit_rate as a coarse proxy), the target effect is entirely within the baseline-noise floor (span ≈ 1–1.5 verbal-conf units across α∈[-4,+4]×σ_proj while per-item std≈41), no locked α mid-plateau was selected, and no random-direction control was run. The combined Phase 2 verdict is FAIL, so C1 is marked INCONCLUSIVE: computing robustness around this broken anchor would be meaningless. Iteration must fix the mechanism-rigor issues first — specifically: (1) add a real capability metric at every α, (2) log raw post-steering samples, (3) run a random-direction control with n_random≥30, and (4) resolve why the direction at E4L10 shows negligible causal effect despite strong probe R² — the disconnect between correlational (P1 PASS) and causal (P2–P4 FAIL) findings is the core scientific finding.

To re-run verify after fixing the mechanism sweep:
  `/auto-verify C1 — resume: false`  (re-run Phase 2 after correcting M5 and adding a capability metric)
