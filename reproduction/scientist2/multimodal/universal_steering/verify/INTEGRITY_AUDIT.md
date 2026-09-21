# Integrity Audit

**Overall**: WARN — max severity across Phase 2 (main experiment) + Phase 9 (variants, filled below)
**Main-experiment integrity (Phase 2)**: WARN — max severity across per-claim combined verdicts
**Variant integrity (Phase 9)**: [filled by Phase 9 — only for claims admitted by Phase 2]

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1 | WARN | WARN | WARN | continue-with-warn (experiment: alpha_star labelling cosmetic + scope 3-scenario overclaim; mechanism: degenerate block selection for trivially-separable concepts + alpha on same held-out set) | verify/C1_political_rfm_steering/main_experiment_audit/ |
| C2 | WARN | PASS | WARN | continue-with-warn (experiment: suspected_under_power n=10 held-out, stat power insufficient) | verify/C2_cpp_steering_hackerrank/main_experiment_audit/ |
| C3 | WARN | PASS | WARN | continue-with-warn (experiment: no multiple-comparisons correction; FR sign reversal breaks claim's 'all 3 langs' predicate) | verify/C3_crosslingual_honesty_vector/main_experiment_audit/ |
| C4 | WARN | WARN | WARN | continue-with-warn (experiment: measurement-ceiling failure masks compositional gain; mechanism: missing random-direction control) | verify/C4_compositional_steering/main_experiment_audit/ |
| C5 | PASS | N/A | PASS | continue | verify/C5_internal_monitor_beats_gpt/main_experiment_audit/ |

> **Column glossary:**
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` (methodology, GT, normalization, scope).
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` (steering coefficient sweep, controls). `N/A` when the claim uses no additive steering intervention (C5 uses classifier mode).
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a`; `n/a` treated as `pass` (severity=0). All 5 claims are admitted (PASS or WARN); no claim received FAIL.
> - **Gate decision** — Action for Stages 2–3.

**Summary**: All 5 claims admitted (Phase 2 combined = WARN or PASS). No claim marked INCONCLUSIVE. MAX_VERIFY_CLAIMS=1 will gate Stage 2 to top-1 claim by importance (Phase 3 step 0).

## Variant integrity (Phase 9)

| Claim | Variant | Exp. audit | Mech. audit | Combined | Eligible? | Detail |
|-------|---------|------------|-------------|----------|-----------|--------|
| C5 | model-swap-deepseek-r1-llama8b | PASS | N/A | PASS | yes | verify/C5_internal_monitor_beats_gpt/variant_audit/ |

> **N_eligible = 1** (the sole variant passed integrity). It is included in both numerator and denominator of the robustness calculation.
>
> Variant result: `variant_claim_supported=False` (HaluEval internal=0.985 > GPT-4o=0.685, ToxicChat internal=0.858 < GPT-4o=0.882; strict AND fails).
> Consistency with main experiment (supported): **fail** — variant disagrees.
>
> **Phase 9 overall: PASS** (no integrity failures; variant evaluation is methodologically clean)
