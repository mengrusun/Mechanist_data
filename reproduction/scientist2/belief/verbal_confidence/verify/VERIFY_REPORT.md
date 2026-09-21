# Verification Report

**Date**: 2026-07-13
**Swap variants**: false — INCONCLUSIVE at Phase 2; Stage 2 never reached (all claims FAIL Phase 2 audit)
**Dimensions tested**: model (Qwen 2.5 7B base — swap candidate; never deployed because Phase 2 failed)
**Threshold**: robustness ≥ 0.50, min eligible variants = 1
**Main-experiment integrity (Phase 2)**: C1 combined = FAIL (exp=WARN, mech=FAIL) — see INTEGRITY_AUDIT.md.

## Summary

| Claim | Statement (short) | Main-experiment verdict | Main-experiment integrity (Phase 2, combined) | Variant integrity (Phase 9, combined) | Eligible variants (post-audit) | Robustness | State | Notes |
|-------|-------------------|------------------------|----------------------------------------------|--------------------------------------|-------------------------------|------------|-------|-------|
| C1 | verbal confidence cache hypothesis | not-supported | FAIL (exp WARN / mech FAIL) | — (skipped) | — | — | 🟡 INCONCLUSIVE | main-experiment mechanism rigor broken (Phase 2) — fix M5 steering: add capability metric, log text samples, run random-direction control; then re-run /auto-verify |

> **Column glossary:**
> - **Main-experiment verdict** — main experiment's own conclusion on the claim (`not-supported` — 2/5 predicates pass; strong causal claim not supported).
> - **Main-experiment integrity (Phase 2, combined)** — `max_severity(exp=WARN, mech=FAIL) = FAIL`. Exp-WARN: P2 ratio aggregation ambiguous, P5 cross-seed inconsistent, answer_acc_preserved trivial by design, 2/3 seeds complete. Mech-FAIL: M5 has no independent capability metric; target effect within noise floor (span ~1.5 units vs std~41); no locked α; no random-direction control. The FAIL is driven entirely by the mechanism audit.
> - **Variant integrity (Phase 9, combined)** — skipped; no variants were ever run (Phase 2 FAIL short-circuits to INCONCLUSIVE for all claims, blocking Stage 2 entry).
> - **Eligible variants** — none (variants never dispatched).
> - **Robustness** — undefined (no eligible variants).
> - **State** — INCONCLUSIVE: the Phase 2 combined audit failed; running swap variants on top of a broken anchor would be meaningless.

## Integrity Audit

**Overall**: FAIL — see `verify/INTEGRITY_AUDIT.md` for full Phase 2 (main experiment) findings. Phase 9 (variant) section is `[skipped — all main-experiment audits FAIL]`.

## Stage-2 Selection

All admitted claims were zero (C1 failed Phase 2). No Stage-2 pick was made.

**Rejected at Phase 2 (→ INCONCLUSIVE):**
- C1: verbal-confidence cache hypothesis — main-experiment verdict: not-supported — mechanism rigor FAIL at Phase 2 — fix `/auto-verify C1 — resume: false` after correcting M5

## Details

- C1: `verify/C1_verbal_confidence_cache/ROBUSTNESS.md` (INCONCLUSIVE)
- C1 Phase 2 audit: `verify/C1_verbal_confidence_cache/main_experiment_audit/`
  - `EXPERIMENT_AUDIT.md` / `EXPERIMENT_AUDIT.json` — overall_verdict: warn
  - `MECHANISM_AUDIT.md` / `MECHANISM_AUDIT.json` — overall_verdict: fail

## Next Step

→ **C1 is INCONCLUSIVE** (Phase 2 main-experiment mechanism rigor broken) → hand back with `/auto-iteration-loop "verbal confidence cache — verify-inconclusive: C1"`.

The instruction to iteration is explicit: **fix the failing mechanism audit; do not change the claim**.

Specifically, read `inconclusive_reason` → `main-experiment mechanism rigor broken`:
  1. Re-run M5 (`scripts/m5_steer.py`) with a proper capability/coherence metric logged at every α (e.g., per-item fluency score, token-entropy proxy, or unrelated-task accuracy). `off_digit_rate` alone does not qualify.
  2. Log raw post-steering text samples (≥5 per α value) so reviewers can verify that metric "effects" correspond to actual behavior changes.
  3. Run a random-direction control (n_random ≥ 30 random unit vectors at the same α sweep) to establish whether the trained direction statistically beats random at the current effect sizes.
  4. Optionally: investigate why E4L10 shows strong probe R² (P1 PASS) but negligible causal effect across M3/M4/M5 — this dissociation is the core finding and should be characterized as a deliberate negative result rather than a methodology failure.
  5. After fixing M5 and re-running: invoke `/auto-verify C1 — resume: false` (re-run Phase 2 with the corrected mechanism sweep; Phase 2 cannot be resumed because MECHANISM_AUDIT artifacts need regeneration with the corrected experiment).

Do NOT attempt a claim-stage rewrite or change the main-experiment verdict at this stage — the iteration loop routes back-edges based on verdict type, and INCONCLUSIVE's back-edge is "fix the main experiment, then re-verify."
