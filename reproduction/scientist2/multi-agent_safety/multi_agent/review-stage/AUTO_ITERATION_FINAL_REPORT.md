# Auto Iteration Final Report — Group-Aggregated Activation Probes for Multi-Agent Collusion Detection

- **Generated**: 2026-07-14T05:30:00
- **Iterations consumed**: 1 / 6
- **Claim-reentries consumed**: 1 / 2
- **Final reviewer score**: 6 / 10
- **Final canonical verdict**: almost
- **Termination reason**: positive_verdict
- **Cumulative cost**: runs_total=0, gpu_hours_total=0.00
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md)

---

## Executive Summary

The loop terminated positively after 2 iterations (1 back-edge action, 1 review-only). The single action was an iteration-1 type-③ lightweight in-loop claim rewrite that narrowed C1 to C1_v2 to match the realized evidence: restricted to Qwen3-32B-AWQ, K=3 committee, realized in-domain benchmark, and short-transcript setting, with explicit disclaimers about layer-selection instability and no claim of robust model-agnostic superiority. Iteration 2 confirmed the rewrite was scientifically honest (not rhetorical), scored 6/10 "almost", and recommended only paper-side narrative polish for the writing phase. C2 and C3 remain INTEGRITY_ONLY (main-experiment audit passed but Stage-2 swap-test was skipped at MAX_VERIFY_CLAIMS=1 cap) and are surfaced as Open Items with the standard `/auto-verify <id> — resume: true` upgrade path. Zero GPU-hours consumed in iteration (well within ~2.59 GPU-h remaining).

### Claim Disposition Overview

| Original state | # Claims | Final status after iteration |
|---|---|---|
| PASS                     | 0 | — |
| FAIL                     | 0 | — |
| INCONCLUSIVE             | 0 | — |
| ZERO_ELIGIBLE_VARIANTS   | 1 | 0 still ZERO / 1 rewritten to C1_v2 (scope-fitted PASS-analogue) |
| INTEGRITY_ONLY           | 2 | 2 held (C2, C3) — Open Items with per-claim upgrade path |
| DEFERRED (legacy)        | 0 | — |

---

## Section 1 — PASS Claims (brief audit)

None at loop start. Iteration 2's Verify-Passed subsection covered C1_v2 (the rewritten claim) as a scope-fit PASS-analogue, but it did not originate in `verify_passed`, so it appears in Section 2's rewrite journey instead.

---

## Section 2 — FAIL Claims (full journey)

None at loop start. `verify_failed` bucket was empty.

---

## Section 3 — INCONCLUSIVE Claims (main-experiment-fix journey)

None at loop start. `verify_inconclusive` bucket was empty.

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims (variant-fix journey)

### 4.1 `C1` (→ `C1_v2`) — Existence of internal collusion signal recoverable by a linear probe

**Original ZERO_ELIGIBLE_VARIANTS reason** (from `verify/C1_probe_existence_signal/ROBUSTNESS.md`'s `stage2_skip_reason`):
- `budget_exceeded`: the picked model-swap variant (Qwen3-32B-bf16 on the same 282-scenario benchmark) needed ~10.45 GPU-h to run to completion, exceeding the ~2.59 GPU-h remaining under the 10 GPU-h HARD cap. Sanity partial (2/6 scenarios extracted cleanly at 0.03 scen/s, model loaded correctly at d_model=5120 / 64 layers / 219.8s) confirmed the variant was technically valid; only budget prevented completion.

**Variant script modifications**

| Iter | Variant | Component | Before | After |
|---|---|---|---|---|
| — | (none — no variant script edits) | — | — | — |

The iteration path did NOT edit the variant script or re-attempt the model swap. The reviewer's iteration-1 assessment was that even a budget-fitting cross-family swap (GPT-OSS-20B, ~1.5–2 GPU-h) would be "a risky spend because it may not resolve the core fairness/truncation issue and could still leave the selected-layer threshold unmet". The type-③ scope-narrowing path was chosen instead.

**Claim modifications** (via iteration 1 lightweight type-③ rewrite)

- Original claim id `C1`: "Model internals carry a collusion signal that text-level monitoring misses; a lightweight linear probe on per-agent residual-stream activations of Qwen3-32B-AWQ can recover it, beating a text-only gpt-5.4 judge on the committee bench."
- After iteration 1 → new claim id `C1_v2`: "In Qwen3-32B-AWQ K=3 committee deliberations on the realized in-domain benchmark, residual-stream activations contain a detectable collusion signal recoverable by a lightweight linear probe, achieving competitive scenario-level discrimination relative to a text-only judge under the short-transcript setting used here; however, effect size and optimal layer are unstable under the current small dev/test splits, so we do not claim robust model-agnostic superiority."
- Scope changes: (a) explicit single-model restriction (Qwen3-32B-AWQ only, drops implicit model-agnostic scope); (b) explicit setup restriction (K=3, short-transcript, realized in-domain benchmark); (c) explicit disclaimer of layer-selection instability; (d) removed "beats text-only judge" phrasing (unfair vs 71%-OTHER-abstaining baseline) in favor of "competitive scenario-level discrimination ... under the short-transcript setting used here"; (e) explicit non-claim of "robust model-agnostic superiority".

**Iteration journey**

| Iter | Type | Reviewer flag | Action | Outcome |
|---|---|---|---|---|
| 1 | ③ lightweight in-loop rewrite | "The minimum credible path is scope narrowing, not another shaky empirical rescue" | Adopted reviewer's proposed C1_v2 text verbatim; no script edits, no upstream skill calls | C1_v2 scope-fitted to existing M1 numbers (0.665 at L48, 0.750 at L27); ZERO_ELIGIBLE_VARIANTS resolved by scope alignment. Claim-reentry sub-budget: 1/2 consumed. |
| 2 | ⓪ narrative-only | "Make sure abstract/introduction/conclusion consistently reflect the setting-qualified scope" | Recommendation recorded for paper-write phase; no compute; no budget consumed | STOP fired: score 6, verdict "almost", no non-INTEGRITY_ONLY claim outstanding. |

**Path taken (summary)**: type-③ lightweight in-loop rewrite (1 hop) → confirmatory ⓪ review — claim-reentry sub-budget used: 1

**Re-verify outcome**

| Iter | Path | Action | Result |
|---|---|---|---|
| 1 | none — rewrite fits existing evidence exactly | no new `/auto-verify` invocation needed | C1_v2 is a strict subset of what M1 already demonstrates; no new robustness data required for iteration-loop STOP purposes |
| 2 | consistency check by reviewer only | narrative-only | scope-fit PASS-analogue confirmed |

**Final experiment summary**
- New runs cited: none (0 GPU-h consumed in iteration)
- Final scope-fit status: **PASS-analogue (under new claim `C1_v2`)** — 0.665 at dev-selected L48, 0.750 at test-best L27 both fit "competitive scenario-level discrimination ... under the short-transcript setting used here"; length-match 0.642, topic-swap 0.651, and label-permute 0.412 sanity controls stand.
- Final status: **PASS-analogue (under new claim `C1_v2`)**
- Open scientific concerns (not blocking, surfaced in Section 8): probe-vs-artifact separation still only partial; short-transcript regime not empirically stressed; no cross-model evidence.

**Reviewer memory thread** (cross-iteration suspicions filtered to this claim's pattern)
- Judge-baseline fairness (71% OTHER on 49 rows): **partially addressed** at wording level, **not addressed** at evidence level.
- Post hoc layer cherry-picking: **addressed** at claim level (C1_v2 explicitly disclaims stable layer choice); instability itself scientifically unresolved.
- Aspirational vs realized benchmark mismatch: **genuinely resolved** — C1_v2 anchors to realized-in-domain wording.
- Semantics vs length/topic artifacts: **not addressed** empirically; no new evidence; rewrite avoids overclaim.

---

## Section 5 — Legacy DEFERRED Claims (empty under current architecture)

Not applicable — the current run's verify report has no legacy `## Deferred Claims` section.

---

## Section 6 — Cross-Cutting Patterns

- **Transfer (C3) result is stronger than the in-domain foundational result (C1)** — flagged in iteration 1, discussed again in iteration 2. Resolution: iteration-1 scope narrowing prevents downstream success from overstating upstream evidence; iteration-2 caveat guidance for the paper reinforces the same pattern. **Resolved at wording level.**
- **Budget pressure pushes toward scope narrowing, not empirical rescue** — flagged in iteration 1; the loop actually followed this recommendation. **Confirmed pattern (not a problem).**
- **The project is more defensible on integrity than on evidential strength** — flagged in iteration 2. **Unresolved (paper-write concern).**
- **Acceptance odds hinge on narrative restraint** — flagged in iteration 2. **Unresolved (paper-write concern; carried to Section 8).**
- **C2 and C3 as INTEGRITY_ONLY force reliance on the caveat text** — flagged in iteration 1; the caveat is now in the loop's Open Items list. **Deferred to standalone `/auto-verify` runs (upgrade path recorded).**

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed**: 1 / 6
- **Claim-reentries consumed**: 1 / 2
- **Iteration `/run-experiment` calls**: runs_total = 0
- **Iteration GPU-hours**: gpu_hours_total = 0.00
- **Cumulative pipeline GPU-h** (experiment + verify + iteration): 7.41 / 10.00 (2.59 GPU-h unspent; loop returns pipeline-cap-safe)

### Per-iteration breakdown

| Iter | Type | Target claims | Produced claims | Runs | GPU-hours | Score after | Verdict after |
|---|---|---|---|---|---|---|---|
| 1 | ③ claim_reentry_lightweight | C1 | C1_v2 | 0 | 0.00 | 4 | not ready |
| 2 | ⓪ narrative_only | C1_v2 | — | 0 | 0.00 | 6 | almost |

Note: iteration 2's type-⓪ action does **not** consume the iteration counter per the loop contract — the "1/6" count reflects only the type-③ back-edge from iteration 1. Iteration 2 was a pure review that confirmed the STOP condition.

---

## Section 8 — Open Items for Human Reviewer

- **Still-FAIL claims**: none
- **Still-INCONCLUSIVE claims**: none
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: none (C1 resolved by scope narrowing to C1_v2)
- **INTEGRITY_ONLY claims (Stage 2 skipped — not stress-tested)**:
  - **C2** [stage2_skip_reason: max_verify_claims_cap; main_experiment_integrity: warn; warn_source: experiment (under-power)]: `/auto-verify C2 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run for this claim)
  - **C3** [stage2_skip_reason: max_verify_claims_cap; main_experiment_integrity: pass]: `/auto-verify C3 — resume: true`
- **Legacy deferred claims (empty in new runs)**: none
- **Recurring unresolved patterns** (from reviewer memory, iteration 2):
  - Residual possibility that probes exploit length/topic/style correlates rather than collusion semantics.
  - Weak evidence that group aggregation adds meaningfully beyond strongest single agent (C2 delta only +0.025).
  - Short-generation setting (max_new_tokens=60) may be distorting both text-judge and probe comparisons; empirically un-tested (budget-blocked).
  - Small dev/test splits (33/49) still make layer/effect-size conclusions unstable.
  - Paper-write concern: acceptance odds hinge on narrative restraint — iteration 2's type-⓪ recommendation for consistent, setting-qualified language throughout the write-up.
- **Claim-reentry refusals** (where reviewer requested ③ but sub-budget was exhausted): none.
