# Auto Iteration Final Report — Emotional Framing in Prompts as a Weak, Input-Dependent Signal

- **Generated**: 2026-07-14
- **Iterations consumed**: 0 / 6 (all reviewer-recommended fixes were type-0 narrative-only; no back-edge action of type 1/2/3 dispatched)
- **Claim-reentries consumed**: 0 / 2
- **Final reviewer score**: 6 / 10
- **Final canonical verdict**: almost
- **Termination reason**: iterations_exhausted (interpreted: reviewer explicitly recommended "stop" after iter-3 convergence check; no productive back-edge action possible within remaining ~0.79 GPU-h budget)
- **Cumulative cost**: runs_total = 0, gpu_hours_total = 0.0 (this loop consumed zero GPU-hours — all fixes were narrative + one CPU-only post-hoc analysis)
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md)

---

## Executive Summary

Over three reviewer cycles (2 substantive + 1 convergence check), the loop materially strengthened the paper's evidence-to-claim alignment through five type-0 narrative fixes and one CPU-only post-hoc analysis, moving the reviewer's score from 4/10 (initial "almost") to 6/10 (positive "almost") at TARGET_SCORE, with zero GPU spend. The five INTEGRITY_ONLY claims (C1, C2, C3a, C3b) — Stage-2 skipped by the max_verify_claims_cap policy — were each addressed with the reviewer-recommended paper-narrative treatment (partial-support surfacing for C1, failed-hypothesis demotion for C2, conservative rephrasing for C3a, threshold-level CI surfacing for C3b). The PASS claim CM was bifurcated in narrative into CM-location [supported] and CM-causal [not-supported at tested scale] per the reviewer's advice. C4 (INCONCLUSIVE due to M7 being descoped for budget in the main experiment) was demoted at the paper level to "untested, deferred, not falsified" but remains INCONCLUSIVE in the strict verify-report ledger — a full M7 execution requires ~2.17 GPU-h, well over the ~0.79 GPU-h remaining after main experiment + verify. The reviewer explicitly recommended termination at iter-3, noting no productive within-budget action remains and warning against reviewer-bait polish (a partial 100-item heuristic demo or CM specificity controls would not change the score and would risk credibility).

### Claim Disposition Overview

| Original state | # Claims | Final status after iteration |
|---|---|---|
| PASS                     | 1 (CM)               | 1 PASS (held; narrative bifurcated) |
| FAIL                     | 0                    | — |
| INCONCLUSIVE             | 1 (C4)               | 1 still INCONCLUSIVE (budget-blocked; ⓪ narrative demoted at paper level) |
| ZERO_ELIGIBLE_VARIANTS   | 0                    | — |
| INTEGRITY_ONLY           | 4 (C1, C2, C3a, C3b) | 4 held as INTEGRITY_ONLY (Open Items); each got a ⓪ narrative refinement |

---

## Section 1 — PASS Claims (brief audit)

### 1.1 `CM` — Residual direction carries emotion identity (Location) AND causally modulates GSM8K accuracy (Causal Intervention)

- **Original robustness signal**: robustness=1.00 (1 pass / 1 eligible), variant `model-swap-qwen3-4b` (same family, Qwen3-4B replacing Qwen3-14B); integrity=PASS.
- **Reviewer consistency check**: confirmed at iter-1 and iter-2. Location arm: probe accuracy 1.00 at layers 4-36 vs length-shuffled null 0.28 (+71 pp gap); onset at L4; robust across the Qwen3 family. Causal arm: steering dose-response L4 range 2 pp / L8 range 4 pp, at or below the 5.24 pp noise floor; no monotone α dependence; identity-patch sanity only (cross-emotion patching descoped); specificity controls (filler, off-target, off-layer) descoped.
- **Touched in iterations**: [1, 2, 3]
- **Final status**: PASS (held); narrative-bifurcated into CM-location [supported] and CM-causal [not-supported at tested scale].
- **Notes for downstream**:
  - **Variant length-null caveat**: the swap variant used only 7 conditions (vs 24 in main M5); the length-shuffled null was 0.69 (not the ≤0.2 originally designed for the 24-condition setup). Probe-null gap of 31 pp remains positive, so the Location finding holds, but paper should acknowledge this variant-scale caveat.
  - **Paper wording constraint**: do NOT claim causal mechanism beyond "decodable representations exist; causal leverage under tested interventions is weak." No unified "causal mechanism" language.
  - **Optional Priority-2 fix (not taken)**: reviewer explicitly rejected spending 0.3-0.5 GPU-h on filler/off-target/off-layer specificity controls — would strengthen an already-defensible claim without unblocking C4; risks reviewer-bait polish.

---

## Section 2 — FAIL Claims (full journey)

*None. No claim entered the loop with state FAIL.*

---

## Section 3 — INCONCLUSIVE Claims (main-experiment-fix journey)

### 3.1 `C4` — An adaptive per-query policy (EmotionRL) yields more reliable accuracy gains than any fixed emotional prefix or neutral baseline

**Original INCONCLUSIVE reason** (from `verify/C4_adaptive_policy_beats_fixed/ROBUSTNESS.md`):
- Main-experiment integrity FAIL — M7 (EmotionRL adaptive prompt selection) was **entirely descoped** during the main experiment stage: no reward table, no policy training, no held-out evaluation was run. All M7 scripts are dead code relative to actual execution. "Experiment not carried out" = FAIL integrity → INCONCLUSIVE verdict.
- Budget context: M2/M2b/M3/M5/M6 consumed 9.08 GPU-h of the 10-h task cap. M7a alone was estimated at 2.17 GPU-h — no room to run it within the original run's budget, and only ~0.79 GPU-h remained for this iteration loop.

**Experiment plan & script modifications**

| Iter | Component | Before | After |
|---|---|---|---|
| — | none | — | — (a full M7 execution would require ≥2.17 GPU-h — over budget. Only ⓪ narrative demotion was applied.) |

**Re-experiment outcome**

| Iter | Path | New runs | Result |
|---|---|---|---|
| 1 | ⓪ narrative demotion only | 0 | C4 removed from paper's core contributions; ledger status remains INCONCLUSIVE |
| 2 | ⓪ confirmed — reviewer accepts option (C) treatment (grade as 5-claim paper) | 0 | Reviewer confirms paper-level treatment is correct; ledger status remains INCONCLUSIVE |
| 3 | ⓪ convergence check — no productive action available within budget | 0 | Reviewer explicitly recommends "stop" |

**Final status**: **still INCONCLUSIVE — flagged in Open Items**

**Paper-level treatment (from ⓪ demotion, iter-1)**: The paper text explicitly states that (a) the adaptive-policy hypothesis was NOT tested; (b) M7 was descoped for budget reasons; (c) the paper makes NO empirical claim about adaptive policies outperforming neutral or best-fixed prompts; (d) this is deferred to future standalone verification. No "inconclusive but suggestive" language; no partial 100-item heuristic demo (reviewer explicitly rejected this as reviewer-bait). Reviewer's option (C) recommendation: treat the submission as a 5-claim paper (C1, C2, C3a, C3b, CM) with C4 as future-work note only.

**Reviewer memory thread**:
- **Iter-1**: "C4 is the only readiness blocker; recommend pure narrative demotion rather than weak partial experiment." Resolved at paper level.
- **Iter-2**: "C4 remains inconclusive in strict ledger terms → still unresolved unless removed from submission claim set." Reviewer suggests the 5-claim framing.
- **Iter-3**: "C4 cannot be resolved within the remaining budget."

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims (variant-fix journey)

*None. No claim entered the loop with state ZERO_ELIGIBLE_VARIANTS.*

---

## Section 4b — INTEGRITY_ONLY Claims (no-action-with-upgrade-suggestion)

*Per the skill contract, INTEGRITY_ONLY claims incur no back-edge action inside the loop — the recommended upgrade is a standalone /auto-verify run outside the loop. Each is listed here with any ⓪ narrative refinement that landed inside the loop.*

### 4b.1 `C1` — Static emotional prefixes cause only small, input-dependent shifts

- **stage2_skip_reason**: max_verify_claims_cap
- **Main-experiment integrity**: WARN (warn_source: experiment; sign-consistency predicate was not computed at experiment time)
- **Upgrade suggestion**: `/auto-verify C1 -- resume: true` (Phase 2 audit reused via RESUME; only Stages 2-3 run)
- **⓪ narrative refinement applied in iter-1**:
  - Sign-consistency band computed post-hoc via `scripts/c1_sign_consistency.py` on M2 per-item data (CPU-only, no GPU).
  - Result: 16/24 conditions in-band [0.4, 0.6]; 8/24 out-of-band. The 8 out-of-band cells align exactly with the 6 noise-floor crossers + 2 borderline cells.
  - Written to `reports/C1_sign_consistency.json`.
  - EXPERIMENT_RESULTS.md M2b section now includes the full sign-consistency table + revised C1 verdict: "partial support at emotion-mean level; individual-prefix directional signals present."
  - **The main-experiment WARN reason (sign-consistency skipped) is now resolved post-hoc**, though the WARN status is not retroactively updated in `INTEGRITY_AUDIT.md`.

### 4b.2 `C2` — Task-family spread ordering (social ≥ 2× math AND social > factual > math)

- **stage2_skip_reason**: max_verify_claims_cap
- **Main-experiment integrity**: WARN (warn_source: experiment; eval-mode confound CoT vs MCQ-LL)
- **Upgrade suggestion**: `/auto-verify C2 -- resume: true`
- **⓪ narrative refinement applied in iter-1**:
  - CLAIMS_LEDGER.md C2 section explicitly labels C2 as a **failed hypothesis**: observed ordering REVERSED (GSM8K 18.6 pp >> SocialIQA 3.0 pp > MedQA 2.8 pp); only 1/3 pairwise orderings hold; eval-mode confound foregrounded as a primary limitation (not buried).

### 4b.3 `C3a` — No single emotion is argmax across all three task families

- **stage2_skip_reason**: max_verify_claims_cap
- **Main-experiment integrity**: PASS (no WARN)
- **Upgrade suggestion**: `/auto-verify C3a -- resume: true`
- **⓪ narrative refinement applied in iter-1**: Conservative rephrasing to "the best-performing emotion is task-dependent and not invariant across domains" (fear on math/factual; surprise on social). Do NOT oversell as broad task-specific specialization.

### 4b.4 `C3b` — No monotone intensity gradient (≥3/6 emotions fail on GSM8K)

- **stage2_skip_reason**: max_verify_claims_cap
- **Main-experiment integrity**: WARN (warn_source: experiment; passes exactly at 3/6 threshold with no per-emotion CI bounds surfaced)
- **Upgrade suggestion**: `/auto-verify C3b -- resume: true`
- **⓪ narrative refinement applied in iter-1**:
  - Per-emotion Δ(int2−int1) 95% CIs surfaced from `reports/M4_c3_analysis.json` (already computed, previously not in main text).
  - fear [−0.004, +0.046], anger [−0.019, +0.027], surprise [−0.062, −0.008] fail monotonicity; sadness [+0.001, +0.053] is monotone but marginal.
  - Labeled as **threshold-level support** (exactly 3/6 meets ≥3 gate).
  - **The main-experiment WARN reason (per-emotion CI bounds not surfaced) is now resolved narratively**.

---

## Section 5 — Legacy DEFERRED Claims (empty)

*New verify runs never populate this section. Not applicable.*

---

## Section 6 — Cross-Cutting Patterns

- **Overclaim → recast**: The original claim set was materially overclaimed relative to the evidence. The reviewer's dominant iter-1 pattern was recasting the paper as a **negative/qualified-result paper on emotional-prefix effects with mechanism-encoding-only support**. This recast, applied via ⓪ narrative fixes across all six claims, moved the score from 4/10 to 6/10 with zero GPU spend. **Pattern resolved at paper level; not fully resolved for C4-in-ledger.**
- **Budget-blocked pruning**: When a compute fix is over-budget (C4/M7 at 2.17 GPU-h vs 0.79 available), the reviewer consistently prefers ⓪ narrative honesty over any partial compute pass. Rationale: partial compute pass risks looking like reviewer-bait. **Pattern honored throughout the loop.**
- **Bifurcation over merging**: The reviewer preferred splitting CM into location/causal in narrative rather than defending the joint claim. Similarly, C1 was refined into "emotion-mean level" vs "individual-prefix level" support. **Pattern: prefer honest sub-claims to defended super-claims.**
- **Front-matter risk**: New iter-2 concern. Even after ledger repair, title/abstract/introduction may still oversell — must be visibly modest. **Pattern: not yet resolved on disk; carries forward to paper-writing stage.**
- **Eval-mode confound (C2)**: CoT vs MCQ-LL cross-task comparison — foregrounded in ledger but remains a substantive limitation. **Pattern: acknowledged, not eliminated.**

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed**: 0 / 6
- **Claim-reentries consumed**: 0 / 2
- **Iteration `/run-experiment` calls**: runs_total = 0
- **Iteration GPU-hours**: gpu_hours_total = 0.0
- **CPU-only post-hoc analyses**: 1 (`scripts/c1_sign_consistency.py`)

### Per-iteration breakdown

| Iter | Type | Target claims | Produced claims | Runs | GPU-hours | Score after | Verdict after |
|---|---|---|---|---|---|---|---|
| 1 | ⓪ narrative-only (all 6 claims) + 1 CPU post-hoc | C1, C2, C3a, C3b, C4, CM | — | 0 | 0.0 | 4 | almost (initial baseline) |
| 2 | ⓪ (no on-disk change — re-scoring pass) | — | — | 0 | 0.0 | 6 | almost (positive) |
| 3 | ⓪ (convergence check) | — | — | 0 | 0.0 | 6 | almost (stable) |

**Notes**:
- All three iterations are logged in `AUTO_REVIEW.md` for audit purposes even though none consumed the iteration budget (per the skill contract, type-⓪ narrative-only iterations do not increment `iterations_consumed`).
- Iter-1 is the substantive one: 5 claims received ⓪ narrative refinements + 1 CPU-only post-hoc analysis on C1.
- Iter-2 is the reviewer's re-score of iter-1 changes.
- Iter-3 is the convergence check — reviewer explicitly says "stop."

---

## Section 8 — Open Items for Human Reviewer

- **Still-INCONCLUSIVE claims**:
  - **C4**: main-experiment integrity broken — M7 (EmotionRL adaptive prompt selection) was entirely descoped in the main experiment stage due to budget exhaustion (9.08/10 GPU-h consumed by M2/M2b/M3/M5/M6). Full M7 re-run requires ~2.17 GPU-h; only ~0.79 GPU-h remained for this iteration loop. **Recommended next steps**:
    1. Paper: keep C4 removed from title/abstract/introduction/contributions/conclusion (⓪ demotion already applied in ledger); frame as deferred future work.
    2. Standalone re-verification: run M7 (M7a reward table → M7b SFT → M7c REINFORCE → M7d held-out eval) as a separate compute allocation, then `/auto-verify C4 -- resume: false`.
    3. Missing model backbones: Llama-3.2-1B and bert-base-cased are absent from $MODEL_DIR; download required or substitute Llama-3.2-3B-Instruct (present).

- **Still-FAIL claims**: none.
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: none.

- **INTEGRITY_ONLY claims (Stage 2 skipped — not stress-tested)**:
  - **C1** [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: warn; warn_source: experiment]: `/auto-verify C1 -- resume: true`. Note: the WARN reason (sign-consistency skipped) is now resolved post-hoc via `reports/C1_sign_consistency.json`.
  - **C2** [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: warn; warn_source: experiment]: `/auto-verify C2 -- resume: true`. Note: WARN reason (eval-mode confound CoT vs MCQ-LL) is foregrounded in paper but not eliminated.
  - **C3a** [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: pass]: `/auto-verify C3a -- resume: true`.
  - **C3b** [stage2_skip_reason: max_verify_claims_cap; main-experiment integrity: warn; warn_source: experiment]: `/auto-verify C3b -- resume: true`. Note: WARN reason (per-emotion CIs not surfaced) is now resolved via M4 surfacing.

- **Legacy deferred claims**: empty (this is a new-architecture verify run).

- **Recurring unresolved patterns**:
  - Front-matter oversell risk (iter-2 new suspicion): title/abstract/introduction must be visibly modest and match the repaired ledger text. Not on-disk-visible until paper-writing stage.
  - Novelty/impact ceiling: even after all repairs, the paper may still read as "we tested a provocative framing and mostly found it does not robustly hold." Publishable only if analysis quality + honesty are unusually strong.
  - CM causal specificity (filler / off-target / off-layer descoped): acknowledged in ledger but not eliminated. Optional 0.3-0.5 GPU-h fix declined by reviewer as reviewer-bait.
  - CM variant length-null 0.69 (7 conditions in variant vs 24 designed for): note but does not overturn Location finding.

- **Claim-reentry refusals** (where reviewer requested ③ but sub-budget was exhausted): none. No ③ was ever requested — all reviewer recommendations were ⓪.

- **Reviewer's summary recommendation**:
  - **Paper-level narrative**: defensible "almost-ready" qualified/negative-result submission.
  - **Auto-verify / strict-ledger outcome**: NOT READY due to unresolved C4 INCONCLUSIVE.
  - **Recommended framing**: 5-claim paper (C1, C2, C3a, C3b, CM) with C4 as future-work note.
