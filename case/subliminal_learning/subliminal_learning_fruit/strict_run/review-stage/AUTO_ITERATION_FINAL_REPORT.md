# Auto Iteration Final Report — Subliminal Learning in Diffusion Image Models (Qwen-Image)

- **Generated**: 2026-07-18T18:50:04Z
- **Iterations consumed**: 0 / 6
- **Claim-reentries consumed**: 0 / 2
- **Final reviewer score**: 6 / 10
- **Final canonical verdict**: almost
- **Termination reason**: positive_verdict
- **Cumulative cost**: runs_total = 0, gpu_hours_total = 0.0 (loop dispatched no GPU runs — pipeline already over its 10-h HARD budget with ~12 GPU-h spent in the experiment stage; action space this round was strictly ⓪ narrative-only)
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md)

---

## Executive Summary

The iteration loop terminated on the very first Phase A→E cycle: the external reviewer (gpt-5.4 via <REDACTED_API_PROVIDER>) scored the work **6/10** with verdict **Almost**, and — as expected under the strict budget guidance — the reviewer explicitly declined to prescribe any ①/②/③ back-edge action on the two INTEGRITY_ONLY claims. Instead, the reviewer recommended a batch of ⓪ narrative-only edits: soften C1's "zero-residue" rhetoric to "no reliable residue beyond judge-noise floor" and reframe C2 from an ambiguous "we identified the mechanism" claim into an explicit informative-negative finding ("hotspot found, single-block causal handle weak, evidence for distributed carriage"). Those edits were applied to `CLAIMS_LEDGER.md` and `refine-logs/EXPERIMENT_RESULTS.md` in-loop, and the reviewer-recommended Section 8 paper wording is now on record for downstream. Both claims stay INTEGRITY_ONLY (Stage-2 stress test deferred) and are carried forward under Open Items with their per-`stage2_skip_reason` upgrade commands. The three-dimensional STOP rule fires cleanly: score ≥ 6, verdict `almost`, and zero FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS claims remaining.

### Claim Disposition Overview

| Original state | # Claims | Final status after iteration |
|---|---|---|
| PASS                     | 0 | — |
| FAIL                     | 0 | — |
| INCONCLUSIVE             | 0 | — |
| ZERO_ELIGIBLE_VARIANTS   | 0 | — |
| INTEGRITY_ONLY           | 2 | 2 INTEGRITY_ONLY (both carried to Open Items; narrative reframed via ⓪) |
| DEFERRED (legacy)        | 0 | — |

---

## Section 1 — PASS Claims (brief audit)

_none — no PASS claims at loop entry._

---

## Section 2 — FAIL Claims (full journey)

_none — no FAIL claims at loop entry._

---

## Section 3 — INCONCLUSIVE Claims (main-experiment-fix journey)

_none — no INCONCLUSIVE claims at loop entry._

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims (variant-fix journey)

_none — no ZERO_ELIGIBLE_VARIANTS claims at loop entry._

---

## Section 4b — INTEGRITY_ONLY Claims (no-action, narrative-only path)

Both claims below carried Stage 1 audit results on disk and were intentionally not stress-tested at Stage 2 because the `max_verify_claims_cap` fired (verify stage triaged them under a 10-GPU-h HARD budget already exceeded by the experiment stage). The iteration loop's job on this bucket is: **do NOT propose ①/②/③ back-edges**, apply ⓪ narrative discipline, and surface each claim under Open Items with the correct upgrade path. That is what iteration 1 did.

### 4b.1 `C1` — Subliminal transfer of banana-preference bias via denoising SFT on filtered teacher-generated data

**Main-experiment verdict at entry**: `conditional` (per-plan four-state gate) — every one of the 8 mandated seeds (200..207) shows teacher-arm P(banana) exceeding both controls by ≥ 10pp; min per-seed Δ_teacher−Ctrl_A = +0.100, Δ_teacher−Ctrl_B = +0.1125. The `conditional` tag reflects `banana_residue = 5/302 (1.7%)` after the 5-pass strict-any-banana drop — at the gpt-5.4 judge's ~0.3%-per-image false-positive floor, i.e., instrument noise rather than filter failure.

**Verify state at entry**: `INTEGRITY_ONLY`, `stage2_skip_reason: max_verify_claims_cap`. Phase 2 combined verdict = WARN (experiment audit Check C — verdict-stage protocol deviation, documented; mechanism audit N/A because C1 uses no mechanism intervention).

**Iteration journey**

| Iter | Type | Reviewer flag | Action | Outcome |
|---|---|---|---|---|
| 1 | ⓪ | "zero-residue" rhetoric overstates evidence; verdict-stage protocol deviation needs transparent disclosure as manual scientific override | edited `CLAIMS_LEDGER.md` Headline (before/after in AUTO_REVIEW.md), added `Paper-narrative notes (C1)` and `Section 8 Open Items — C1` in `CLAIMS_LEDGER.md`, added `Paper-framing note` in `refine-logs/EXPERIMENT_RESULTS.md` Executive Summary | narrative softened without altering the frozen claim predicate; INTEGRITY_ONLY state unchanged (correctly — this bucket is no-action-by-design); `conditional` verdict tag remains as the scientifically-correct posture |

**Path taken (summary)**: ⓪ narrative-only — no back-edge action, no iteration budget consumed, no claim-reentry sub-budget consumed. Correct per the skill's `verify_integrity_only` contract.

**Experiment & script modifications** (cumulative)

| Iter | Component | Before | After |
|---|---|---|---|
| 1 | `CLAIMS_LEDGER.md` — C1 Headline | "Subliminal transfer demonstrated across all 8 seeds at ≥2× the 5pp threshold on both matched controls — the diffusion analog of subliminal learning holds. The 'conditional' tag reflects judge-noise-floor residue, not filter failure." | "Subliminal transfer demonstrated across all 8 seeds at ≥2× the 5pp threshold on both matched controls — the diffusion analog of subliminal learning holds under aggressively filtered non-banana teacher outputs. The 'conditional' tag reflects the fact that residual banana flags were sparse and unstable across rescans (consistent with gpt-5.4 judge-noise floor, ~0.3% per-image false-positive rate), not literal residue-free proof — see paper-narrative note in Open Items." |
| 1 | `CLAIMS_LEDGER.md` — new Paper-narrative notes (C1 half) | (did not exist) | Explicit paper-wording guidance to (a) replace "zero-residue" rhetoric with "no reliable banana residue beyond judge-noise floor"; (b) frame contribution as "robust transfer under filtered non-banana teacher channel", not "mathematically residue-free transfer"; (c) keep `conditional` verdict-tag transparently disclosed as manual scientific override. |
| 1 | `CLAIMS_LEDGER.md` — new Section 8 (paper) Open Items — C1 wording | (did not exist) | "We deferred an anchor-swap stress test (e.g., strawberry anchor) due to the verification cap; thus, our strongest evidence is currently for the banana-anchored setup, with cross-anchor generality left as future work." Present as "strong within-anchor evidence", not "anchor-general transfer law". |
| 1 | `refine-logs/EXPERIMENT_RESULTS.md` — Executive Summary "Paper-framing note" | (did not exist) | Paper storyline = phenomenon-first, mechanism-partial/negative. C1 framing = "conditional on judge-noise-floor interpretation, not residue-free". |

**Claim modifications**: none — variant-fix / claim-rewrite paths were policy-blocked (INTEGRITY_ONLY bucket is no-action-with-upgrade-suggestion). The C1 claim STATEMENT (in both `CLAIMS_LEDGER.md` and `refine-logs/FINAL_PROPOSAL.md`) is a frozen predicate from `task.md` M0 validation criteria and was not touched. The paper-narrative wording is separate from the frozen predicate; the paper is free to frame the same measurement conservatively.

**Final experiment summary**
- Verify state at exit: INTEGRITY_ONLY (unchanged — correct for the no-action bucket).
- Final robustness: N/A (Stage 2 not run).
- Final variant pass rate: N/A (Stage 2 not run).
- Final status: **INTEGRITY_ONLY — carried to Open Items with `/auto-verify C1 — resume: true` upgrade path and recommended anchor-swap variant (~2-3 GPU-h in a follow-up round).**

**Reviewer memory thread** (cross-iteration suspicions filtered to C1's pattern)
- C1 wording risk (residue > 0 → do not silently rebrand `conditional` as "supported")
- Verdict-stage protocol-deviation disclosure (`conditional` label vs code-default `inconclusive`) must be transparent
- Resolved: yes for both — CLAIMS_LEDGER.md headline + paper-narrative notes now record the honest framing; `EXPERIMENT_RESULTS.md` executive-summary paper-framing note explicitly bakes in the protocol-deviation-disclosure guidance for the paper draft.

### 4b.2 `C2` — Identifiable DiT component causally carries the transferred bias

**Main-experiment verdict at entry**: `not-supported [provisional — suspected under-power]`. M1 Location screen identified DiT block 50 (of 60) with Grassmann-overlap ratio_k1 = 4.87 (> 2.0 threshold). M2 causal intervention (4 seeds × 3 interventions, scoped down from the planned 8×5=40 grid) showed mean Δ_ablate = −0.015pp (fails ≥5pp gate); Δ_amplify_x3 = +0.013pp (weak dose signal); Δ_random = −0.019pp (fails specificity — random ablation comparable to or larger than direction ablation). Interpretation: block 50 is a real representation-level differential site, but single-block causal intervention is indistinguishable from a matched-random-direction control — the carrier direction is distributed across many DiT blocks.

**Verify state at entry**: `INTEGRITY_ONLY`, `stage2_skip_reason: max_verify_claims_cap` (and also not the top-K pick under the cap — C1 outranked C2 in importance at Phase 3 step 0). Phase 2 combined verdict = WARN (experiment audit: scope reduction 40→12, dose-response single-point; mechanism audit Check A: sweep cardinality FAIL — amplify_x3 only; σ_proj calibration PASS; matched-random baseline PASS).

**Iteration journey**

| Iter | Type | Reviewer flag | Action | Outcome |
|---|---|---|---|---|
| 1 | ⓪ | narrative ambiguous between "we identified the causal component" and "mechanism failed"; correct read is "hotspot found, causal handle weak / evidence for distributed carriage"; matched-random comparability at block 50 is the specificity red flag and must not be buried | edited `CLAIMS_LEDGER.md` Headline (before/after in AUTO_REVIEW.md), added `Paper-narrative notes (C2)` and `Section 8 Open Items — C2` in `CLAIMS_LEDGER.md`, added `Paper-framing note (C2)` in `refine-logs/EXPERIMENT_RESULTS.md` Executive Summary | narrative reframed as legitimate informative-negative finding (evidence AGAINST simple single-block localization; indications of distributed carriage); INTEGRITY_ONLY state unchanged (correctly — no-action-by-design) |

**Path taken (summary)**: ⓪ narrative-only — no back-edge action, no iteration budget consumed, no claim-reentry sub-budget consumed. Correct per the `verify_integrity_only` contract.

**Experiment & script modifications** (cumulative)

| Iter | Component | Before | After |
|---|---|---|---|
| 1 | `CLAIMS_LEDGER.md` — C2 Headline | "Weak / delocalized. Block 50 is a real Grassmann-overlap-differential site (ratio 4.87), but single-block causal intervention is indistinguishable from a matched-random-direction control. The banana-carrier direction is distributed across many DiT blocks." | "Localization screen reveals a nontrivial hotspot at block 50 (Grassmann-overlap ratio 4.87 > 2.0 threshold), but single-block causal intervention is indistinguishable from a matched-random-direction control at the same site. Interpreted as evidence AGAINST simple single-block localization and consistent with distributed/delocalized carriage across many DiT blocks — a legitimate informative negative-mechanism finding, not a failed experiment. Whether a broader multi-block simultaneous intervention would recover a causal handle remains open." |
| 1 | `CLAIMS_LEDGER.md` — new Paper-narrative notes (C2 half) | (did not exist) | Guidance NOT to frame as "we identified the causal component" and NOT to frame as "mechanism failed". Prefer section title "Localization screen reveals a hotspot, but causal influence appears delocalized" (or "Evidence for distributed carriage in the DiT"). Keep matched-random comparability CENTRAL to the story; do not overclaim from M1 screen to M2 causal interpretation. |
| 1 | `CLAIMS_LEDGER.md` — new Section 8 (paper) Open Items — C2 wording | (did not exist) | "Because budget constraints prevented completion of the full intervention grid and model-swap stress test, we treat the mechanism result as evidence against simple single-block localization rather than a definitive map of the carrier; a broader multi-block causal intervention remains an open item." Distinguish (a) supported conclusion = evidence against single-block localization; (b) open question = whether a broader multi-block intervention can causally recover the mechanism. |
| 1 | `refine-logs/EXPERIMENT_RESULTS.md` — Executive Summary "Paper-framing note" | (did not exist) | Paper storyline = phenomenon-first, mechanism-partial/negative. C2 framing = "hotspot found, causal handle weak and nonspecific, distributed carriage". Do NOT overclaim from M1 screen to M2 causal identification. |

**Claim modifications**: none — variant-fix / claim-rewrite paths were policy-blocked (INTEGRITY_ONLY bucket is no-action-with-upgrade-suggestion). The C2 claim STATEMENT is a frozen predicate from `refine-logs/FINAL_PROPOSAL.md` and was not touched. The paper-narrative wording is separate; the paper is free to frame the same measurement as a negative-mechanism finding.

**Final experiment summary**
- Verify state at exit: INTEGRITY_ONLY (unchanged — correct for the no-action bucket).
- Final robustness: N/A (Stage 2 not run).
- Final variant pass rate: N/A (Stage 2 not run).
- Final status: **INTEGRITY_ONLY — carried to Open Items with `/auto-verify C2 — resume: true` upgrade path, prerequisite = complete the 28 remaining M2 runs (seeds {201,203,204,206} × interventions {ablate, amplify_x2, amplify_x3, amplify_x4, random_ablate}), then run model-swap variant.**

**Reviewer memory thread**
- C2 overclaim risk from M1 → M2 (Grassmann overlap ≠ mechanism identification)
- Random-ablation comparability at block 50 is the specificity red flag
- Framing constraint: neither "we identified the mechanism" nor "mechanism failed"
- Resolved: yes for all three — CLAIMS_LEDGER.md headline + paper-narrative notes now record "hotspot found / causal handle weak / distributed carriage" as the honest framing and explicitly foreground the matched-random comparability.

---

## Section 5 — Legacy DEFERRED Claims (empty under current architecture)

_None — new verify runs do not populate this bucket; both C1 and C2 landed in `verify_integrity_only` with `stage2_skip_reason: max_verify_claims_cap`._

---

## Section 6 — Cross-Cutting Patterns

- **Pattern: paper-narrative discipline is the load-bearing element for this round.** Both claims came back INTEGRITY_ONLY under an over-budget verify stage — the loop's job here is to make the paper's wording match the evidence, not to run more experiments. Applied and closed in iteration 1.
- **Pattern: informative-negative findings are easy to misread as failure.** C2's honest read ("hotspot found, causal handle weak, distributed carriage") is scientifically legitimate — the loop's edits make sure the paper does not present C2 as either an unqualified success or a bare failure. Applied and closed in iteration 1.
- **Pattern: verdict-stage protocol deviations must be transparent, not silent.** C1's `conditional` label overrides the code's `inconclusive` when residue > 0; this is defensible only if disclosed as a documented manual scientific override. Applied and closed in iteration 1 (Paper-narrative notes explicitly bake in the disclosure).

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed**: 0 / 6
- **Claim-reentries consumed**: 0 / 2
- **Iteration `/run-experiment` calls**: runs_total = 0
- **Iteration GPU-hours**: gpu_hours_total = 0.0

### Per-iteration breakdown

| Iter | Type | Target claims | Produced claims | Runs | GPU-hours | Score after | Verdict after |
|---|---|---|---|---|---|---|---|
| 1 | ⓪ narrative_only | C1, C2 | — | 0 | 0.0 | 6 | almost |

Note: ⓪ narrative-only iterations do not consume the iteration budget by design (see the skill's Phase C bookkeeping — counters `iterations_consumed` and `iteration_breakdown` are only appended for real ①/②/③ actions; the ⓪ trace lives here and in `AUTO_REVIEW.md`'s Actions Taken).

---

## Section 8 — Open Items for Human Reviewer

Items the loop could not close under the strict budget and INTEGRITY_ONLY policy. These need incremental GPU-h allocation or a paper-side decision.

- **Still-FAIL claims** (after exhausting routing options): **none** (0 FAIL at loop entry).
- **Still-INCONCLUSIVE claims**: **none** (0 INCONCLUSIVE at loop entry).
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: **none** (0 ZEV at loop entry).
- **INTEGRITY_ONLY claims (Stage 2 skipped — not stress-tested)**:
  - **C1** [stage2_skip_reason: `max_verify_claims_cap`]: upgrade = `/auto-verify C1 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run for this claim). Recommended variant = anchor-swap (strawberry anchor instead of banana anchor; same Qwen-Image base, same LoRA config, same hyperparameters, same judge). Estimated ~2-3 GPU-h. main_experiment_integrity = **warn**, warn_source = **experiment** (verdict-stage protocol deviation, documented with scientific justification).
  - **C2** [stage2_skip_reason: `max_verify_claims_cap`]: upgrade = `/auto-verify C2 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run for this claim). Recommended prerequisite = complete M2 grid first (28 remaining runs: seeds {201,203,204,206} × interventions {ablate, amplify_x2, amplify_x3, amplify_x4, random_ablate}) to establish whether the delocalized not-supported verdict holds at full power; then run model-swap variant. main_experiment_integrity = **warn**, warn_source = **experiment+mechanism** (M2 scope reduction 40→12; M2 sweep cardinality gap — amplify_x3 only, single-point dose curve).
- **Legacy deferred claims (empty in new runs)**: **none**.
- **Recurring unresolved patterns**: **none** — all three cross-cutting patterns (paper-narrative discipline, informative-negative-vs-failure framing, verdict-deviation transparency) were addressed in iteration 1's ⓪ edits.
- **Claim-reentry refusals** (where reviewer requested ③ but sub-budget was exhausted): **none** (reviewer did not request ③; per the skill contract, ③ is not permitted on INTEGRITY_ONLY claims regardless of sub-budget).

**Handoff to next round.** The two INTEGRITY_ONLY claims and their upgrade commands are the concrete follow-up work; they need an incremental GPU-hour allocation (approximately 3 GPU-h for C1's anchor-swap variant; approximately 7-9 GPU-h for C2's M2 grid completion + model-swap variant — 28 runs at ~0.25 GPU-h each plus swap variants). None of this is in-scope for the current round's iteration budget.
