# Auto Iteration Final Report — Sparse Modular Circuit for Propositional-Logic Reasoning (Mistral-7B)

- **Generated**: 2026-07-15T03:30:00
- **Iterations consumed**: 0 / 6 (⓪-only iteration 1 confirmed a positive verdict — narrative-only actions do not consume iteration budget)
- **Claim-reentries consumed**: 0 / 2
- **Final reviewer score**: 7 / 10
- **Final canonical verdict**: ready
- **Termination reason**: positive_verdict (three-dimensional STOP fired iteration 1)
- **Cumulative cost**: runs_total=0, gpu_hours_total=0.0 (no back-edge experiments dispatched)
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md), [`REVIEW_STATE.json`](./REVIEW_STATE.json)

---

## Executive Summary

The autonomous review loop terminated on iteration 1 with the reviewer awarding **7/10, verdict "ready"**. The three-dimensional STOP rule fired immediately: (i) score ≥ TARGET_SCORE=6, (ii) verdict ∈ {ready, almost}, (iii) no claim in FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS state (C3 PASS via Gemma-2-9B model-swap variant; C1 and C2 INTEGRITY_ONLY, both stage2_skip_reason=`max_verify_claims_cap`, do not block READY per contract). The reviewer proposed **no ①/②/③ back-edge actions** — all identified weaknesses are paper-side caveats (type ⓪), documented in each claim's narrative record without touching scripts or dispatching new experiments. The publishable core (necessity-yes / sufficiency-no asymmetry, cross-family robust on Mistral-7B + Gemma-2-9B) is intact; the 20/158 minimality-sample caveat for C1 and the top-40/158 scope caveat for C2 are surfaced as paper-side disclosures. Two Open Items — `/auto-verify C1 — resume: true` and `/auto-verify C2 — resume: true` — would upgrade C1/C2 from INTEGRITY_ONLY to full swap-tested but are outside the iteration-loop scope.

### Claim Disposition Overview

| Original state | # Claims | Final status after iteration |
|---|---|---|
| PASS                     | 1 (C3)      | 1 PASS (held; robustness=1.00, cross-family Gemma-2-9B recurrence) |
| FAIL                     | 0           | — |
| INCONCLUSIVE             | 0           | — |
| ZERO_ELIGIBLE_VARIANTS   | 0           | — |
| INTEGRITY_ONLY           | 2 (C1, C2)  | 2 INTEGRITY_ONLY (no back-edge; upgrade suggestions recorded in Open Items) |
| DEFERRED (legacy)        | 0           | — |

---

## Section 1 — PASS Claims (brief audit)

### 1.1 `C3` — necessity + sufficiency

- **Statement (verbatim from task.md)**: The 158-component shortlist is BOTH necessary (path-patching clean → corrupt recovers ≥ 0.8 logit-diff, specificity_gap ≥ 0.6) AND sufficient (reinsertion + resample-ablation everywhere else recovers ≥ 0.8, specificity_gap ≥ 0.6, per-seed std < 0.1).
- **Main-experiment verdict**: partial (necessity PASS at LD=0.955, PD=0.905, specificity_gap=0.836; sufficiency FAIL at LD=0.113, PD=0.116, specificity_gap=0.113, per-seed std=0.036).
- **Verify verdict**: PASS (robustness = 1.00; 1/1 eligible variant passed; model-swap Gemma-2-9B reused M5 milestone result). The verify "PASS" here means the *conclusion* on C3 is robustly not-supported: BOTH models agree necessity ✓, sufficiency ✗.
- **Original robustness signal**: robustness=1.0, variants_passed=1/1 (model-swap-gemma2-9b, integrity=PASS).
- **Reviewer consistency check**: narrative + numeric coherent; the asymmetry is large (Mistral 0.955 vs 0.113; Gemma 1.018 vs 0.019), stable across 5 seeds (per-seed std 0.036), and the same pattern recurs on Gemma-2-9B. This is the study's most publishable result.
- **Touched in iterations**: [1] (⓪ narrative-only caveats added)
- **Final status**: PASS (held)
- **Notes for downstream (paper writeup)**:
  1. **KL scaling artifact** — recovery_KL=-5.618 in M2 is an artifact of using KL(clean||corrupt)=0.043 as denominator when patch_KL exceeds it. Logit-diff and prob-diff recoveries (0.955 / 0.905) are the definitive metrics and both pass. Already noted in `EXPERIMENT_RESULTS.md` Notes; surface in paper methods.
  2. **Verify variant reused M5 milestone** — one variant, reused Gemma-2-9B result, no fresh verify GPU run. Acceptable for consistency check but weak as robustness package; **do not oversell as broad cross-model validation** (it is one additional family/model, not a general law).
  3. **Interpretive gloss** — "the remaining 85% contains task-general infrastructure" is speculative. The intervention only shows insufficiency; it does not distinguish among (a) generic infrastructure, (b) many low-ranked task contributors, (c) nonlinear interactions, (d) resample-ablation distribution shift, (e) reinsertion-context mismatch. Paper should stay **interventionally modest**.
  4. **Anchor-cell scope** — the sufficiency + necessity numbers come primarily from `k3_chain2_natural`; M4.stab confirmed the *qualitative* pattern on 2 additional cells, but the paper's Discussion must not smuggle scope back to the general claim.

---

## Section 2 — FAIL Claims (full journey)

*None* — no claim was in FAIL state at loop entry.

---

## Section 3 — INCONCLUSIVE Claims (main-experiment-fix journey)

*None* — no claim was in INCONCLUSIVE state at loop entry.

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims (variant-fix journey)

*None* — no claim was in ZERO_ELIGIBLE_VARIANTS state at loop entry.

---

## Section 5 — Legacy DEFERRED Claims

*None (empty under current architecture).*

---

## Section 6 — Cross-Cutting Patterns

Patterns the reviewer flagged that touch multiple claims (deduped from REVIEWER_MEMORY.md iteration 1):

- **Scope-vs-headline gap** (touches C1, C2, C3):
  - All three claims are stated at the level of "the propositional-logic circuit", but the evidence in each case has a specific analysis scope:
    - C1 minimality — 20/158 component sample (17-in-8 sampled),
    - C2 modularity — top-40 of 158 components,
    - C3 numbers — anchor cell `k3_chain2_natural`.
  - Paper Discussion must not smuggle the scope back to the general claim.
  - Resolution status at termination: **unresolved** — surfaced as paper-side caveats only.

- **Interpretive language outruns intervention** (touches C3, with echoes in C1/C2):
  - Interpretive/mechanistic phrases ("distributed code", "task-general infrastructure", "modularity", "sparse circuit") consistently outrun what the interventions can distinguish.
  - Resolution status at termination: **unresolved** — recommended write-up posture is "stay interventionally modest".

- **Verification coverage is thin**:
  - Only C3 got Stage-2 verify. C1 and C2 remain INTEGRITY_ONLY.
  - Recommended upgrade (Open Items): `/auto-verify C1 — resume: true` and `/auto-verify C2 — resume: true`.
  - Resolution status at termination: **carried forward as Open Items** (outside iteration-loop scope by policy — no FAIL to fix, and INTEGRITY_ONLY does not consume iteration budget).

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed**: 0 / 6 (only a single ⓪-only pure-review iteration; per skill contract, ⓪ does not consume budget)
- **Claim-reentries consumed**: 0 / 2
- **Iteration `/run-experiment` calls**: runs_total = 0
- **Iteration GPU-hours**: gpu_hours_total = 0.0

### Per-iteration breakdown

| Iter | Type | Target claims | Produced claims | Runs | GPU-hours | Score after | Verdict after |
|---|---|---|---|---|---|---|---|
| 1 | ⓪ narrative-only | C1, C2, C3 (paper-side caveats added) | — | 0 | 0.0 | 7 | ready |

---

## Section 8 — Open Items for Human Reviewer

Items the loop could not close (each entry surfaced verbatim to orchestrator for `AUTO_PIPELINE_REPORT.md` / `CLAIMS_LEDGER.md` `open_items[]`):

- **Still-FAIL claims**: none
- **Still-INCONCLUSIVE claims**: none
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: none
- **INTEGRITY_ONLY claims (Stage 2 skipped — not stress-tested)**:
  - **C1** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C1 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run). [main-experiment integrity: warn; warn_source: experiment — undisclosed minimality sampling 20/158 components]
  - **C2** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C2 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run). [main-experiment integrity: warn; warn_source: experiment — shortlist scope reduction top-40 of 158 components for M4/M4.stab]
- **Legacy deferred claims**: none
- **Recurring unresolved patterns**:
  - Scope-vs-headline gap for C1, C2, C3 — surface as paper-side caveats.
  - Interpretive language outruns intervention (esp. C3's "task-general infrastructure" gloss) — recommend interventionally-modest write-up.
- **Claim-reentry refusals**: none (no ③ requested this iteration)

---

## Paper-side caveats (must appear in the write-up)

Consolidated from iteration 1's ⓪ narrative-only actions:

1. **C1 minimality** — the 0.14% average single-removal drop is measured on a 20/158 component sample, not exhaustively. Surface prominently in methods.
2. **C1 sparsity** — 0.150 hits the preset hard cap; report as threshold-satisfying, not as a discovered natural minimum.
3. **C2 scope** — the modularity analysis used top-40 of the 158-component shortlist. Phrase the negative conclusion as "no strong modular decomposition was found under the tested shortlist-and-top-40 analysis", not "the model lacks modular decomposition".
4. **C3 KL scaling artifact** — recovery_KL=-5.618 in M2 is a denominator-scaling artifact of a tight clean||corrupt baseline (0.043); logit-diff and prob-diff recoveries are the definitive metrics.
5. **C3 cross-family robustness is one-model** — reused M5 milestone, no fresh verify GPU run. State as "one additional family/model" not "broad cross-model validation".
6. **C3 interpretive framing** — "the remaining 85% contains task-general infrastructure" is speculative. Present the finding as "necessary but not sufficient under reinsertion + resample-ablation" and enumerate alternative explanations (low-ranked task signal, nonlinear interactions, resample-ablation distribution shift, reinsertion-context mismatch).
7. **Anchor-cell scope for C3 numbers** — the sufficiency + necessity numbers come primarily from `k3_chain2_natural`; qualitative pattern confirmed on 2 additional cells via M4.stab but the numbers are anchor-cell-specific.
