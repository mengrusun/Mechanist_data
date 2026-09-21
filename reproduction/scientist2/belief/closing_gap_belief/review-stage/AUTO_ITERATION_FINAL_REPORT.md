# Auto Iteration Final Report — Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence in Llama-3.1-8B-Instruct on TriviaQA

- **Generated**: 2026-07-14
- **Iterations consumed**: 0 / 6
- **Claim-reentries consumed**: 0 / 2
- **Final reviewer score**: 6.5 / 10
- **Final canonical verdict**: almost
- **Termination reason**: positive_verdict (three-dimensional STOP fired iteration 1)
- **Cumulative cost**: runs_total = 0, gpu_hours_total = 0.0
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md)

---

## Executive Summary

Iteration 1's external reviewer (gpt-5.4) scored the work **6.5/10 "almost"** and identified **no genuine methodology bug** that would warrant breaking the INTEGRITY_ONLY no-action-with-upgrade-suggestion contract for C1/C2/C3b/C3c. The three-dimensional STOP rule fired immediately: (a) score 6.5 ≥ TARGET_SCORE=6, (b) verdict "almost" ∈ {ready, almost}, and (c) no claim in FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS — only C3a in PASS plus four claims in INTEGRITY_ONLY (which never block STOP). Every reviewer criticism resolved either to (i) paper-writing scoping / framing (⓪ narrative-only, not iteration-actionable), (ii) an INTEGRITY_ONLY claim's canonical `/auto-verify <id> -- resume: true` upgrade path (out-of-scope for this budget by pre-registered cap), or (iii) accepting the honest C3c provisional null. Zero iterations were consumed; zero GPU-hours spent.

### Claim Disposition Overview

| Original state (from `/auto-verify`) | # Claims | Final status after iteration |
|---|---|---|
| PASS                     | 1 (C3a)                | 1 PASS (held) |
| FAIL                     | 0                      | — |
| INCONCLUSIVE             | 0                      | — |
| ZERO_ELIGIBLE_VARIANTS   | 0                      | — |
| INTEGRITY_ONLY           | 4 (C1, C2, C3b, C3c)   | 4 INTEGRITY_ONLY (no-action-by-contract; upgrade path recorded in Open Items) |
| DEFERRED (legacy)        | 0                      | — |

---

## Section 1 — PASS Claims (brief audit)

### 1.1 `C3a` — geometric near-orthogonality of `v_c` and `v_v` (LOAD-BEARING)
- **Original robustness signal**: `robustness = 1.00` on 1 eligible Qwen2.5-7B-Instruct swap variant (`verify/C3a_near_orthogonality/ROBUSTNESS.md`).
- **Reviewer consistency check**: Main-experiment (`|cos| = 0.015 [0.001, 0.034]` at L*=31; neighborhood mean L*±2 = 0.025; random-direction null 0.011; shuffled-label 0.016) numerically consistent with Qwen swap (`|cos| = 0.021 [0.001, 0.037]` at L*=22; neighborhood mean 0.015). Single-pass unified-prompt variant (`|cos| = 0.139`) honestly reported and still << 0.3 threshold.
- **Touched in iterations**: [1] — consistency check + ⓪ narrative caveat.
- **Final status**: **PASS (held)** — architecture-independent, geometric near-orthogonality confirmed.
- **Notes for downstream (paper-side)**:
  - The wording "near-orthogonal linear subspaces" (proposal title) should be replaced with "near-orthogonal optimal linear probe directions" throughout the paper. The reviewer's framing note: this is strong evidence *against* a simple colinear knowledge-deficit story, **but not definitive proof of a clean two-subspace mechanistic factorization** — in high dimensions, many useful readouts can be nearly orthogonal without the underlying computations being causally independent.
  - The context-sensitivity of `v_c` across elicitation contexts (`cos(v_c_single, v_c*) = 0.47`) should be reported explicitly alongside the context-stable `v_v` (`cos = 0.025`).

---

## Section 2 — FAIL Claims (full journey)

*none — no claim in `verify_failed` bucket.*

---

## Section 3 — INCONCLUSIVE Claims (main-experiment-fix journey)

*none — no claim in `verify_inconclusive` bucket.*

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims (variant-fix journey)

*none — no claim in `verify_zero_eligible_variants` bucket.*

---

## Section 4b — INTEGRITY_ONLY Claims (no-action-with-upgrade-suggestion)

Four claims (C1, C2, C3b, C3c) landed in INTEGRITY_ONLY with `stage2_skip_reason: max_verify_claims_cap` — Stage 1 audit passed (WARN) but Stage 2 swap-test was deferred by the pre-registered `MAX_VERIFY_CLAIMS = 1` cap (C3a was picked as load-bearing). Per the routing contract, this bucket is **no-action-with-upgrade-suggestion**. The reviewer explicitly reviewed each claim and **did not identify a genuine methodology bug** beyond the pre-registered cap; each is carried into `Section 8 — Open Items`.

### 4b.1 `C1` — gold-correctness probe linearly accessible
- **Main-experiment verdict**: supported (AUROC = 0.840 [0.818, 0.835] at L*=31; ECE_iso = 0.031; nulls at chance).
- **Stage 1 audit**: WARN (experiment). Bootstrap CI does not contain point estimate (normal statistical artifact — bootstrap mean vs full-data). Bootstrap n=200 vs planned 1000 (disclosed).
- **Reviewer per-claim note**: "Not a methodology bug. In the paper, do not oversell this as discovering a clean latent 'truth neuron'-style variable; it is a decodable correctness correlate on TriviaQA."
- **Iteration action**: type ⓪ narrative-only — paper caveat "decodable correctness correlate on TriviaQA, not a task-general truth variable". No back-edge.
- **Upgrade instruction (post-iteration)**: `/auto-verify C1 -- resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run).

### 4b.2 `C2` — verbalized-confidence probe linearly accessible pre-emission
- **Main-experiment verdict**: supported (AUROC = 0.948 [0.913, 0.940]; ordinal top-1 = 0.994; macro-F1 = 0.862).
- **Stage 1 audit**: WARN (experiment). Proxy GT (model's own verbalization, by design). Bootstrap CI mismatch. Paraphrase Δ AUC P1 = -0.11 slightly exceeds ≤ 0.10 tolerance (AUC still above 0.70 floor, disclosed).
- **Reviewer per-claim note**: "Also plausibly supported in the narrow sense. Given the extreme ceiling/skew (96% ≥ 95), a probe recovering verbalized confidence with very high AUC is unsurprising. Paraphrase drop is a real caveat but not a methodology bug — it is an interpretation limitation already disclosed. Paper caveat: frame this as readout of the model's impending verbalized-confidence behavior under your prompting protocol, not a robust task-general introspective confidence variable."
- **Iteration action**: type ⓠ narrative-only — paper framing caveat above. No back-edge.
- **Upgrade instruction (post-iteration)**: `/auto-verify C2 -- resume: true`.

### 4b.3 `C3b` — causal separability under activation steering
- **Main-experiment verdict**: supported on PRIMARY internal readout (all 4 cross-direction Δ criteria pass in absolute mode); null on SECONDARY emitted output.
- **Stage 1 audit**: WARN (experiment + mechanism). Sparse α grid (3 values across 1 order of magnitude; audit criterion is < 3 orders of magnitude → WARN). Single random-direction control (n = 1 vs 30 recommended).
- **Reviewer per-claim note**: "The raw numbers are not impressive — steering v_v changes probe_c by ~0.41 vs matched random ~0.41–0.43, essentially identical on the key cross-effect. With only one random control and three α points, this is not persuasive causal evidence. The emitted-output null is important: if steering does not move actual accuracy or verbalized confidence, the causal claim should remain very modest. The paper should sharply downgrade this from 'causal separability' to 'weak, inconclusive steering diagnostic with no output-level effect.'"
- **Iteration action**: type ⓠ narrative-only — downgrade paper wording as above. No back-edge (INTEGRITY_ONLY by cap; genuine strengthening would require a denser α grid + more random-direction controls, which is what the upgrade path executes).
- **Upgrade instruction (post-iteration)**: `/auto-verify C3b -- resume: true`.

### 4b.4 `C3c` — dissociation-when-disagree diagnostic
- **Main-experiment verdict**: not-supported [provisional — suspected under-power] (z = 1.35, p_one_sided = 0.911; sign reversed; load-bearing cell (probe_low, verbal_low) n = 4 / 2000).
- **Stage 1 audit**: WARN (experiment). Load-bearing cell n = 4/2000 test — under-powered. Honestly reported.
- **Reviewer per-claim note**: "This is a clean provisional null and I appreciate that you did not spin it. The sign reversal plus n=4 in the load-bearing cell means the diagnostic is currently non-evidential. Given the 96% mass at confidence ≥ 95, this cell structure is largely a consequence of the model's degenerate verbal-confidence behavior under this prompt, not necessarily a coding bug. So I accept the no-action treatment here. In the final paper this should be presented as a failed/underpowered diagnostic, not as support for the broader thesis."
- **Iteration action**: type ⓠ narrative-only — present as failed/underpowered diagnostic; do not fold into the thesis. Accept the provisional null under UNDERPOWER = tag. No back-edge.
- **Upgrade instruction (post-iteration)**: `/auto-verify C3c -- resume: true` — but note the reviewer flagged that the *root* fix here is a prompt/binning change (Likert-mapped subset, permutation test, or larger n), not a swap-test. Any future re-attempt should watch for researcher degrees of freedom.

---

## Section 5 — Legacy DEFERRED Claims (empty under current architecture)

*none.*

---

## Section 6 — Cross-Cutting Patterns

From `REVIEWER_MEMORY.md` iteration 1 — patterns to track (all are paper-writing / framing patterns, not methodology-bug patterns):

- **Pattern P1 — Scoping / language over-reach**: The reviewer's headline "almost" verdict comes from the risk that the paper's title/theme overstates *near-orthogonal probe directions* as *near-orthogonal latent computational subspaces*. Touched claims: C3a (primary), C1 (framing "truth variable"), C3b ("causal separability" → "weak diagnostic"). Resolution at termination: **partially resolved** — the ⓠ narrative caveats above address the wording, but the paper draft itself is downstream of this loop and must implement them.
- **Pattern P2 — Verbalized-confidence distribution degeneracy**: 96% of verbalized c ≥ 95 (mean 98.6, std 10.3). Consequences chain across claims: (a) C2's very high AUROC 0.948 is partly explainable by the degenerate target distribution, (b) C3c's load-bearing cell (probe_low, verbal_low) collapses to n=4/2000, (c) paraphrase Δ AUC of -0.08 to -0.11 signals prompt-sensitivity of the verbalized target itself. Touched claims: C2, C3c. Resolution at termination: **acknowledged but not addressed** — this is a genuine Llama-3.1-8B-Instruct behavior under the P0 prompt, not a bug. Upgrade path (out-of-scope): re-do C2/C3c with a Likert-mapped C2 probe target and permutation-test on C3c.
- **Pattern P3 — Statistical-presentation cleanliness**: Bootstrap CI not containing point estimate is a normal bootstrap-mean-vs-full-data artifact but reads sloppy to a reviewer. Touched claims: C1, C2. Resolution at termination: **not yet resolved on-disk** — paper writeup should either explain the artifact explicitly or re-do bootstraps with the point-estimate-anchored percentile method.
- **Pattern P4 — Cap-driven under-verification**: Four of five claims were INTEGRITY_ONLY due to `MAX_VERIFY_CLAIMS = 1`. The reviewer treated this as an acceptable pre-registered choice given the load-bearing status of C3a, but the paper should be transparent that only C3a is architecturally cross-verified. Touched claims: C1, C2, C3b, C3c. Resolution at termination: **carried forward** — Section 8 lists all four upgrade commands.

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed**: 0 / 6
- **Claim-reentries consumed**: 0 / 2
- **Iteration `/run-experiment` calls**: runs_total = 0
- **Iteration GPU-hours**: gpu_hours_total = 0.0

### Per-iteration breakdown

| Iter | Type | Target claims | Produced claims | Runs | GPU-hours | Score after | Verdict after |
|---|---|---|---|---|---|---|---|
| 1 | ⓠ narrative-only (STOP fired) | C1, C2, C3a, C3b, C3c | — | 0 | 0.0 | 6.5 | almost |

**Interpretation**: Iteration 1 was a pure ⓠ iteration (paper-side caveats + INTEGRITY_ONLY carry-forward + PASS-claim consistency check). The three-dimensional STOP rule fired at Phase B; no Phase-C back-edge dispatch was needed. Zero iteration-loop budget consumed; the full 6-iteration budget and 2-reentry sub-budget remain available if any of the four `/auto-verify -- resume: true` upgrade paths (Section 8) are executed later and re-enter the loop.

---

## Section 8 — Open Items for Human Reviewer

> Items the loop could not close on this pass. These are non-blocking (all in INTEGRITY_ONLY) but should be closed before submission if compute allows.

- **Still-FAIL claims**: none.
- **Still-INCONCLUSIVE claims**: none.
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: none.

- **INTEGRITY_ONLY claims (Stage 2 skipped — not stress-tested)**:

  | Claim | `stage2_skip_reason` | Upgrade instruction | `main_experiment_integrity` | `warn_source` |
  |---|---|---|---|---|
  | C1  | `max_verify_claims_cap` | `/auto-verify C1 -- resume: true`  | warn | experiment |
  | C2  | `max_verify_claims_cap` | `/auto-verify C2 -- resume: true`  | warn | experiment |
  | C3b | `max_verify_claims_cap` | `/auto-verify C3b -- resume: true` | warn | experiment+mechanism |
  | C3c | `max_verify_claims_cap` | `/auto-verify C3c -- resume: true` | warn | experiment |

- **Legacy deferred claims (empty in new runs)**: none.

- **Recurring unresolved patterns** (from Section 6):
  - **P1 — scoping/language over-reach**: paper wording must be tempered per the ⓠ caveats in Section 4b before submission.
  - **P2 — verbalized-c distribution degeneracy**: a Likert-mapped C2 probe target and a permutation-test on C3c would materially strengthen those two claims (upgrade path is out-of-scope for this loop's budget).
  - **P3 — statistical presentation**: bootstrap CI mismatch (C1, C2) should either be explained explicitly in the paper or re-computed with a point-estimate-anchored percentile method.
  - **P4 — cap-driven under-verification**: the paper should be transparent that only C3a is architecturally cross-verified; C1/C2/C3b/C3c cross-model verification remains as future work.

- **Claim-reentry refusals** (where reviewer requested ③ but sub-budget was exhausted): none. Reviewer never requested ③.

---

## Recommended next step (for the orchestrator)

**READY-for-paper-writeup with tempered scoping** (per reviewer). Recommended:
1. Implement the ⓠ narrative caveats in the paper's Abstract, Introduction, and Section 3 wording.
2. (Optional, budget-permitting: ~9.4 GPU-h remain of 10-h HARD budget) run one or more of the four upgrade `/auto-verify ... -- resume: true` commands from Section 8 to promote INTEGRITY_ONLY → PASS/FAIL for C1/C2/C3b/C3c. Each such upgrade re-enters this iteration loop with `resume: true` and consumes iteration budget only if the swap-test verdict is FAIL / INCONCLUSIVE / ZEV.
3. Do NOT retry C3c under this budget: the underlying Llama-3.1-8B-Instruct verbalized-c skew makes the current predicate untestable; researcher degrees of freedom in re-binning risk over-fitting.
