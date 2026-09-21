# Auto-Iteration Review Log

**Project**: Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence in Llama-3.1-8B-Instruct on TriviaQA
**Started**: 2026-07-14
**Reviewer LLM**: gpt-5.4 (dmxapi.cn bypass proxy)
**Budget**: MAX_ITERATIONS=6, MAX_CLAIM_REENTRIES=2, TARGET_SCORE=6

---

## Iteration 1 (2026-07-14)

### Assessment (Summary)
- **Score**: 6.5/10
- **Verdict**: almost
- **Budget after this iteration**: iterations 0/6, claim-reentries 0/2 (Phase C was ⓪-only; no back-edge action)
- **Key criticisms**:
  - Paper-level framing risk: near-orthogonal *probe directions* is a weaker statement than near-orthogonal *latent computations*; the paper should be tightly scoped around C3a with tempered causal language.
  - C3b causal steering evidence is weak-to-null in raw terms — the paper should downgrade "causal separability" to "weak, inconclusive steering diagnostic with no output-level effect."
  - C2 verbalized-confidence is extreme-skewed + prompt-sensitive; the probed direction may be partly formatting/default-style rather than pure introspective confidence.
  - C3c distribution collapse is honestly reported and accepted as a provisional null under UNDERPOWER=tag policy.
  - Bootstrap CI not containing point estimate is a normal artifact but should be cleaned up in the paper.
- **Reviewer action verdict**: none — every criticism is either (a) an INTEGRITY_ONLY claim's post-cap upgrade path (out-of-scope by contract), (b) a paper-writing scoping/framing issue (⓪ narrative-only), or (c) an accepted honest null (C3c).

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

- **Score**: 6.5/10
- **Verdict**: almost

**Per-claim assessment**

**C1**: The result is strong enough to support the narrow claim that a linear probe can read out correctness information from hidden states on this setup: AUC 0.84 is comfortably above chance and the nulls behaving at ~0.5 are the right sanity checks. ECE_iso=0.031 also suggests the signal is not just rank-ordering noise. The CI mismatch is sloppy but not fatal; with only 200 bootstraps, this is believable as a bootstrap-summary artifact rather than evidence of instability. The bigger caveat is conceptual: "gold correctness accessibility" here is still dataset/model/task-specific and not obviously a pure internal calibration variable. Since this claim is INTEGRITY_ONLY, I would not ask for action in this loop. But in the paper, do not oversell this as discovering a clean latent "truth neuron" style variable; it is a decodable correctness correlate on TriviaQA.

**C2**: Also plausibly supported in the narrow sense. Given the extreme ceiling/skew in verbalized confidence, it is unsurprising that a probe can recover it with very high AUC / ordinal accuracy. In some sense this is the least interesting of the claims because the target is the model's own emitted confidence proxy, and the distribution is so degenerate that even coarse separation can look excellent. The paraphrase drop of -0.11/-0.08 is a real caveat: it slightly weakens the "stable internal verbal-confidence signal" interpretation, especially when the target itself is prompt-sensitive. Still, this is not a methodology bug; it is an interpretation limitation already disclosed. Paper caveat: frame this as readout of the model's impending verbalized-confidence behavior under your prompting protocol, not a robust task-general introspective confidence variable.

**C3a**: This is the strongest part of the paper and, on the numbers provided, internally consistent. Main result |cos|=0.015 with CI [0.001, 0.034], neighborhood mean 0.025, and random/shuffled nulls around 0.011–0.016 all tell the same story: the angle is effectively what you would expect from unrelated directions in a 4096-d space. The swap-test replication on Qwen2.5-7B-Instruct gives |cos|=0.021 [0.001, 0.037], which is very much the same qualitative effect at a different best layer. So yes, the experiment and robustness numbers are consistent. The caveat I would absolutely surface in the final paper is that "near-orthogonality of optimal linear probe directions" is not identical to "the underlying internal computations are causally independent." In high dimensions, many useful readouts can be nearly orthogonal, and the result depends on probe objective, layer choice, target definition, and representation basis. This is strong evidence against a simple colinear knowledge-deficit story, but not definitive proof of a clean two-subspace mechanistic factorization.

**C3b**: As written, this does not do much for causal separability. The internal-readout criterion may pass under your preregistered thresholds, but the raw numbers are not impressive: steering the verbal-confidence direction changes probe_c by ~0.41, essentially identical to random-direction control (~0.41–0.43), which is close to a null on the key cross-effect test. On the other side, the probe_v changes are small, but with one random control and only three alpha points, I would not call that persuasive causal evidence. The emitted-output null is honestly reported and important: if steering does not move actual accuracy or verbalized confidence, then the causal claim should remain very modest. Since this is INTEGRITY_ONLY, I will not propose action, but the paper should sharply downgrade this from "causal separability" to "weak, inconclusive steering diagnostic with no output-level effect."

**C3c**: This is a clean provisional null, and I appreciate that you did not spin it. The sign reversal plus n=4 in the load-bearing cell means the diagnostic is currently non-evidential. Given the 96% mass at confidence ≥95, this cell structure is largely a consequence of the model's degenerate verbal-confidence behavior under this prompt, not necessarily a coding bug. So I accept the no-action treatment here. Still, in the final paper this should be presented as a failed/underpowered diagnostic, not as support for the broader thesis. If anything, it highlights how limited the verbalized-confidence variable is in this setting.

- **Actions**: none

## Memory update
- Primary positive: C3a appears genuinely robust and is the only clearly venue-worthy claim at present; main numbers and swap-test replication are consistent.
- Main skepticism: the paper's title/theme risks overstating what is really shown. Near-orthogonal *probe directions* are weaker than near-orthogonal *latent subspaces/processes*.
- C2 concern to track: verbalized confidence is extremely skewed and prompt-sensitive; this may make the "verbalized confidence direction" partly a readout of formatting/default style rather than introspective uncertainty.
- C3b concern to track: causal evidence is weak-to-null in raw terms, especially because random-direction controls look similar and output-level effects are null.
- C3c concern to track: dissociation-under-disagreement diagnostic currently fails due to distribution collapse; if future iterations change prompting/binning, watch for researcher degrees of freedom.
- Statistical presentation concern: bootstrap CI not containing point estimate should be cleaned up in the paper to avoid avoidable reviewer distrust.
- Submission status: technically not blocked by routing rules, but scientifically this feels **almost**, not fully ready, unless the paper is tightly scoped around C3a as the main result and heavily tempers causal/mechanistic language.

</details>

### Verify-Passed Claims (brief audit)
- **C3a — CONSISTENT**: The main-experiment numbers in `refine-logs/EXPERIMENT_RESULTS.md#M2` (`|cos|=0.015 [0.001, 0.034]` at L*=31; neighborhood mean L*±2 = 0.025; random-direction null 0.011; shuffled-label 0.016) are numerically consistent with the swap-test in `verify/C3a_near_orthogonality/ROBUSTNESS.md` (Qwen2.5-7B-Instruct: `|cos|=0.021 [0.001, 0.037]` at L*=22; neighborhood mean = 0.015). Same qualitative story — angle is indistinguishable from a random-direction null in the model's 4096-d residual stream. The single-pass unified-prompt variant (|cos|=0.139) is honestly reported. **Paper-side caveat (reviewer-flagged)**: the wording "near-orthogonal subspaces" must be replaced with "near-orthogonal optimal linear probe directions" to avoid overstating a computational-independence claim from a geometric-angle observation.

### Actions Taken (per claim, per type)
- **C3a — type ⓪ narrative-only**: Add paper-side caveat "near-orthogonality of optimal linear probe directions is not identical to causal independence of underlying computations". No scripts or data touched. No iteration budget consumed.
- **C2 — type ⓪ narrative-only**: Add paper-side framing caveat "readout of impending verbalized-confidence behavior under this prompting protocol, not a task-general introspective confidence variable". Do NOT propose ①/②/③ (INTEGRITY_ONLY by cap; reviewer accepted no-action contract). No iteration budget consumed.
- **C3b — type ⓪ narrative-only**: Downgrade paper wording from "causal separability" to "weak, inconclusive steering diagnostic with no output-level effect" (per reviewer). Do NOT propose ②; INTEGRITY_ONLY by cap. No iteration budget consumed.
- **C3c — type ⓪ narrative-only**: Present as a failed/underpowered diagnostic, honestly. Reviewer accepted the provisional null under UNDERPOWER=tag. Root cause is Llama-3.1-8B-Instruct's degenerate verbalized-c behavior (96% ≥ 95), not a bug. No iteration budget consumed.
- **C1 — type ⓪ narrative-only**: Paper-side caveat "decodable correctness correlate on TriviaQA, not a task-general truth variable". Do NOT propose ①/②/③; INTEGRITY_ONLY by cap. No iteration budget consumed.

### Claim Rewrites (type ③ — empty when no rewrite this iteration)
none

### Claim-Stage Re-entries Triggered (orchestrator handoff — empty unless type ③ full path used this iteration)
none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C1** [stage2_skip_reason: max_verify_claims_cap]: upgrade path `/auto-verify C1 -- resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run) [main-experiment integrity: warn; warn_source: experiment]
- **C2** [stage2_skip_reason: max_verify_claims_cap]: upgrade path `/auto-verify C2 -- resume: true` [main-experiment integrity: warn; warn_source: experiment]
- **C3b** [stage2_skip_reason: max_verify_claims_cap]: upgrade path `/auto-verify C3b -- resume: true` [main-experiment integrity: warn; warn_source: experiment+mechanism]
- **C3c** [stage2_skip_reason: max_verify_claims_cap]: upgrade path `/auto-verify C3c -- resume: true` [main-experiment integrity: warn; warn_source: experiment]

### Results
- [run-experiment] iteration=1 runs_this_iteration=0 gpu_hours_this_iteration=0 cumulative_gpu_hours=0
- No new experiments dispatched: all reviewer criticisms were paper-writing scoping/framing issues (⓪ narrative-only) or INTEGRITY_ONLY no-action-by-contract.
- Verify-passed claim (C3a) numerically consistent between main-experiment and Qwen swap.

### Status
- STOP conditions satisfied (three-dimensional): score 6.5 ≥ 6, verdict "almost" ∈ {ready, almost}, no claim in FAIL/INCONCLUSIVE/ZEV. INTEGRITY_ONLY does NOT block STOP.
- **completed** — proceeding to Termination and Final Report assembly.
