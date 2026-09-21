# Auto Review — Sparse Modular Circuit for Propositional-Logic Reasoning (Mistral-7B)

**Loop config**: MAX_ITERATIONS=6, MAX_CLAIM_REENTRIES=2, TARGET_SCORE=6, AUTO_PROCEED=true, GPU_ID=0,1,2,3.
**Reviewer LLM**: `gpt-5.4` @ `https://www.dmxapi.cn/v1` (source: shell env).
**Verify context (initial)**:
- verify_passed: [C3]
- verify_failed: []
- verify_inconclusive: []
- verify_zero_eligible_variants: []
- verify_integrity_only: [C1, C2] (both stage2_skip_reason: max_verify_claims_cap)

---

## Iteration 1 (2026-07-15)

### Assessment (Summary)

- **Score**: 7/10
- **Verdict** (canonical): **ready**
- **Budget after this iteration**: iterations 0/6 (⓪ narrative-only actions do not consume budget); claim-reentries 0/2
- **Key criticisms** (paper-side, no back-edge required):
  1. C1 minimality is measured from a 20/158 component sample and sparsity holds only at the hard-cap (0.150) — the "sparse" claim is threshold-satisfying, not a discovered optimum.
  2. C2's negative conclusion should be scoped to "under the tested top-40 subset", not stated as a global anti-modularity claim.
  3. C3's insufficiency is real, but the interpretive gloss "the other 85% is task-general infrastructure" outruns what the intervention establishes.
  4. Cross-family robustness is thin — one additional model (Gemma-2-9B), reused M5 result, no fresh verify GPU run.
  5. Anchor-cell dependency for the C3 numbers is a scope caveat.
- **STOP rule check** (three-dimensional):
  1. `score = 7 ≥ TARGET_SCORE = 6` ✓
  2. canonical verdict `ready` ∈ `{ready, almost}` ✓
  3. `verify_failed = []`, `verify_inconclusive = []`, `verify_zero_eligible_variants = []` (all empty) ✓
  → **STOP conditions satisfied — terminate the loop after Phase E**.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

## Score

**7/10**

Good negative-result reproduction with a clear core finding, decent quantitative support, and a cross-family robustness check on the main asymmetry. Not top-tier as-is because the headline package is weakened by limited verification coverage, a nontrivial integrity warning on C1, a scope-reduced analysis for C2, and a central interpretive leap ("distributed code / infrastructure lives in the other 85%") that is plausible but not actually established by the presented interventions.

## Verdict

**READY for submission: Yes**

By your stated criterion, this is ready: there are **no FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS verify states** blocking readiness, and **INTEGRITY_ONLY does not block READY**. More importantly, the scientific core is coherent and publishable **as a partial refutation / negative reproduction**:

- **C3**: robustly **not supported** in the intended "necessary and sufficient" form, with a replicated **necessity-yes / sufficiency-no asymmetry** across Mistral-7B and Gemma-2-9B.
- **C1**: only partial support even in the main experiment.
- **C2**: refuted on the reported evidence.

So yes, this is a legitimate submission-ready negative finding. But it should be written **as a constrained negative reproduction**, not as a decisive mechanistic characterization of propositional-logic reasoning.

## Per-claim assessment

### C1 — sparse component set

**Status:** **PARTIAL** in main experiment; **INTEGRITY_ONLY** in verify.

**Narrative check**
- Reported metrics:
  - sparsity fraction = **0.150** vs target **≤ 0.15**
  - completeness = **0.955** vs target **≥ 0.9**
  - minimality avg single-removal drop = **0.0014** vs target **≥ 0.05**
- Interpretation is internally consistent: the shortlist is **small enough by thresholding convention** and **highly complete**, but **not minimal** in any meaningful sense. That is exactly what these numbers say.

**Numeric consistency check**
- No arithmetic contradiction in the summary.
- However, calling this "sparse" is fragile:
  - **0.150 exactly at cap** means this is threshold-satisfying, not evidence of a naturally emerging sparse minimum.
  - The shortlist size (**158 components**) and minimality (**0.0014**) jointly imply heavy redundancy / over-inclusion under the chosen ranking.
- The disclosed integrity warning matters a lot: **minimality sampled only 20/158 components**. Since minimality is the failed sub-criterion and a core part of C1, this limitation should be surfaced prominently in the paper.

**Caveat that must surface in final paper**
- **Yes**: "minimality was estimated from a 20/158 sample rather than exhaustively."
- Also: "sparsity hit the preset hard cap rather than a discovered optimum."

**Bottom line**
- Fair to present as **partial support only**.
- Not fair to present as evidence for a genuinely sparse circuit without stronger minimality analysis.

---

### C2 — modular decomposition

**Status:** **FAIL** in main experiment; **INTEGRITY_ONLY** in verify.

You instructed not to propose fixes for INTEGRITY_ONLY claims, so I won't.

**Assessment**
- The evidence as presented does **not** support modular fact/rule/answer specialization.
- The reported numbers are substantially below threshold:
  - median dominance = **1.20** vs **≥ 2.0**
  - dissociation = **0.089 / 0.027 / 0.026** vs **≥ 0.1**
  - p-values = **0.05 / 1.0 / 1.0** vs **≤ 0.01**
  - stability Jaccard = **0.75 / 0.36 / 0.77**
- This is not a near miss. It is a real failure for rule and answer specialization under the operationalization used.

**Important caveat**
- The analysis used a **top-40 subset of a 158-component shortlist** for M4/M4.stab. That sharply limits the strength of the negative conclusion. It does **not** invalidate the failure, but it does mean the paper should phrase it as:
  - "No strong modular decomposition was found under the tested shortlist-and-top-40 analysis"
  - not
  - "the model lacks modular decomposition."

**Bottom line**
- Scientifically useful negative result.
- But the scope restriction must be disclosed clearly and early.

---

### C3 — necessity + sufficiency

**Status:** **PASS** in verify, in the sense that the study's conclusion on C3 is robustly **not-supported**; main result replicated cross-family.

**Narrative check**
The study claim is that C3 requires **both** necessity and sufficiency. What you found:
- **Necessity passes strongly**
  - Mistral path patching recovery LD = **0.955**
  - specificity gap = **0.836**
- **Sufficiency fails decisively**
  - Mistral reinsertion recovery LD = **0.113**
  - per-seed std = **0.036**
- **Cross-family recurrence**
  - Gemma-2-9B necessity LD = **1.018**
  - Gemma-2-9B sufficiency LD = **0.019**

This is internally coherent and exactly supports the paper's intended negative result: **the shortlist is necessary-ish under path patching but very far from sufficient under reinsertion + ablation**.

**Numeric consistency check**
- The asymmetry is large, not marginal:
  - Mistral: **0.955 vs 0.113**
  - Gemma: **1.018 vs 0.019**
- The low per-seed std (**0.036**) is actually useful: this is not a noisy failure; it is a stable failure.
- Dose-response saturating at ~31/158 components is also consistent with redundancy on the necessity side.

**Key caveat that must surface in final paper**
- The verify pass is based on **one eligible variant** and **reused milestone output**, with **no fresh verify GPU run**. That is acceptable for consistency checking, but weak as a robustness package.
- The paper should not oversell this as broad cross-model validation. It is **one additional family/model**, not a general law.

**Interpretation caution**
- The empirical result "necessary but not sufficient" is strong.
- The explanatory gloss — "the remaining 85% contains task-general infrastructure" — is **speculative**. The intervention only shows that ablating everything outside the shortlist destroys performance. It does **not** isolate whether what is missing is:
  - generic infrastructure,
  - many low-ranked task-relevant contributors,
  - nonlinear interactions,
  - patching/reinsertion mismatch artifacts,
  - or ablation-induced distribution shift.
- The claim should therefore stay interventionally modest.

**Bottom line**
- This is the strongest part of the paper.
- It is publishable and should be the centerpiece.

## Overall assessment

This is a **solid negative reproduction** with a real mechanistic lesson:

1. A top-attribution shortlist can look **complete under necessity-style interventions**.
2. The same shortlist can fail **catastrophically under sufficiency-style interventions**.
3. Therefore, "important under patching" does **not** imply "implements the computation by itself."

That is a worthwhile result and exactly the kind of thing the field needs more of.

That said, the work is not as strong as it could be, and the current package has four notable weaknesses:

- **Verification coverage is thin.**
  - Only C3 got stage-2 verify.
  - C1 and C2 remain **INTEGRITY_ONLY**, not independently stressed.
- **C1's minimality evidence is weakly measured.**
  - Sampling **20/158** for minimality is a meaningful limitation.
- **C2's negative conclusion is narrower than the prose may suggest.**
  - Using **top-40 of 158** weakens any strong anti-modularity claim.
- **The mechanistic interpretation outruns the intervention.**
  - "Distributed code / infrastructure in the other 85%" is plausible, but the data shown establish insufficiency, not the specific reason for insufficiency.

So: **submission-ready, but not polished enough for a top-score recommendation**. If written carefully as a negative reproduction with explicit scope limits, I'd view it as credible and useful.

## Memory update

- Suspicion 1: **C1 minimality is under-supported** because the failure metric comes from a **20/158 sample**, and sparsity only holds at the **hard cap** rather than a discovered optimum.
- Suspicion 2: **C2 may be partly an analysis-artifact negative** due to restricting modularity analysis to **top-40 of 158** components; track whether full-shortlist role analyses change the conclusion.
- Suspicion 3: **C3 sufficiency failure may reflect ablation/reinsertion pathology**, not only true distributed computation; watch for evidence disentangling infrastructure loss vs low-ranked task signal vs intervention mismatch.
- Suspicion 4: **Cross-family robustness is currently shallow**: only one extra model, reused result, no fresh verify run; track whether any additional swap/stress tests are added.
- Suspicion 5: The paper must avoid **overclaiming mechanism discovery**; current evidence supports a **negative claim about shortlist sufficiency**, not a definitive positive account of where computation resides.
- Suspicion 6: Watch for any hidden dependence on the **anchor cell (`k3_chain2_natural`)**; current core finding may be real but still narrower than "propositional-logic reasoning in Mistral-7B" broadly.

</details>

### Verify-Passed Claims (brief audit)

- **C3**: consistency check ✓ — the reviewer confirms narrative + numeric consistency. The main-experiment WARN caveat (KL scaling artifact — baseline KL=0.043 causing recovery_KL=-5.618 in M2) should surface in the paper's methods section. Reviewer flags an additional caveat not previously highlighted: "verify pass is based on one eligible variant and reused milestone output, with no fresh verify GPU run" — this is acceptable for consistency but weak as robustness; the paper should not oversell as broad cross-model validation.

### Actions Taken (per claim, per type)

- **C3 — type ⓪ narrative-only**: paper must add caveats: (a) KL scaling artifact for M2 recovery_KL (already in Notes), (b) verify pass uses one variant / reused M5 result — not fresh GPU run, (c) "necessary-but-not-sufficient" empirical claim is strong but the "task-general infrastructure lives in the other 85%" interpretive gloss should stay interventionally modest. No scripts touched, no runs fired.
- **C1 — type ⓪ narrative-only**: paper must add caveats: (a) minimality metric estimated from 20/158 component sample (must be surfaced prominently in methods), (b) sparsity hit hard-cap 0.150, not a discovered optimum. The reviewer respected the INTEGRITY_ONLY contract and did not propose ①/②/③.
- **C2 — type ⓪ narrative-only**: paper must scope the negative conclusion to "under the tested top-40 subset" rather than a global anti-modularity claim. The reviewer respected the INTEGRITY_ONLY contract and did not propose ①/②/③.

### Claim Rewrites (type ③ — empty when no rewrite this iteration)

none

### Claim-Stage Re-entries Triggered (orchestrator handoff — empty unless type ③ full path used this iteration)

none

### Open Items — Unverified Under Swaps (from verify_integrity_only)

- **C1** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C1 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run) [main-experiment integrity: warn; warn_source: experiment — undisclosed minimality sampling 20/158 components]
- **C2** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C2 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run) [main-experiment integrity: warn; warn_source: experiment — shortlist scope reduction top-40 of 158 components for M4/M4.stab]

### Results

- No experiments fired this iteration (all reviewer weaknesses are paper-side caveats — type ⓪).
- iteration=1, runs_this_iteration=0, gpu_hours_this_iteration=0, cumulative_gpu_hours=0
- Per-claim outcome:
  - C3 → still PASS (robustness=1.00), paper-side caveats added to narrative record
  - C1 → still INTEGRITY_ONLY, upgrade path recorded in Open Items
  - C2 → still INTEGRITY_ONLY, upgrade path recorded in Open Items

### Status

**completed** — three-dimensional STOP rule fired iteration 1 (score 7 ≥ 6, verdict ready, no FAIL/INCONCLUSIVE/ZEV claims). No back-edge action was proposed or executed. Proceed to Termination and write `AUTO_ITERATION_FINAL_REPORT.md`.
