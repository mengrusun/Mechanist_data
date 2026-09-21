## Scores

1. **Problem Fidelity** — **8.9/10**  
2. **Method Specificity** — **8.8/10**  
3. **Contribution Quality** — **8.6/10**  
4. **Frontier Leverage** — **8.7/10**  
5. **Feasibility** — **8.0/10**  
6. **Validation Focus** — **8.4/10**  
7. **Venue Readiness** — **8.2/10**

### Weighted Overall Score
**8.6/10**

---

## Overall Assessment

This is a strong revision. It fixed the main issues from the prior round: the M0 gate is now properly binary, teacher retraining per seed is removed from the critical path, the mechanism arc is materially more concrete, and the judge audit / bootstrap are now correctly demoted to stability diagnostics rather than soft alternate pass criteria.

Most importantly, the proposal now behaves like a **validation-first paper with a gated mechanism follow-up**, rather than a phenomenon claim entangled with too many exploratory branches. That is the right shape for the frozen-claim regime.

The remaining weaknesses are not about claim drift. They are about a few **interface inconsistencies**, some **residual over-commitment inside the M0 gate semantics**, and **mechanism-stage statistical criteria that are still underspecified relative to the claimed causal conclusion**.

---

## Dimension-by-dimension review

### 1. Problem Fidelity — 8.9/10

This is largely faithful to the frozen claim.

What improved:
- You preserved the exact anchor inequalities.
- You kept the full-dataset and ≥3-seed requirement.
- You made mechanism conditional on M0 `PASS`.
- You retained the re-scan requirement and did not substitute a different benchmark.

Minor fidelity concern:
- The formal `PASS` definition now includes **judge calibration stability** and **mean-across-seeds satisfaction** in addition to the anchor’s per-seed criterion. These are reasonable verification hardenings, but they are also technically an added bar beyond the frozen claim as written.
- This is not severe drift, because you present them as protocol hardening rather than claim rewriting, but it is slightly stricter than the anchor.

Why I am not calling drift:
- The added conditions are about **measurement validity** and **run validity**, not changing the substantive phenomenon being tested.
- The core quantitative claim remains the same.

### 2. Method Specificity — 8.8/10

This is now implementable by an engineer. Good improvements:
- intervention site pinned,
- top-K pinned,
- off-target set pinned,
- metric options pinned,
- fallback logic pinned.

The strongest part is that the mechanism arc now has a real executable shape:
- extraction target,
- selection policy,
- intervention family,
- control family,
- expected signatures.

Remaining issue:
- a few decision rules still mix “descriptive” and “decisional” language in ways that could create ambiguity during execution, especially in M0 failure handling and mechanism success criteria.

### 3. Contribution Quality — 8.6/10

Much better focused than before. The contribution is now clearly:
1. a sharpened verification protocol for a frozen claim, and
2. a minimal mechanism ladder only after verification succeeds.

That is appropriate and does not oversell novelty.

Still, there is a little bit of scope pressure from:
- Stage-B regex + human audit,
- judge paraphrase matrix,
- optional Ctrl-C,
- L-Core plus L-Secondary,
- off-target audit logic.

Each of these is individually defensible, but together they are close to the upper bound of what still counts as “minimal.”

### 4. Frontier Leverage — 8.7/10

This uses the right primitives at roughly the right altitude:
- contrastive activation directions,
- residual-stream interventions,
- dose-response steering,
- matched random-direction controls,
- LoRA-space fallback.

Good choice to demote full LoRA attribution to fallback. Good choice to frame L-Core as contrastive direction extraction rather than pretending this is a full attribution-patching paper.

Remaining limitation:
- the mechanism still reads a bit too much like a hand-built bespoke stack rather than a very standard 2025 “extract direction → validate by intervention → test specificity” pipeline. This is mostly wording/interface, not substance.

### 5. Feasibility — 8.0/10

This is feasible in broad strokes on 4×80GB, but this is the weakest dimension among the higher-scoring ones.

The key positive change was removing per-seed teacher retraining. That likely saved the plan.

Remaining feasibility risks:
- The GPU-hour estimate is probably optimistic once you include:
  - repeated full QA_I evals,
  - activation caching across layers,
  - judge audit reruns,
  - seed replacement logic,
  - possible fallback to L-Secondary.
- The phrase “re-evaluate QA_I” appears several times in mechanism tests; depending on implementation, this can become the true bottleneck.
- Cross-seed intersection on top-3 layers is clean conceptually, but if the flipped-wrong set is small in one seed, direction estimates may become unstable and force fallback or repeated tuning.

Still feasible, but with less slack than the proposal suggests.

### 6. Validation Focus — 8.4/10

The hardening is now mostly proportional. The biggest win is that:
- bootstrap CI is no longer quasi-gating,
- judge audit no longer acts like a brittle veto on small label noise,
- Ctrl-C is optional and interpretive only.

The remaining excess is mostly procedural complexity around special-case invalidation logic. It is not egregious, but could still be simplified.

### 7. Venue Readiness — 8.2/10

If M0 passes cleanly and the mechanism arc yields even a moderate causal result, this is plausibly top-venue-worthy as a **careful validation + mechanism paper**. If M0 fails, the negative-result note is respectable, but top-venue readiness then depends heavily on how cleanly the controls rule out trivial explanations.

Current state:
- stronger than a workshop draft,
- not yet fully polished into a top-tier-ready experimental plan,
- but close.

The main remaining gap is that some of the method interface still needs to be more internally consistent so that the paper cannot be criticized for “protocol elasticity.”

---

## Dimensions below 7
**NONE**

---

## Main remaining weaknesses and fixes

Even though no dimension is below 7, there are still important issues worth fixing before calling this READY.

### 1. M0 gate semantics are still slightly internally inconsistent
**Weakness:**  
You define `PASS` as requiring per-seed inequalities across ≥3 seeds **and mean across seeds satisfies both inequalities**. But in `FAIL`, you also say “any per-seed inequality fails after re-run at a fresh seed replacement (i.e. at least ⌈2/3⌉ of the required seeds pass).” That parenthetical conflicts with the earlier statement that the claim is enforced per seed across ≥3 seeds. It reintroduces ambiguity about whether seed replacement can rescue an initial failing seed and what the final denominator is.

**Concrete method fix:**  
Define the seed protocol as:
- Pre-register exactly 3 primary seeds.
- If a run is `RUN INVALID`, rerun the **same seed** after fixing tooling.
- If a scientific failure occurs on any of the 3 valid seeds, M0 = `FAIL`.
- Only use extra seeds in an appendix robustness extension, not as replacement for a valid failed seed.

If you want replacement, then specify a fixed rule like “draw seeds until 3 valid non-tooling runs are obtained,” but **do not** permit replacing scientific failures.

**Priority:** IMPORTANT

---

### 2. Judge audit should not be part of `PASS`; it should only determine `RUN INVALID`
**Weakness:**  
You currently bake judge calibration stability into `PASS` and `FAIL` language in a way that risks turning a measurement diagnostic into a scientific bar. The prior revision correctly moved away from a brittle veto, but the proposal still sometimes treats audit findings as changing scientific status rather than labeling validity.

**Concrete method fix:**  
Refactor the M0 interface as:
- Scientific `PASS/FAIL` determined only by the anchor inequalities on validly measured runs.
- Judge audit only determines whether the run is **measurement-valid**:
  - if arm ordering flips or flip rate >10%, label `RUN INVALID`,
  - otherwise proceed and report matrix descriptively.
- Remove any wording that maps judge-audit problems directly to scientific `FAIL` unless repeated prompt redesign still cannot produce a valid evaluator, in which case terminate the project as “unable to measure,” not phenomenon-false.

**Priority:** IMPORTANT

---

### 3. Mechanism success criteria need a cleaner pre-registered hierarchy
**Weakness:**  
Claim 2 currently mixes several standards:
- recovery ≥30%,
- monotone dose-response,
- slope significantly bounded away from 0,
- specificity ≤1 pp,
- partial claims if some but not all pass.

This is reasonable scientifically, but not yet crisp enough for a skeptical reviewer. It leaves room for ex post interpretation.

**Concrete method fix:**  
Pre-register a three-level mechanism verdict:
- **STRONG POSITIVE:** 2a recovery ≥30% in at least 2/3 seeds, 2b monotone median trend with negative pooled slope, 2c all controls ≤1 pp.
- **PARTIAL POSITIVE:** 2a passes but either 2b or 2c misses.
- **BOUNDED NULL:** no stable candidate or 2a fails.

Also specify exactly how monotonicity is tested:
- e.g. Spearman ρ between α and accuracy pooled by seed, with negative sign expected, plus visual curve.

**Priority:** IMPORTANT

---

### 4. The primary location metric should be singular, not dual
**Weakness:**  
You currently say the logit-margin metric is first-token letter margin **or** accuracy-conditioned activation contrast on the flipped-wrong subset, depending on evaluation channel. That is understandable, but it still gives the mechanism stage two partly different anchors.

**Concrete method fix:**  
Choose one primary location score:
- **Primary:** accuracy-conditioned activation contrast on the flipped-wrong vs matched-agree partition.
- **Secondary diagnostic:** option-letter first-token margin when available.

This better matches the judged free-form answer regime and avoids overfitting the mechanism to tokenization details.

**Priority:** MINOR

---

### 5. Feasibility estimate should include a “must-fit” activation-caching implementation
**Weakness:**  
The compute section is plausible but underspecified where it matters most: activation extraction and intervention evaluation. Without a concrete caching plan, reviewers may doubt the 50 GPU-hour total.

**Concrete method fix:**  
Add one implementation constraint:
- cache only the pinned site residual vectors at the final prompt position, for selected arms/items, in bf16 or fp16;
- do not cache full token trajectories;
- for L-Core, subsample to the flipped-wrong and matched-agree sets only, capped by a pre-registered max item count per seed if necessary.

This would make the mechanism compute estimate much more credible.

**Priority:** IMPORTANT

---

## Simplification Opportunities

1. **Delete the mean-across-seeds ≥3% condition from formal `PASS`.**  
   Keep mean ± std as a reporting summary only. The frozen claim is already per-seed across ≥3 seeds; adding a mean threshold is unnecessary strengthening.

2. **Delete seed replacement for scientific failures.**  
   Replacement should be for `RUN INVALID` only. This simplifies the gate and removes ambiguity.

3. **Merge L-Secondary trigger wording into one line.**  
   “Run LoRA-row attribution only if no layer appears in the top-3 of all seeds under L-Core.”  
   This is simpler than the current more diffuse wording.

---

## Modernization Opportunities

1. **Use pooled seed-wise rank aggregation for layer selection rather than strict top-3 intersection.**  
   Strict intersection is brittle. A more natural 2025 choice is Borda/rank aggregation across seeds, then require stability by bootstrap or overlap threshold.  
   This is a method improvement, not drift.

2. **Use a simple signed linear probe on cached activations as a diagnostic companion to direction extraction.**  
   Not as the main claim, but as a cheap stability readout on whether the candidate direction separates flipped-wrong from matched-agree out of sample. Very standard and low overhead.

3. **For dose-response, report isotonic-fit deviation in addition to Spearman trend.**  
   This is a more modern way to quantify near-monotone intervention curves without overcommitting to linearity.

---

## Drift Warning

**NONE**

The frozen claim is preserved. The remaining concerns are about protocol tightening and consistency, not claim drift.

---

## Verdict

**REVISE**

Very solid improvement and close to READY, but not there yet under a strict standard. The main blockers are:
- inconsistent seed/failure semantics,
- slight over-strengthening of M0 via mean-threshold and audit entanglement,
- mechanism verdict hierarchy not yet cleanly pre-registered,
- feasibility section still a bit too optimistic without a tighter activation-caching plan.

Fix those, and this should move into READY territory.