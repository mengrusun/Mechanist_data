# Auto Iteration Loop — Adversarial Review Log

**Project**: Faithful SAGE Reproduction — iterative, activation-grounded SAE-feature explanation vs. Neuronpedia
**Reviewer model**: gpt-5.4 via DMXAPI (`https://www.dmxapi.cn/v1`)
**Reviewer-config source**: shell env
**Budgets**: `MAX_ITERATIONS=6`, `MAX_CLAIM_REENTRIES=2`, `TARGET_SCORE=6`
**GPU_ID pin**: `1,2,3,5,6` (allowlist)
**Auto-proceed**: true
**Resume**: false (fresh run)
**Start**: 2026-07-14T03:29Z

---

## Iteration 1 (2026-07-14T03:29Z)

### Assessment (Summary)
- **Score**: 6/10
- **Verdict** (canonical): almost
- **Budget after this iteration**: iterations 0/6, claim-reentries 0/2 (Phase C ran only type-⓪ narrative-only actions; iteration counter NOT incremented)
- **Key criticisms**:
  - Central "positive story" (C1 SAGE > Neuronpedia gen-acc) does not survive matched-backbone GPT-5-1shot control (Δ=-0.005 n.s.) — this must be the headline framing.
  - Main results underpowered (realized n = 15-23 % of planned).
  - Only C4 got Stage 2 verification; C1/C2/C3 are integrity-only.
  - Paper-side language must not overclaim: C4 "PASS" is a robust *null*, not a cross-pair success; direction of the two cross-pair deltas differs (+0.153 vs -0.045).
  - Attribution-confound is the story; must be elevated, not buried.
- **STOP-rule check (three-dimensional)**:
  1. score (6) ≥ TARGET_SCORE (6) ✓
  2. canonical verdict "almost" ∈ POSITIVE_VERDICT_TERMS ✓
  3. `verify_failed` / `verify_inconclusive` / `verify_zero_eligible_variants` all empty ✓
- **STOP fires — termination reason: `positive_verdict`.**

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

## Score: 6/10

Competent, honest, and scientifically useful reproduction. The score is held back because the central positive story does not survive the matched-backbone control, the main results are underpowered, and only one claim got Stage 2 verification. But the work does something many papers do not: it finds and reports the attribution confound instead of laundering it into a win. That is real value.

If submitted with the current honest framing, this is a respectable negative/clarifying reproduction. If submitted with any "SAGE works" spin, it would be misleading.

---

## 1) Top-level assessment

The strongest contribution here is **not** "we reproduced SAGE's gains." It is:

- the apparent C1 gain over public Neuronpedia is **largely explained by backbone substitution**,
- C2 does **not** replicate,
- C3 is **unresolved due to power**,
- C4 is a **robust negative**: no significant cross-pair advantage on either tested pair.

That is a publishable reproduction story if written cleanly.

What weakens the package:
- realized sample sizes are far below plan,
- the only verified claim is the cross-pair negative,
- C1/C2/C3 are still integrity-only,
- the "resource bottleneck" was API throughput rather than compute, so underpower is understandable operationally but does not magically stop reviewers from asking why conclusions are being drawn from n=44 instead of n=300.

---

## 2) FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS claims

None in these buckets.

So: **no 1./2./3. fixes required**.

That said, do not abuse this bookkeeping fact. The absence of FAIL/INCONCLUSIVE buckets does **not** mean the evidential situation is strong. It means the unresolved items are sitting in **INTEGRITY_ONLY**, which is procedurally okay but scientifically still incomplete.

---

## 3) PASS claim check: C4

### Verdict
**Narrative is basically honest if phrased carefully.**
The correct framing is:

> Across two cross-LLM+SAE pairs, SAGE-lite does not significantly outperform the Neuronpedia proxy on predictive accuracy; this null result appears robust to swapping the cross-pair variant.

That is defensible.

### Numeric consistency
Main reported cross-pair:
- Qwen3-4B + transcoder-hp, n=34
- Delta Pearson = +0.153
- CI = [-0.031, +0.341]
- p = 0.18
- not significant

Verify variant:
- GPT-OSS-20B + resid-post-aa, n=45
- Delta Pearson = -0.045
- n.s.

This is consistent with the summary:
- one mildly positive but noisy estimate,
- one mildly negative estimate,
- no significant benefit on either pair,
- therefore **no clear support** for cross-pair generalization.

### Caveats that must appear in the paper
1. **Do not write "robust across pairs" in a way that implies consistent effect direction.**
   The directions differ (+0.153 vs -0.045). What is robust is the **null verdict**, not the effect.
2. **Do not imply Stage 2 verification proves broad cross-pair failure.**
   It supports failure on these two tested pairs, under this SAGE-lite setup and this sample size.
3. **Call it a negative replication / negative extension result.**
   "PASS" here is internal ledger language. Paper language should say:
   - "We found no statistically significant cross-pair advantage on either evaluated pair."
4. **Be explicit that this is predictive-only SAGE-lite, not full SAGE.**
   Otherwise readers will reasonably object that C4 is not a full reproduction of the full four-role loop under all metrics.

### Bottom line on C4 framing
- **Honest if stated as robust no-clear-support / robust null**
- **Overclaiming if stated as cross-pair success, transfer, or robustness of SAGE performance**

---

## 4) INTEGRITY_ONLY claims: C1, C2, C3

Correct: no 1./2./3. fix proposals here.

### Should any be upgraded later?
Yes.

#### Highest-priority upgrade: **C1**
Reason:
- It is the only "positive" headline,
- it is already known to be attribution-confounded,
- reviewers will focus on it immediately,
- Stage 2 swap-tests would help close the argument and reduce attacks on implementation validity.

If you only upgrade one, upgrade **C1**.

#### Second priority: **C2**
Reason:
- It is a direct core claim and currently null,
- if verified null under swap-tests, that strengthens the paper a lot,
- a robust negative on predictive accuracy is important.

#### Lowest priority: **C3**
Reason:
- currently underpowered at n=14-15/depth,
- even with verification, the fundamental problem is sample size, not just integrity,
- unless you also rerun with more data or targeted cleanup, verification alone probably won't make this claim persuasive.

### Specific note on L20 l0 alignment
If C3 is ever upgraded, yes, use the **l0_71** swap for L20. Otherwise one obvious noise source remains and reviewers will rightly say the depth analysis is too messy to interpret.

---

## 5) Is this READY for submission?

**Almost.**

By your procedural criterion, it is technically READY because there are no FAIL/INCONCLUSIVE/ZERO_ELIGIBLE_VARIANTS claims.

By actual paper standard, I would call it **Almost, not fully ready**:
- ready **if** the submission is framed as an honest, mostly negative / attribution-confounded reproduction with limited verification,
- **not** ready if the authors want to sell a positive reproduction of SAGE's claimed advantages.

So the real answer is:

- **READY as a negative/clarifying reproduction**
- **NOT READY as a positive replication claim**

If forced into one word under your rubric: **Almost**.

---

## 6) What the final paper should say, bluntly

The paper-side headline should be something like:

> We reproduce the SAGE pipeline under a matched GPT-5 backbone and find that the apparent gain over the public Neuronpedia catalog in generative accuracy largely disappears under a matched-backbone control. We find no support for predictive-accuracy gains, insufficient evidence for layer-depth generalization due to underpower, and no significant cross-model/SAE transfer benefit across two tested pairs.

That is the honest result.

What it should **not** say:
- "We reproduced SAGE's superior interpretability performance."
- "SAGE generalizes across model/SAE pairs."
- "Multi-agent decomposition outperforms single-pass explanation."
- "SAGE beats Neuronpedia" without immediately qualifying "only against the public older-backbone catalog, not against a matched-backbone GPT-5 baseline."

---

## Main paper risks

1. **Backbone confound is the story.**
   If buried in appendix or phrased timidly, the paper becomes misleading.
2. **Underpower can't be hand-waved away.**
   n=44/300 and n=34/150 are not small deviations; they are major shortfalls.
3. **Matched-backbone control must be elevated to primary comparator.**
   Neuronpedia public catalog is a historical baseline, not the fairest causal comparator.
4. **Internal verification status should not be disguised.**
   C1/C2/C3 are not fully verified. Say so cleanly.
5. **Do not oversell "methodology gate passed."**
   Passing an internal methodology gate is not evidence of successful scientific replication.

---

## Minimal publication-strength framing adjustments

These are not "fix tags," just wording corrections you should make now:

- Recast C1 as:
  - **C1a:** SAGE exceeds the public Neuronpedia catalog built with an older backbone.
  - **C1b:** SAGE does **not** significantly exceed a matched-backbone GPT-5 single-pass baseline.
- Promote C1b over C1a in the abstract/conclusion.
- Describe C2 as a failed replication / null result.
- Describe C4 as a verified negative transfer result across two pairs.
- Describe C3 as unresolved due to power, not as evidence for absence of depth dependence.

---

## Final judgment

This is a **good reproduction** in the scientific sense and a **weak positive replication** in the headline sense.

That distinction matters.

The team did the right thing by surfacing:
- the attribution confound,
- the matched-backbone control,
- the null predictive result,
- the robust negative cross-pair result.

That honesty raises the value of the work. The current evidence does not support strong claims for SAGE's superiority beyond the public older-backbone baseline.

---

## Memory update

- Track whether authors try to keep **C1a** ("beats public Neuronpedia") as the main headline instead of **C1b** ("no gain vs matched-backbone GPT-5-1shot"). That would be the main overclaim.
- Watch for misuse of **"PASS"** on C4 as if it meant SAGE succeeded cross-pair. The only honest PASS interpretation is that the **null verdict** was robust.
- Track whether **underpower** is framed as a mere operational nuisance rather than a major inferential limitation.
- Watch if **C2 null** gets buried behind C1 language. It is central and should be prominent.
- If any further verification budget is spent, prioritize **C1 first, C2 second, C3 last**.
- If C3 is revisited, check whether **L20 l0_71** is actually swapped in; otherwise depth conclusions remain contaminated by avoidable mismatch.
- Future concern: ensure paper clearly distinguishes **full SAGE** from **SAGE-lite predictive-only** in C4; otherwise readers may infer stronger cross-pair conclusions than the experiment warrants.

</details>

### Verify-Passed Claims (brief audit)
- **C4** — Reviewer concurs the "no-clear-support across two cross-pairs" narrative is defensible and numerically consistent. Robustness=1.00 is the *null verdict being robust*, not SAGE succeeding. **Caveats to surface in paper**:
  1. Do not write "robust across pairs" implying consistent effect direction — the two cross-pair Δ Pearson values have opposite sign (Qwen3-4B: +0.153; GPT-OSS-20B: -0.045).
  2. This is **SAGE-lite predictive-only**, not full 4-role SAGE — must be stated every time C4 is mentioned.
  3. Paper wording should read "no statistically significant cross-pair advantage on either evaluated pair," not "PASS".

### Actions Taken (per claim, per type)

- **C1** — type ⓪ narrative-only — accept reviewer's framing correction: split into C1a (headline vs public Neuronpedia) + C1b (matched-backbone GPT-5-1shot); promote C1b in abstract/conclusion. **No script or data change; no iteration counter increment.**
- **C2** — type ⓪ narrative-only — reframe as "failed replication / null result" (already substantially the current wording; ensure it is prominent, not buried behind C1 language).
- **C3** — type ⓪ narrative-only — reframe as "unresolved due to under-power" (not as "evidence for absence of depth dependence"). If ever upgraded, use L20 l0_71 swap.
- **C4** — type ⓪ narrative-only — reframe as "verified negative transfer result across two tested pairs (null verdict robust; effect direction differs)"; SAGE-lite predictive-only scope must be stated every time.
- **INTEGRITY_ONLY upgrade recommendation** (paper-owner action, not a loop back-edge): priority order is C1 > C2 > C3; if only one is upgraded, C1 (highest reviewer-visibility; closes the attribution-confound argument). Budget-permitting (each ~1 GPU-hr), the human can dispatch `/auto-verify C1 — resume: true, swap-variants: true` and optionally `/auto-verify C2 — resume: true, swap-variants: true` before submission.

### Claim Rewrites (type ③ — no rewrite this iteration)
none — all reviewer-requested framing changes are paper-side wording (type ⓪); no ledger-level claim rewrite is warranted. The C1a/C1b split is a *presentation* structure, not a new claim id (the underlying two-comparator design is already in the results file).

### Claim-Stage Re-entries Triggered (orchestrator handoff — none this iteration)
none.

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C1** [stage2_skip_reason: max_verify_claims_cap; main_experiment_integrity: warn; warn_source: experiment (statistical resolution — Wilcoxon n_eff=4, coarse 5-probe gen_acc metric)]
  - Upgrade command: `/auto-verify C1 — resume: true, swap-variants: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2-3 run)
  - Reviewer priority: **highest** (attribution-confound headline claim; ~1 GPU-hr)
- **C2** [stage2_skip_reason: max_verify_claims_cap; main_experiment_integrity: pass]
  - Upgrade command: `/auto-verify C2 — resume: true, swap-variants: true`
  - Reviewer priority: second (would strengthen the robust null on predictive accuracy)
- **C3** [stage2_skip_reason: max_verify_claims_cap; main_experiment_integrity: warn; warn_source: experiment (Bonferroni framing not coded; per-layer n=14-15; L20 gen_acc all-zero deltas)]
  - Upgrade command: `/auto-verify C3 — resume: true, swap-variants: true` (best combined with a larger n_features_per_layer rerun of M1 and L20 l0_71 swap)
  - Reviewer priority: lowest (underlying issue is under-power, not integrity; verification alone won't fix the depth analysis)

### Results
- No `/run-experiment` calls fired this iteration (only ⓪ narrative-only actions).
- `[run-experiment] iteration=1 runs_this_iteration=0 gpu_hours_this_iteration=0.00 cumulative_gpu_hours=0.00`

### Status
- **completed** — STOP rule fired (score=6 ≥ 6, verdict="almost" ∈ {ready, almost}, no FAIL/INCONCLUSIVE/ZERO_ELIGIBLE_VARIANTS claims). Termination reason: `positive_verdict`. Proceeding to Termination writeup.
