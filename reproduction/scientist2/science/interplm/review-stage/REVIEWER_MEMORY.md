# Reviewer Memory

## Iteration 1 — Score: 4/10 (top-venue as-on-disk) / 6/10-ceiling (post-fix), Verdict: almost

- **New suspicions**:
  - c5a dead-code: `per_concept_pr_auc()` (LogisticRegression lbfgs, max_iter=200) is defined but never called; active code path uses `SGDClassifier(loss='log_loss', max_iter=30, alpha=1e-4)`. This is a major integrity concern — the reported null result is not testing the intended probe.
  - c4 metric is non-discriminative: 0 acceptance on both real (0/100) and control (0/50) arms. The strict synonym-check-null gate has zero acceptance rate — the criterion cannot discriminate. This is an "evaluation null", not a "negative result".
  - c5b steering ladder under-scaled: α∈{0.5,1,2,4} spans <1 OOM; claim expects monotone dose-response over ≥3 OOM. Within-sample σ_f normalization may suppress intervention magnitude further.
  - c1 LLM-gate sample too small for layer-wide extrapolation: 100/10240 features/layer with ±15% margin. Increase to 300-500 minimum.
  - c1/c2/c3 all under-powered on sequence count: 1500/10000 SP-test used. Likely main reason absolute counts lag reference.
  - Potential concept-granularity mismatch in c2/c3: 400 fine-grained concepts may depress absolute counts vs. reference. Track whether this is faithful or a hidden benchmark shift.
- **Previous suspicions addressed?**: n/a (first iteration)
- **Unresolved (carried forward)**: all of the above.
- **Patterns**:
  - Under-power is the common thread across c1/c2/c3 — same 1500/10000 SP-test subsample. A single "restore full 10k" main-experiment fix could uplift all three simultaneously (but each still counts as one type-② action for the claim it targets).
  - Two flavors of "not-supported" in this reproduction: (i) under-power (c1, c3) where the direction holds but absolute target missed — fixable with more compute; (ii) methodology-broken (c5a dead code) or metric-broken (c4 synonym-gate) where the number itself doesn't test the claim — fixable with a surgical code / criterion change.
  - c2 is the cleanest replicated claim (direction robust under 650M→8M swap) but ONLY the direction, not the absolute ~143 count. Paper must carry that caveat.

## Ranked ②-fix priority (reviewer's suggestion, for future iteration planning):
  1. c5a — fix dead-code path + score all concepts (~0.5-1.0 GPU-h)
  2. c1  — expand LLM-gate sample to 300-500 + 10k seqs (~1.0-1.5 GPU-h)
  3. c3  — 10k-seq rerun (~1.0-1.5 GPU-h, shares infra with c1/c2)
  4. c5b — wider dose ladder (~2.0-3.0 GPU-h)
  5. c4  — criterion redesign (~0.2-0.5 GPU-h, but risk of subjective metric)

Reviewer suggested total spend: 4.7-7.5 GPU-h (fits ~7.79 remaining). Recommends cutting c4 first if under-budget; then c5b.

## Iteration 2 — Score: 5/10 (top-venue as-is) / 6/10-ceiling, 7/10 workshop as-is / 8/10 ceiling. Verdict: almost

- **New suspicions**:
  - c5a is now correctly classified as **scope-limited negative** (not "under-powered null" — the correction landed). But 20/50 dropped concepts (zero test positives) create a selection-on-evaluable-concepts caveat. Fine for scoped reporting, not for broad generalization.
  - Bifurcation is now clear: (1) c5a = well-tested negative; (2) c1/c3 = likely under-powered near-misses; (3) c4/c5b = methodologically invalid current assays.
- **Previous suspicions addressed?**:
  - **c5a dead-code / wrong active probe path**: **RESOLVED** (genuine methodological correction; not merely cosmetic). Prior null was integrity-compromised; new null (SAE 0.5391 < neurons 0.5476, p=0.516) is a credible negative.
  - c1 under-power: still unresolved.
  - c3 under-power: still unresolved.
  - c4 non-discriminative metric: still unresolved and still serious.
  - c5b under-scaled dose ladder: still unresolved.
- **Unresolved (carried forward)**: c1, c3, c4, c5b (as above); scope caveat on c5a's negative.
- **Patterns**:
  - Paper is improving by replacing ambiguous nulls with more interpretable outcomes.
  - Highest-value remaining work is NOT chasing every claim — it is **separating true negatives from under-powered misses**.
  - c1 and c3 are the only remaining claims with realistic chance of upgrading the paper materially per unit budget.

Updated ②-fix priority (reviewer's rec for iter 2 given ~7.13 GPU-h left):
  1. c1  — expand LLM-gate 100→300-500 + more seqs (~1.0-1.5 GPU-h, high chance PARTIAL→SUPPORTED)
  2. c3  — 10k-seq rerun (~1.0-1.5 GPU-h, another near-miss)
  3. c5b — wider dose ladder (~2.0-3.0 GPU-h, uncertain payoff)
  4. c4  — criterion redesign (~0.2-0.5 GPU-h, low strategic priority, review-optics risk)
  Do NOT spend more on c5a.

Reviewer explicit routing plan:
  - 2 fixes → c1 + c3
  - 3 fixes → c1 + c3 + c5b
  - Stop after c1/c3 don't move if that's the case, and frame paper as "well-scoped partial reproduction with important negative correction (c5a) + c2 robust positive + c1/c3 near-misses".

## Iteration 3 — Score: 6/10 (top-venue as-is workshop track), Verdict: ready (workshop framing)

- **New suspicions**: none — reviewer explicitly says all critical concerns are resolved as of iter 2.
- **Previous suspicions addressed?**:
  - **c1 under-power hypothesis**: **FALSIFIED** in iter 2 (3× more features + 2× more sequences yields stable 14-15% pass rate; 95% CI on SAE_interp [1034, 1825] excludes target 2548). c1 is now a well-characterized partial (direction met, absolute not met).
  - **c5a**: still resolved (iter 1) as credible negative.
- **Unresolved (carried forward)**:
  - c3 could optionally get one more characterization run (~1.5 GPU-h) to become "well-characterized partial" instead of "partial directional support under current budget". Not required.
  - c5b remains protocol-weak; reviewer explicitly recommends NOT spending 3 GPU-h to rescue.
  - c4 remains assay-failure; not a compute issue.
- **Patterns**:
  - The reproduction is now honestly summarized as: strong replication of comparative gap (c2) + directional-only replication of scale claims (c1, c3) + failed-assay classification of c4 + credible negative for c5a + weakly-tested negative for c5b.
  - Scoring criterion switched: reviewer bumped +1 (5→6) NOT because a claim was rescued, but because c1 is now well-characterized (informative, not ambiguous). Iteration quality > iteration count.
  - Recommendation from reviewer: **stop and write**. Optional: one c3 run for polish. Do not spend on c5b.

## Termination trigger
Three-dimensional STOP rule satisfied at iter 3:
  - score = 6 >= TARGET_SCORE = 6 ✓
  - canonical verdict = "ready" (workshop framing per reviewer) ∈ POSITIVE_VERDICT_TERMS ✓
  - verify_failed / verify_inconclusive / verify_zero_eligible_variants all empty ✓
  - INTEGRITY_ONLY claims (c1, c3, c4, c5a, c5b) do NOT block STOP — they are no-action-with-upgrade-suggestion.
Terminating loop with termination_reason = positive_verdict.
