# Reviewer Memory

## Iteration 1 — Score: 3/10, Verdict: not ready

- **New suspicions**:
  - Probe (M2) is the only clean positive; strong causal claim ("cached and retrieved") is not earned by present evidence.
  - Highest risk: authors overinterpret E4L10 probe R² despite M3/M4/M5 failing.
  - M5 is the immediate bottleneck — no capability metric, no locked α, no random-direction control, no text logs → non-diagnostic.
  - P2 main/control ratio failure (0.243 vs required ≥3) is severe, not cosmetic; aggregation appears inconsistent with per-seed values.
  - `answer_acc_preserved=1.0` is trivially true post-answer commit and must not be counted as evidence of capability preservation.
  - Cross-seed instability in M6c (seed42=3.57, seed123=0.10) may reflect noise/fragility rather than mechanism.
- **Previous suspicions addressed?**: n/a (first iteration)
- **Unresolved (carried forward)**: all six above.
- **Patterns**:
  - "Passing" predicates are trivially or non-diagnostically passing (answer_acc_preserved, coarse off_digit_rate as only "capability" proxy).
  - Effect sizes at cache site are on the same order as per-item baseline std → risk of publishing noise as mechanism.
  - Aggregation transparency issue: reported ratios don't match derivable per-seed numbers.
- **Watchlist for iteration 2**: does the repaired M5 show a capability-preserving effect that exceeds a random-direction baseline? If not, the strong claim remains unsupported even with clean methodology.


## Iteration 2 — Score: 4/10, Verdict: not ready

- **New suspicions**:
  - Biggest remaining risk: authors may repackage a failed causal claim as a negative-result paper without adding minimal evidence to make the null interpretation airtight.
  - Greedy-decoding nulls may leave a loophole unless authors also check logit/probability-level effects on score tokens (sub-argmax preferences).
  - "One good probe + rigorous causal null" story may be too narrow for top-tier unless it articulates a broader lesson about decodability-vs-causal-use across more than one analysis lens.
- **Previous suspicions addressed?** (per reviewer):
  - M5 non-diagnostic — RESOLVED (real fix, not cosmetic).
  - P2 aggregation opacity — RESOLVED (per-seed ratios exposed, aggregate recipe explicit).
  - answer_acc_preserved misuse — RESOLVED (correctly demoted as trivial-by-construction).
  - M6c missing seed2024 — RESOLVED administratively; variance problem acknowledged rather than hidden.
- **Unresolved (carried forward)**:
  - Original strong claim still not earned — even more clearly contradicted now.
  - Probe overinterpretation risk still the central scientific problem.
  - Mechanistic specificity remains weak (M5 null AND M3/M4 fail).
  - Cross-seed fragility/noise still present in M6c.
- **Patterns**:
  - Robust decodability without causal efficacy is the dominant pattern.
  - Most "passes" disappear once trivial/non-diagnostic criteria are removed.
  - Cross-seed variability and small effects suggest fragility, not crisp mechanism.
- **Watchlist for iteration 3**: does the logit-level M5-v3 experiment confirm the null persists at the probability level (not just at argmax)? If yes, mechanistic-dissociation story is airtight for M5. If no (i.e., a sub-argmax effect surfaces), the interpretation shifts and the paper story needs to be rethought.


## Iteration 3 — Score: 5/10, Verdict: not ready

- **New suspicions**:
  - Main remaining failure mode: site-selection narrowness. E4L10 alone may be a decodable-but-non-causal correlate while nearby top-ranked sites could be causally relevant.
  - Without a local site sweep, the dissociation story may be overfit to one chosen site.
- **Previous suspicions addressed?** (per reviewer):
  - M5 sub-argmax loophole — RESOLVED (M5-v3 confirms null at both readout levels).
- **Unresolved (carried forward)**:
  - Mechanistic specificity remains weak — until top-k sites also show null causal use.
  - Probe overinterpretation risk still there — one probe R^2 at one site.
  - Cross-seed fragility warning remains in M6c.
  - Evidence package still mismatched to broader story.
- **Patterns**:
  - Strong decodability continues to outstrip causal efficacy.
  - Most originally promising predicates collapse under stricter tests.
  - Current story is a NEGATIVE DISSOCIATION but still too narrow for top-tier without one more concrete generalization test.
- **Watchlist for iteration 4**: does the top-5-site M5-v3 sweep show that all 5 sites have negligible logit-level effect? If yes, dissociation generalizes and the paper is ready as a mechanistic-dissociation paper. If no (one site actually steers), interpretation must be rewritten.


## Iteration 4 — Score: 6/10, Verdict: almost

- **Resolved (per reviewer)**:
  - Site-selection narrowness: closed by top-5 M5-v3 sweep.
  - Prior requested experiment executed exactly, matched predicted pattern.
- **New suspicions**:
  - Distributed / multi-site causal use loophole: single-site nulls may miss joint control.
  - Overclaim risk if authors phrase result as global "not causally used" instead of bounded "not strongly controllable via single-site steering".
- **Unresolved patterns**:
  - E1L5's small sign-consistent effect (~0.14 units) suggests weak causal signal may exist though not practically meaningful.
  - M6c/P5 remains cross-seed unstable.
- **Watchlist for iteration 5**: does joint top-5 multi-site steering (α·u_s at ALL 5 sites simultaneously) also return near-null? If yes → 7/10 ready. If yes with substantial effect → "confidence is distributed, not localized" story.


## Iteration 5 — Score: 7/10, Verdict: READY

- **Reviewer honored iteration-4 commitment**: score bumped 6 → 7, verdict "almost" → "ready" upon successful execution of the joint top-5 multi-site steering experiment.
- **Resolved (all remaining loopholes closed)**:
  - Distributed multi-site loophole: joint top-5 experiment shows small distributed effect (span 0.32 max) but far below causal threshold.
  - Site-selection narrowness: resolved by top-5 sweep (iteration 3).
  - Sub-argmax loophole: resolved by M5-v3 (iteration 2).
  - M5 non-diagnostic: resolved by M5-v2 (iteration 1).
- **Unresolved but acceptable limitations** (per reviewer):
  - Joint steering shows a small real effect (span up to 0.32) — story is "weak distributed causal sensitivity", not absolute null.
  - P5 M6c cross-seed variability persists (acknowledged, not blocking).
  - Scope narrow: one model, one dataset, three seeds. (By design of /auto pipeline.)
- **Final suspicion preserved**: main residual risk is OVERCLAIMING. If the paper frames the result as "decodable but not strongly controllable under tested interventions", it is ready. If it frames as "confidence is not causally used", NOT justified.
- **Recommended paper framing**: bounded negative-result / dissociation paper. Title: "Decodable but Not Strongly Controllable: A Dissociation Between Confidence Readout and Single-Site Causal Steering in Gemma-3-27B."
- **Recommended action for closure**: type (0) narrative closure — no more experiments needed. C1 formally still INCONCLUSIVE on-disk; orchestrator should invoke /auto-verify C1 externally to formally close the state (the mechanism-audit gate now PASSES per this reviewer's certification).
