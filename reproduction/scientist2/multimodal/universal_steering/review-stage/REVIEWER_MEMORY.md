# Reviewer Memory

Persistent suspicion log across iterations. Append-only. Phase A of iteration 2+ prepends this file verbatim to the reviewer prompt so the reviewer can check whether prior suspicions were genuinely addressed.

---

## Iteration 1 — Score: 3/10, Verdict: not ready

- **New suspicions**:
  - Project appears overclaimed relative to what actually works — especially C1 and C3 are rhetorically broader than the data.
  - C5 advantage likely depends on RLHF-specific harmlessness representations, not general latent features across all 8B models. DeepSeek-R1 ToxicChat failure is probably a real boundary condition (reasoning-distilled vs RLHF-tuned).
  - C1 refusal pipeline may be fundamentally compromised by a degenerate Location-screen procedure (all 32 blocks tie at 1.0), not just bad luck.
  - C2 "C++ concept vector" may not represent language choice at all — steered outputs were all Python (never actually switched language).
  - C3 cross-lingual transfer likely fragile / language-specific; French reversal is not random noise.
  - C4 experiment is non-diagnostic due to single-vector ceiling saturation — missing random-direction control on combos is additional sloppiness.
- **Previous suspicions addressed?**: n/a (first iteration)
- **Unresolved (carried forward)**: all of the above.
- **Patterns**:
  - Scope inflation: multiple claims worded broader than the underlying evidence supports (C1 bundles refusal + political + honesty when only political works; C3 says "transfers to ZH/FR/ES" when FR reverses sign; C4 says compositionality without a ceiling-free test bed).
  - Missing controls / methodology gaps: alpha selection on the same held-out set (C1), no matched-random-direction control on combos (C4), no Bonferroni correction across languages (C3), suspected under-power (C2).

---

## Iteration 2 — Score: 4/10, Verdict: not ready

- **New suspicions**:
  - Authors may be using the loop contract as cover — INTEGRITY_ONLY is a policy skip, not a validity clearance. "Robustness = 2/3 > threshold 0.50" is bookkeeping, not scientific persuasion.
  - "Architecture-general" wording around C5 is still overstated; the regime dependence dominates the toxicity result.
- **Previous suspicions addressed?**:
  - C5-depends-on-RLHF: YES, empirically confirmed via 3-variant swap (RLHF passes, reasoning-distilled fails). Real evidence, credible. But paper wording must formally narrow, not just add a footnote.
  - C1-refusal-degenerate: NO (not addressed this iteration by design — outside loop scope).
  - C2-C++-vector-may-not-be-language: NO (not addressed).
  - C3-French-reversal: NO (not addressed).
  - C4-ceiling-null-diagnostic: NO (not addressed).
- **Unresolved (carried forward)**:
  - C1 refusal likely broken / degenerate block-localization
  - C2 concept vector may not represent C++ at all
  - C3 cross-lingual contradicted by FR + weak stats
  - C4 compositionality untested due to ceiling
  - C5 acceptable only under formal narrowed claim
- **Patterns**:
  - When challenged, authors add targeted variants (good). But they still preserve maximal claim surface area instead of pruning.
  - Recurring pattern: process compliance being used as a proxy for scientific adequacy.
  - **Explicit reviewer requirement**: for READY / Almost, C1-C4 must be **demoted from headline contribution status** to exploratory / preliminary. Optional standalone `/auto-verify` remediation can supplement the demotion but does not substitute for it.

---

## Iteration 3 — Score: 6/10, Verdict: almost (STOP fired)

- **New suspicions**:
  - Residual rhetorical inflation around C5 ("robustly," "spanning both regimes") may still be broader-than-warranted in tone; not fatal but should be scrubbed manuscript-wide.
- **Previous suspicions addressed?**:
  - C5-depends-on-RLHF: YES — C5_v2 rewrite is responsive and mostly fixes overclaim. Benchmark separation, boundary condition, and training-regime dependence all explicit.
  - C1-C4 headline-status: YES — demotion to exploratory / preliminary / null / non-diagnostic case studies with failure modes named. Adequate contingent on consistent manuscript tone.
- **Unresolved (carried forward)**:
  - Optional (post-loop) manuscript-wide wording harmonization pass — scrub residual "robustly / general / architecture-agnostic" phrasing from abstract / intro / conclusion. Not blocking for READY under the loop's contract.
  - Standalone C1_exp / C2_exp / C3_exp / C4_exp `/auto-verify — resume: true` upgrades remain available for post-loop remediation if desired.
- **Patterns**:
  - Now scientifically defensible in core framing.
  - Remaining risk is subtle scope creep through presentation, not claim substance.
- **STOP conditions met (three-dimensional)**:
  1. Score 6/10 ≥ TARGET_SCORE=6 ✓
  2. Verdict "almost" ∈ POSITIVE_VERDICT_TERMS ✓
  3. verify_failed = verify_inconclusive = verify_zero_eligible_variants = empty ✓ (C5_v2 PASS; C1_exp-C4_exp INTEGRITY_ONLY does not block STOP per contract)
