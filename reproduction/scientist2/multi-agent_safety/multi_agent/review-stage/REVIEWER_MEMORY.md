# Reviewer Memory

Persistent suspicion log across auto-iteration-loop iterations. Append-only.

## Iteration 1 — Score: 4/10, Verdict: not ready

- **New suspicions**:
  - Judge-baseline fairness: any future comparison to text judges must address truncation/abstention explicitly (71% OTHER rate on 49 test rows is not a fair-baseline comparison; the "excl. OTHER" AUROC 0.806 shows the judge is competitive when it has text).
  - Post hoc layer cherry-picking: L27's 0.75 cannot be promoted as the main result unless selection logic changes prospectively (dev-selection rule chose L48 = worst test AUROC of the four probed layers).
  - Mismatch between aspirational benchmark size and realized benchmark size in claim wording: any wording must reference realized `used_n=282` and 16 realized domains, not the plan's 408 aspirational / 17 domains.
- **Previous suspicions addressed?**: n/a (first iteration)
- **Unresolved (carried forward)**:
  - Whether the probe is detecting collusion semantics vs. residual length/topic artifacts remains only partially ruled out (length-match 0.642, topic-swap 0.651 — both above chance, though below signal).
  - Whether group aggregation truly adds information beyond the strongest single agent remains unresolved; current evidence is weak (delta = +0.025 vs +0.05 target).
  - Whether short `max_new_tokens=60` depressed probe quality as well as judge quality remains unresolved but cannot be actioned under current budget (~2.59 GPU-h remaining; re-extraction would need ~4–5 GPU-h).
- **Patterns**:
  - The strongest result is transfer (C3), not the in-domain foundational existence (C1); be careful not to let downstream success overstate upstream evidence.
  - Budget pressure pushes the right decision toward scope narrowing (type-3 on C1) rather than another shaky empirical rescue (type-1 with a marginal-fit variant that likely wouldn't resolve the core fairness/truncation issue).
  - C2 and C3 are INTEGRITY_ONLY (no back-edge allowed); their paper text must foreground the un-stress-tested status, especially given both share C1's underpower root cause.

## Iteration 2 — Score: 6/10, Verdict: almost

- **New suspicions**:
  - The paper may still be vulnerable to "honest but too subtle" overreading if abstract/conclusion compress C1_v2 into stronger-sounding prose than the body supports.
  - "Competitive relative to a text-only judge" is acceptable under the rewrite, but still rhetorically delicate because many readers will mentally parse it as a strength comparison rather than a setting-qualified observation.
- **Previous suspicions addressed?** (audit of iteration-1 memory):
  - Judge-baseline fairness: **partially yes** at wording level, **no** at evidence level — 71% OTHER on 49 rows still means the main judge comparison is weak as evidence of method superiority.
  - Post hoc layer cherry-picking: **yes** at wording level; instability itself remains scientifically unresolved.
  - Aspirational vs realized benchmark mismatch: **yes**, genuinely fixed — C1_v2 wording anchors to the realized in-domain benchmark and restricted model/setup.
  - Semantics vs artifacts: **no**, still unresolved; no new evidence added but rewrite does not falsely claim semantic specificity.
  - Group aggregation incremental value: **no**, still unresolved (delta still only +0.025); C2 caveated / INTEGRITY_ONLY.
  - Token-limit depression of quality: **no**, still unresolved / budget-blocked; rewrite avoids overclaim under this regime, causally still unknown.
- **Unresolved (carried forward)**:
  - Residual possibility that probes exploit length/topic/style correlates rather than collusion semantics.
  - Weak evidence that group aggregation adds meaningfully beyond strongest single agent.
  - Short-generation setting may be distorting both text-judge and probe comparisons.
  - Small dev/test splits still make layer/effect-size conclusions unstable.
- **Patterns**:
  - The authors responded correctly by narrowing scope instead of forcing another underpowered experiment.
  - The project is now more defensible on integrity than on evidential strength.
  - Final acceptance odds hinge heavily on narrative restraint: if the paper markets this as preliminary-but-real evidence in a narrow regime, it is plausible; if it markets broad collusion detection superiority, reviewers will punish it.
