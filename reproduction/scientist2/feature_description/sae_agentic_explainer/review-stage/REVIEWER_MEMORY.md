# Reviewer Memory

Cross-iteration reviewer suspicion log. Append-only: each `## Iteration N` block records the reviewer's own perspective at that iteration. Phase A of iteration N+1 prepends this file to the reviewer prompt.

---

## Iteration 1 — Score: 6/10, Verdict: almost

- **New suspicions**:
  - Authors may attempt to preserve **C1a** ("SAGE beats public Neuronpedia") as the paper headline instead of **C1b** ("SAGE does NOT beat matched-backbone GPT-5-1shot"). C1b is the honest scientific comparator — anything else is overclaim.
  - The C4 **"PASS"** verdict may be misused as if it meant SAGE succeeded on cross-pair. The only honest interpretation is that the *null verdict* was robust across two architecturally distinct pairs — and the two cross-pair Δ Pearson estimates even have opposite sign (+0.153 vs -0.045), so "robust across pairs" for the effect itself is unsupported.
  - **Under-power** may be minimized as an "operational nuisance" (DMXAPI throughput variance) rather than treated as a major inferential limitation. Realized n = 44 / 300 planned for M1, and n = 14-15 per depth for C3, are major shortfalls, not marginal ones.
  - **C2's null** could get buried behind C1 language in the paper narrative. C2 is a direct core claim (predictive accuracy on the main pair) and its null replication is central — must be prominent, not subordinated.
  - **L20 SAE l0 alignment**: candidate variant `l0_71` was downloaded but never swapped in during M1. If C3 is ever revisited, this must be corrected; otherwise depth conclusions carry an avoidable noise source.
  - **Full SAGE vs SAGE-lite** distinction: C4's PASS is on *SAGE-lite predictive-only*, not the full 4-role SAGE loop with both metrics. Every C4 mention must carry this scope qualifier.
- **Previous suspicions addressed?**: n/a (first iteration).
- **Unresolved (carried forward)**:
  - All of the above — the loop terminated at iteration 1 on positive verdict, so none of the paper-side framing corrections have been executed yet; they are recommendations to the human paper author.
  - INTEGRITY_ONLY upgrades of C1/C2/C3 remain deferred (out of the iteration loop's scope; up to the human to dispatch `/auto-verify <id> — resume: true, swap-variants: true` before submission).
- **Patterns**:
  - The pipeline's *internal* PASS/FAIL ledger language diverges from the *paper-side* honest headline. "PASS" (C4) here means "null verdict robust", not "SAGE succeeded". This gap between ledger vocabulary and scientific narrative is the highest-risk source of overclaim.
  - **Every** claim carries `suspected_under_power: true` — a systemic issue rooted in DMXAPI throughput, not per-claim methodology. Future reproductions of this direction need a higher-QPS backbone or locally-hosted alternative.
