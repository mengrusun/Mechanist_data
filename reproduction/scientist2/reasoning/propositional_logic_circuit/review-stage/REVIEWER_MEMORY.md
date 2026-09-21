# Reviewer Memory

Cross-iteration suspicion log. Append-only. Phase A of iteration 2+ prepends this file verbatim.

## Iteration 1 — Score: 7/10, Verdict: ready

- **New suspicions**:
  - **C1 minimality is under-supported** — failure metric comes from a 20/158 component sample, and sparsity only holds at the hard cap (0.150) rather than at a discovered optimum. The claim of "sparse" is a threshold-satisfying artifact, not evidence of a natural sparse minimum.
  - **C2 may be partly an analysis-artifact negative** due to restricting the modularity analysis to top-40 of 158 components. The failure signal is genuine (median_dominance 1.20 vs 2.0 target; p=1.0 for rule and answer), but the negative conclusion should be scoped to "under the tested top-40 subset" rather than a global anti-modularity claim.
  - **C3 sufficiency failure may reflect ablation/reinsertion pathology** rather than only true distributed computation. The intervention shows insufficiency, but cannot distinguish among: (a) generic task infrastructure lives outside shortlist, (b) many low-ranked task contributors, (c) nonlinear interactions, (d) resample-ablation distribution shift, (e) reinsertion-context mismatch. Interpretive gloss "the other 85% is task-general infrastructure" outruns the evidence.
  - **Cross-family robustness is shallow**: one additional model (Gemma-2-9B), reused result (M5 milestone recycled as verify variant, no fresh GPU run). One family is enough for cross-family recurrence but not a general law.
  - **Anchor-cell dependency**: the core finding is measured on `k3_chain2_natural` — while M4.stab tested 2 additional cells and confirmed the qualitative pattern, the sufficiency + necessity numbers themselves come primarily from the anchor cell.
  - **The paper must avoid overclaiming mechanism discovery**: current evidence supports a negative claim about shortlist sufficiency, not a definitive positive account of where computation resides.

- **Previous suspicions addressed?**: n/a (first iteration).

- **Unresolved (carried forward)**:
  - All six suspicions above are carried into the paper-writeup stage — they are paper-side caveats, not iteration back-edges (STOP fired iteration 1).
  - Two upstream Open Items remain: `/auto-verify C1 — resume: true` and `/auto-verify C2 — resume: true` would upgrade both from INTEGRITY_ONLY to full swap-tested. These are documented but not executed within the iteration loop budget.

- **Patterns**:
  - Reviewer flagged a systemic scope-vs-headline gap — three claims are all stated at the level of "the propositional-logic circuit" but the evidence in each case has a specific analysis scope (20/158 for C1 minimality; top-40 of 158 for C2; anchor cell k3_chain2_natural for C3 numbers). The paper's Discussion must not smuggle the scope back to the general claim.
  - Interpretive/mechanistic language ("distributed code", "task-general infrastructure") consistently outruns what the interventions can distinguish. The write-up should stay interventionally modest.
