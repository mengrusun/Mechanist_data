# Reviewer Memory

## Iteration 1 — Score: 4/10, Verdict: almost

- **New suspicions**:
  - Paper's contribution set is **overclaimed** relative to evidence — headline framing is too strong for what the data support.
  - C2 (task-family ordering) should not be presented as "mixed evidence" — it is a **failed hypothesis** (ordering reversed, only 1/3 pairwise orderings hold).
  - C4 is the only true **submission-readiness blocker**; recommend pure narrative demotion (untested/deferred), NOT a partial 100-item heuristic demo (which would look like reviewer-bait).
  - CM should be **narratively bifurcated** into (a) representation evidence [strong] and (b) causal evidence [weak/null]; avoid unified "causal mechanism" language.
  - C1 needs careful phrasing: emotion-mean level within noise floor, but individual-prefix level shows real directional shifts (8/24 out of [0.4, 0.6] band). Do NOT say "only noise-like effects."
  - C3b support is real but **marginal at threshold** (3/6 exactly meets ≥3 gate); must show exact CIs and label as threshold-level support.
- **Previous suspicions addressed?**: n/a (first iteration)
- **Unresolved (carried forward)**:
  - Eval-mode confound (CoT vs MCQ-LL) for C2 must be foregrounded as a limitation, not buried in an appendix.
  - CM Causal arm specificity controls (filler, off-target, off-layer) descoped; length-null 0.69 in variant.
  - C4 remains INCONCLUSIVE in a strict verify-report sense even after ⓪ narrative demotion — the paper text can defer it, but the CLAIMS_LEDGER still lists it as unresolved.
- **Patterns**:
  - Over-strong original claims + budget-constrained execution → the paper's most defensible reading is a **negative/qualified-result paper** on emotional-prefix effects with mechanism-encoding-only support. Reviewer prefers this recast over any partial GPU-spend rescue attempt.
  - When a compute fix is over-budget, ⓪ narrative honesty beats a token compute pass that risks looking like reviewer-bait.

## Iteration 2 — Score: 6/10, Verdict: almost

- **New suspicions**:
  - **Front-matter leakage risk**: title/abstract/introduction may still implicitly oversell despite the repaired ledger text — must be visibly modest.
  - Acceptance odds depend heavily on whether the paper is framed as **careful negative-result/falsification-style** rather than a broad positive claim about emotion-guided reasoning.
  - Novelty/impact ceiling: the paper may still read as "we tested a provocative framing and mostly found it does not robustly hold" — publishable only if analysis quality + honesty are unusually strong.
- **Previous suspicions addressed?**:
  - Overclaimed contribution set → **addressed** (recast as qualified/negative-result paper)
  - C2 should be failed hypothesis, not mixed evidence → **addressed**
  - CM should be bifurcated into representation vs causal evidence → **addressed**
  - C1 should not be described as only noise-like → **addressed** (with fresh sign-consistency computation providing genuinely new evidence)
  - C3b needs exact CIs and threshold-level labeling → **addressed** (CIs surfaced from existing analysis)
  - C4 should be narratively demoted, not baited with a tiny demo → **addressed at paper level** (but remains INCONCLUSIVE in strict ledger)
- **Unresolved (carried forward)**:
  - **C4 remains INCONCLUSIVE in strict ledger terms** — reviewer recommends option (C): treat C4 as removed from submission claim set for STOP purposes (grade as 5-claim paper). This is a formal scoping decision, not a compute fix. Loop cannot enact this on the reviewer's authority alone; C4 surfaces in Open Items of final report.
  - C2 eval-mode confound: still a substantive limitation, though now honestly surfaced.
  - CM causal specificity limitations: still present; acceptable only because causal claims were demoted.
- **Patterns**:
  - Materially closing the evidence-to-claim gap via ⓪ narrative fixes moved the score from 4/10 to 6/10 with zero GPU spend. Pattern: honest framing is worth ~2 score points when the underlying data support it.
  - Reviewer accepts the 5-claim paper framing (C1, C2, C3a, C3b, CM) as valid; C4 handled as future-work note. But this is a paper-framing decision, not a state change in the ledger — loop must treat C4 as still INCONCLUSIVE for STOP purposes.

## Iteration 3 — Score: 6/10, Verdict: almost (convergence check)

- **New suspicions**: none — nothing on disk changed since iter-2; reviewer confirms stable state.
- **Previous suspicions addressed?**: all previously addressable ones remain resolved (iter-2 status). C4 strict-ledger INCONCLUSIVE is un-resolvable within remaining ~0.79 GPU-h.
- **Unresolved (carried forward)**: C4 (budget-blocked); C2 eval-mode confound (acknowledged); CM causal specificity (acknowledged as demoted).
- **Patterns**:
  - Reviewer explicitly recommends "stop" — no productive back-edge action available within remaining budget.
  - Optional CM specificity controls (0.3-0.5 GPU-h) rejected: incrementally nice but unlikely to change score, risks becoming reviewer-bait polish.
  - Loop termination is appropriate. Paper-level: defensible "almost-ready" qualified submission. Auto-verify ledger: NOT READY due to C4 INCONCLUSIVE.
