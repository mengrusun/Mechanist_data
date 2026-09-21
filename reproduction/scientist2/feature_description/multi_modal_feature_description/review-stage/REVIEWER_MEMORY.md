# Reviewer Memory

Persistent, append-only reviewer suspicions across iterations of `/auto-iteration-loop`.
Iteration 1's Phase A prompt reads this file back verbatim (from iteration 2+), so the reviewer can check whether prior-iteration suspicions were genuinely addressed or sidestepped.

---

## Iteration 1 — Score: 6/10, Verdict: almost

- **New suspicions**:
  - Broad claim wording is not supported; persistent risk of **scope overstatement** ("every component c in a trained vision model" is only evidenced on ResNet-50/ImageNet).
  - C2 is strong numerically, but robustness evidence is **minimal (1 verified variant only — method-swap-crp-compose)**; watch for overclaiming swap-robustness.
  - C1 remains **integrity-only**; do not let later drafts imply swap-robustness or stronger verification than was actually performed. Stage-2 was cap-skipped (`max_verify_claims_cap`), not disqualified.
  - C1/P1c does **not** support strict monotone-nondecreasing over the full k-range; only a **small-k plateau/peak (k ≤ 16)** interpretation is defensible. Δ_sep degrades at k∈{64, 256}.
  - fc-layer P2c is a recurring interpretability risk; treat as **heuristic-confounded / inconclusive** under CLIP-text-neighbor grouping — not as contradictory evidence, but not as clean evidence either. Layer4 P2c (d=1.96) is where the semantic-grouping claim genuinely tests.
  - Hidden-layer C1/P1b effects, especially layer3 (Δ_sep = 0.00492), are statistically significant but **small in magnitude**; avoid inflated practical interpretation.
  - "matched-control" terminology in C1/P1b docstring is ambiguous vs. the actual top-1/top-2 gap operationalization; needs textual cleanup.
- **Previous suspicions addressed?**: n/a (first iteration)
- **Unresolved (carried forward)**: all of the above — these are paper-presentation risks that must survive into the final write-up if the ⓪ narrative-only re-scoping edits do not land.
- **Patterns**: keep distinction sharp between
  - strong evidence for **ResNet-50 / ImageNet component indexing** (what was actually tested), and
  - unsupported claims about **general trained vision models** (what the frozen wording asserts).
  The claim text is frozen — the paper's *presentation* must adopt the narrower scope in prose even if the frozen wording is preserved for provenance.
