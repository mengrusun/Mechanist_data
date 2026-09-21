# Reviewer Memory

Persistent, append-only account of the external reviewer's cross-iteration suspicions. Prepended verbatim to the reviewer prompt at every iteration from #2 onward, so the reviewer can check whether prior concerns were genuinely addressed rather than sidestepped.

## Iteration 1 — Score: 6/10, Verdict: almost

- **New suspicions**:
  - **C1 wording risk**: the strict claim text ("with zero banana residue in filtered training channel") is stronger than the evidence supports; residue is 5/302 (1.7%) — at judge-noise floor, not literal residue-free. Paper must not silently rebrand `conditional` as "supported".
  - **C1 protocol-deviation disclosure**: the verdict-stage override (`conditional` label vs code-default `inconclusive` when residue>0) is acceptable ONLY if transparently disclosed as a manual scientific override in the paper, tied to the repeat-pass instability / judge-noise-floor interpretation.
  - **C2 overclaim risk from M1 → M2**: Grassmann-overlap signal at block 50 (ratio 4.87) is a real *representation-level* differential, but it is NOT mechanism identification on its own. The paper must not present M1's screen as if it settled the mechanism question.
  - **C2 random-ablation comparability**: mean Δ_random (−0.019pp) equals or exceeds mean Δ_ablate (−0.015pp) at block 50 — this is the specificity-failing red flag; must remain salient in the C2 narrative. The block-50 handle is NOT yet a selective causal knob.
  - **C2 framing**: the honest read is "evidence AGAINST simple single-block localization; indications of distributed carriage", NOT "we identified the causal component" and also NOT "mechanism experiment failed".
- **Previous suspicions addressed?**: n/a (first iteration)
- **Unresolved (carried forward)**:
  - Whether the narrative edits actually land in the paper draft (this iteration edited `CLAIMS_LEDGER.md` and `refine-logs/EXPERIMENT_RESULTS.md` only; the paper draft itself is not in scope of the pipeline).
  - Anchor-swap variant on C1 (strawberry instead of banana) is deferred; cross-anchor generality remains an open scientific question.
  - Multi-block simultaneous M2 intervention is deferred; the delocalized reading is scientifically consistent but not yet stress-tested at full power (28 remaining M2 runs + broader shortlist).
- **Patterns**:
  - Both claims came back INTEGRITY_ONLY under an over-budget verify stage — the loop's job here is narrative discipline, not new experiments. This is a policy-ready / substantively-almost-ready state that resolves by wording, not by GPU work.
  - Scientifically-legitimate negative-mechanism finding (C2) is easy to misread as failure; the paper narrative is the load-bearing element, not the data.
