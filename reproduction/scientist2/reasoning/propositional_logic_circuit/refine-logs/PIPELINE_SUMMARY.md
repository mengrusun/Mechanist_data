# Pipeline Summary

**Problem**: verify — for Mistral-7B on a synthetic propositional-logic template (facts + rules → truth-value query) — that (C1) a sparse subset of attention heads + MLP components implements the task, (C2) that subset decomposes into three modular sub-circuits (fact identification, rule application, answer projection), and (C3) those components are both necessary and sufficient via activation patching / causal mediation.
**Final Method Thesis**: verify all three sub-claims with a single unified activation-patching pipeline — attribution-patching cheap screen (C1) → path-patched necessity (C3) + reinsertion sufficiency (C3) + role-corruption dissociation (C2) — on Mistral-7B main + Gemma-2-9B verify, under 10 GPU-hours.
**Final Verdict**: READY
**Date**: 2026-07-14

## Final Deliverables
- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`

## Contribution Snapshot
- **Dominant contribution**: rigorous, budget-aware **reproduction + cross-family generalisation test** of the sparse-modular circuit hypothesis for propositional-logic reasoning on Mistral-7B (main) and Gemma-2-9B (verify).
- **Optional supporting contribution**: schema recurrence check on Gemma-2-27B (contingent).
- **Explicitly rejected complexity**: SAE / dictionary decomposition; training-time analysis; fine-tuning / editing; full ACDC iterative pruning; new prompt families beyond the parameterised template.

## Must-Prove Claims
- **C1** — Mistral-7B's propositional-logic-solving computation localises to a component set of size ≤ 15% of total, with completeness ≥ 0.9 and single-removal minimality drop ≥ 0.05.
- **C2** — that set partitions into three near-disjoint sub-circuits (`C_fact`, `C_rule`, `C_answer`) with per-component dominance ratio ≥ 2×, per-role dissociation score ≥ 0.1, null-shuffle p-value ≤ 0.01, and Jaccard stability ≥ 0.6 across cells.
- **C3** — Recovery ≥ 0.8 on `{logit_diff, prob_diff}` for both necessity (clean → corrupt patching) and sufficiency (reinsertion into an otherwise-corrupted forward pass), with matched-control specificity gap ≥ 0.6, robust across ≥ 5 resample seeds.
- **Cross-family recurrence** — the schema-level pattern (sparsity, 3-role, necessity+sufficiency) also holds on Gemma-2-9B at relaxed threshold 0.7.

## First Runs to Launch
1. **M0.dataset** — build synthetic propositional-logic template pool (CPU, ~5 min)
2. **M0.setup** — Mistral-7B sanity load + accuracy check on anchor cell (GPU 0, ~15 min)
3. **M1** — attribution-patching screen on Mistral-7B (GPU 0, ~1.2 h) → shortlist for M2/M3/M4

## Main Risks
- **Mistral-7B-v0.1 broken local symlink** — Mitigation: fallback to `Mistral-7B-Instruct-v0.1` (locally present) or HF re-download at M0.setup.
- **Gemma-2-27B not locally available** — Mitigation: contingent (M5.contingent) — main verify is Gemma-2-9B.
- **Metric-driven false positives** — Mitigation: report all three metrics (`logit_diff`, `prob_diff`, `KL`) for every intervention.
- **Bag-of-heuristics null (C2)** — Mitigation: label-shuffle null-hypothesis test built into M4.
- **Sufficiency-reinsertion instability at 7B** — Mitigation: resample from a large pool; report distribution over ≥ 5 seeds; std < 0.1 target.

## Next Action
- Proceed to `/auto-experiment` (Workflow 1.5) — routes mechanism family (`/mechanism-skills`) then implements + deploys M1–M6.
