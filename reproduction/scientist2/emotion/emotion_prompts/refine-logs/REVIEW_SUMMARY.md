# REVIEW SUMMARY — internal review notes on the C1–C4 + CM plan

**Behavior-source**: given (from task.md — no external LLM review of the *claims*; reviewer scrutiny is on the *testing method*)
**Mechanism**: discovery
**Date**: 2026-07-13

Because `BEHAVIOR_SOURCE = given`, the claim text is fixed and not up for review (`/auto-claim` Phase 4 external review is skipped). The internal review here focuses on the **testing method** built around those fixed claims — the six concerns the plan must address (see FINAL_PROPOSAL.md § Reviewer concerns still live) and how the plan resolves each.

## Concerns and mitigations (rolled up)

1. **Confounds** — length, token frequency, position, format. *Mitigation*: length-matched non-emotional filler condition (M1 → M2), token-frequency-stratified probe controls in M5, and the M2b format-perturbation noise floor as the C1 threshold.
2. **CoT decoding stochasticity** — could inflate per-item Δ noise. *Mitigation*: T=0.0 greedy across all conditions; per-item logs archived.
3. **Prefix caching contamination** — vLLM's KV cache could carry state across conditions. *Mitigation*: prefix caching explicitly disabled between conditions in the `run_prefix_eval.py` runner config.
4. **Answer parsing failure** — GSM8K CoT parses may fail sporadically. *Mitigation*: pairwise-drop items where either the neutral or the emotional run failed to parse.
5. **EmotionRL train/test leakage** — must not overlap M2 test items. *Mitigation*: strictly train-partition GSM8K for M7a (first 2000 train items) and hold out the 500 M2 test items for M7d.
6. **Mechanism specificity** — a M6 positive without the filler + off-target controls does not verify CM. *Mitigation*: filler-control run (M6.3) and off-target MedQA run (M6.4) are gates on CM.
7. **Budget realism** — plan sums to ~6.5 GPU-h with ~3.5 h headroom; sensitivity is in M7a (dominates) — it can be chunked and re-launched if wall time is tight.

## Rejected complexity (recorded, not to be re-litigated)

- Weight-space editing / task-vector tuning — off-strategy (see mechanism_strategy.rejected).
- Formation-tracing / influence functions — off-strategy and off-budget.
- Free-form policy for EmotionRL — the 13-way discrete space is required by the task.md prompt-template spec.
- Cross-model comparison in the main experiment — reserved for `/auto-verify` (Workflow 1.75).

## Verdict

**READY** — the testing method covers each captured claim with a measurable predicate, controls, budget, and success gates; the mechanism plan is at the correct altitude for the `MECHANISM=discovery` charter (kind of component, not exact identity).
