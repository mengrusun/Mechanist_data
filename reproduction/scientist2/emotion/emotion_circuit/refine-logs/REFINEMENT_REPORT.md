# Refinement Report

**Problem**: Verify three claims from task.md about emotion-specific global circuits in Llama-3.2-3B on SEV.
**Initial Approach**: Unified matched-budget testing protocol.
**Date**: 2026-07-13
**Rounds**: 3 / 5
**Final Score**: 9.24 / 10
**Final Verdict**: READY

## Problem Anchor (verbatim)

See `FINAL_PROPOSAL.md` § Problem Anchor.

## Output Files
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|------------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| 1     | 8                | 6                  | 8                    | 7                 | 6           | 6                | 7               | 7.0     | REVISE  |
| 2     | 9                | 9                  | 8                    | 8                 | 8           | 9                | 8               | 8.4     | REVISE  |
| 3     | 9.4              | 9.3                | 9.2                  | 9.3               | 9.1         | 9.2              | 9.2             | 9.24    | READY   |

## Round-by-Round Review Record

| Round | Main Reviewer Concerns | What Was Changed | Result |
|-------|-------------------------|------------------|--------|
| 1     | Under-specified operators + un-operationalized matched budget + missing targeted specificity + feasibility risk. | Pinned every operator; N=9 matched budget; C_{e'} control; length audit; two-stage locator; deleted seed resampling; verify swap → Claim-3 only. | Resolved specificity + validation-focus, partially resolved feasibility. |
| 2     | Stage-B metric + Arm C direction construction unspecified; global `k` + Claim-2 rubric + Claim 2e status not explicit. | Stage-B = prefix log-prob under single-component enhancement (judge-free); Arm C direction fully frozen pre-eval-split; `k` global once; four-state Claim-2 rubric; Claim 2e secondary; rank fusion removed; judge audit compressed. | Resolved all blocking issues. |
| 3     | Quick verification. | — | READY, 9.24/10. |

## Final Proposal Snapshot

- Unified verification protocol runs Location → Causal Intervention → Applied Control on one `C_e` fit.
- Stage-B ranker = target-prefix log-prob gain under single-component enhancement at α2, mean over 30 val stems per emotion. Judge-free, deterministic.
- Ablation = mean-substitute from same-stem off-target emotion activations; enhancement = additive activation injection at α ∈ {0.5, 1.0, 2.0}.
- Judge = hidden-target 6-way forced choice (gpt-5.4), gated on 60-item human gold subset with per-emotion confusion matrix + swap ablation + classifier fallback.
- Arm C (single-direction steering) direction constructed on train fold (last-event-token mean-diff); CAA-convention injection at chosen L from shared top-3 shortlist; matched `N=9` val budget mirroring Arms A and B.
- Claim 2 four-state rubric (full / partial / causal-only / not-supported); Claim 2e secondary.

## Method Evolution Highlights

1. **Stage-B causal ranker fully pinned to prefix-logprob gain** — removes the largest residual DoF.
2. **Arm C direction construction frozen pre-eval-split with CAA convention** — makes the matched-budget comparison genuinely apples-to-apples.
3. **Judge audit + Claim 2 rubric + operator single-valuedness** — the protocol reads like a test plan, not a menu.

## Pushback / Drift Log

| Round | Reviewer Said | Author Response | Outcome |
|-------|---------------|-----------------|---------|
| 1     | "Add pairwise judge as primary." | Kept as *optional secondary*; primary remains hidden-target 6-way to preserve one scalar. | Rejected as primary (design choice); accepted as optional. |
| 2     | "Rank fusion between Stage A and Stage B is unnecessary." | Agreed — Stage A = shortlist, Stage B = sole ranker. | Simplified. |
| 2     | "Demote Claim 2e." | Agreed — labeled secondary, does not gate Claim 2 support. | Accepted. |
| 2     | "15%-length threshold — sensitivity check." | Reframed as reporting threshold, not decision threshold. | Accepted. |

## Remaining Weaknesses

- Verify-stage swap is Claim-3-only. Cross-model generality of the *mechanism* itself (`C_e` on Qwen) is not tested — intentional scope discipline given the 10-hour envelope and the fact that Qwen has a different architecture/head count. If cross-model mechanism generality becomes a downstream concern, it can be added in a follow-up round rather than expanding this protocol.
- Judge dependency: an unreliable external judge triggers a small classifier fallback; the fallback is not the primary story, but it does add one training step to the pipeline conditionally.

## Raw Reviewer Responses

<details>
<summary>Round 1 — 7/10 REVISE</summary>

Saved in `round-1-review.raw.md`; parsed action items in `round-1-review.md`.

</details>

<details>
<summary>Round 2 — 8.4/10 REVISE (but close)</summary>

Saved in `round-2-review.raw.md`; parsed action items in `round-2-review.md`.

</details>

<details>
<summary>Round 3 — 9.24/10 READY</summary>

Saved in `round-3-review.raw.md`.

</details>

## Next Steps
- READY → proceed to `/experiment-plan` for the full claim-driven experiment roadmap (this is the next phase of the caller pipeline; done automatically inside `/research-refine-pipeline`).
- After the plan: `/mechanism-skills` routing (called by `/auto-experiment` Phase 1.5) to bind `CHOSEN_FAMILY` pre-eval-split.
- Then `/auto-experiment` for implementation + deployment, `/auto-verify` for the Qwen swap on Claim 3, `/auto-iteration-loop` for review.
