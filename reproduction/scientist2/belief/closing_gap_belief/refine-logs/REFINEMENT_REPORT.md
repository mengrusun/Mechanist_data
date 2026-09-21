# Refinement Report

**Problem**: Determine whether Llama-3.1-8B-Instruct's near-100% verbalized-confidence overclaim on TriviaQA is a knowledge deficit or a readout failure, by measuring the geometric angle + causal cross-coupling between two internal linear probe directions.
**Initial Approach**: Two paired linear probes on residual-stream activations + cosine similarity + activation steering (given-behavior mode; C1/C2/C3 fixed by task.md).
**Date**: 2026-07-13
**Rounds**: 3 / 5
**Final Score**: 9.1 / 10
**Final Verdict**: READY

## Problem Anchor
(see `REVIEW_SUMMARY.md` §"Problem Anchor" — verbatim across all rounds)

## Output Files
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (written next by /experiment-plan)
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md` (written next by /experiment-plan)
- Round records: `round-0-initial-proposal.md`, `round-{1,2}-review-prompt.md`, `round-{1,2,3}-review.md`, `round-{1,2,3}-review-raw.md`, `round-{1,2}-refinement.md`, `score-history.md`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|-------|------------------|--------------------|----------------------|-------------------|-------------|------------------|-----------------|---------|---------|
| 1     | 8                | 8                  | 7                    | 8                 | 6           | 6                | 6               | 7.1     | REVISE  |
| 2     | 8.8              | 9.0                | 8.1                  | 8.4               | 8.8         | 8.4              | 8.0             | 8.5     | REVISE  |
| 3     | 9.2              | 9.2                | 9.3                  | 9.0               | 9.3         | 8.9              | 9.0             | 9.1     | READY   |

## Round-by-Round Review Record

| Round | Main Reviewer Concerns | What Was Changed | Result |
|-------|-------------------------|------------------|--------|
| 1     | Compute-heavy (36k steering passes); C3 identifiability (different forward contexts); mechanism-light framing | Trimmed steering grid → 4500 passes; pinned canonical hook; added emitted-output metric; C2 → continuous linear regression primary; pruned ablations to 6 | REVISE 7.1 |
| 2     | Framing overreach; C2 low-variance contingency; two-context caveat missing | Added Scope paragraph; added Stage 1.5 diagnostic; added single-pass robustness variant; collapsed to single L\*; added absolute-effect companion criterion | REVISE 8.5 |
| 3     | (No blocking) Polish: bootstrap protocol, L\* guardrail, C3b asymmetry, parse-failure diagnostic, single-pass precommit | Adopted retrain-on-bootstrap; added neighborhood-robustness guardrail; internal-readout primary / emitted-output secondary; parse-bias diagnostic; single-pass precommit | READY 9.1 |

## Final Proposal Snapshot
(Canonical clean version in `FINAL_PROPOSAL.md`)

- **Thesis (1 sentence)**: On Llama-3.1-8B-Instruct + TriviaQA, train two linear probes at a canonical residual-stream hook — one for gold correctness (v_c), one for verbalized confidence (v_v) — and measure their per-layer AUROC/Spearman/ECE with bootstrap CIs, cosine similarity at a single primary layer L\* with neighborhood-robustness, and causal cross-coupling under matched-magnitude activation steering evaluated on internal readouts (primary) and emitted outputs (secondary).
- **Dominant contribution**: matched-pair, same-representation-space, canonical-hook characterization.
- **Supporting contribution**: dissociation-when-disagree analysis (secondary).
- **Explicitly rejected**: new probes, SAE, formation tracing, fine-tuning, cross-dataset angle generalization.
- **Compute**: ~10 h GPU (6 main + 2 verify + 2 buffer) within HARD budget.
- **Central risk**: `probe_v_primary` might trivially read the pre-emission confidence commitment — handled as a *finding*, not a *failure*, since C3 can still hold geometrically.

## Method Evolution Highlights
1. **Round 1**: canonical hook pinned + steering budget cut 8× while preserving statistical power.
2. **Round 2**: Stage-1.5 pre-registered contingency for low-variance C2 signals + explicit two-context scope statement + single-pass unified-prompt robustness variant.
3. **Round 3**: retrain-on-bootstrap statistics + neighborhood-robustness narrative guardrail + parse-failure bias diagnostic + internal-readout-primary / emitted-output-secondary steering asymmetry.

## Pushback / Drift Log

| Round | Reviewer Said | Author Response | Outcome |
|-------|---------------|-----------------|---------|
| 1 | Add extra ablations to defend probe stability | Accepted 3 of 8 as must-run; moved 4 to appendix. Rationale: only claim-load-bearing controls should be must-run. | Accepted / trimmed |
| 2 | Overinterpret "readout failure" | Accepted; added Scope-of-Mechanism-Claim paragraph bounding interpretation to "linearly separable + weakly aligned + limited causal coupling in this specific setting." Claim predicates C1/C2/C3 unchanged. | Accepted / framing-only |
| 3 | Bootstrap protocol semantics | Accepted; adopted retrain-on-bootstrap for probe-fit variability, eval-only CIs for cheaper per-layer curves. | Accepted / dual-track |
| — | (no drift ever flagged) | — | — |

## Remaining Weaknesses

- **Interpretation is bounded**: the paper cannot claim to fully resolve the knowledge-deficit vs. readout-failure dichotomy. It can only claim geometric weak alignment + limited causal cross-coupling in this specific model + dataset + prompts + hook. This is a *scope* honesty, not a *weakness*.
- **C2 signal strength**: if verbalized confidence clusters heavily at 100 (which the anchor itself predicts), the ordinal path fires. This is pre-registered but does mean C2's primary evidence is signal-strength-dependent.
- **Cross-direction steering asymmetry**: emitted-confidence-under-v_v-steering is easier to shift than correctness-under-v_c-steering. Corroborative, not primary.

## Raw Reviewer Responses

<details>
<summary>Round 1 raw response</summary>

See `refine-logs/round-1-review-raw.md`.

</details>

<details>
<summary>Round 2 raw response</summary>

See `refine-logs/round-2-review-raw.md`.

</details>

<details>
<summary>Round 3 raw response</summary>

See `refine-logs/round-3-review-raw.md`.

</details>

## Next Steps

- Proceed to `/experiment-plan` for a full claim-driven experiment roadmap (this is invoked next by the wrapping `/research-refine-pipeline`).
- After the plan: `/mechanism-skills` (Workflow 1.25) → `/auto-experiment` (Workflow 1.5) → `/auto-verify` (Workflow 1.75) → `/auto-iteration-loop` (Workflow 2).
