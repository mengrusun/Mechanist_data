# REFINEMENT REPORT

**Date**: 2026-08-03
**Mode**: given-validation × discovery
**Anchor**: `task.md` (frozen); `idea-stage/IDEA_REPORT.md` claims C1 (behavior — cross-modal subliminal QA_I drop) and C2 (mechanism — some internal component causally mediates the drop).

## What was refined

1. **Testing protocol for C1 (M0 gate)** — turned task.md's 6-step recipe into an executable protocol with reviewer-tight details: LR sweep width, LoRA rank/α default, DDP + replicate GPU pattern (no `device_map="auto"`), judge concurrency and back-off, trivial-explanation checks (Ctrl-A range, judge audit, length control, deterministic re-eval), four-state verdict rule.
2. **Mechanism ladder for C2** — instantiated the `/mechanism-explore`-selected Location → Causal Intervention chain into concrete M1 / M2 / M3 milestones, each with (i) submethod pool for the routing stage to pick from, (ii) sufficiency + necessity + specificity in M2, (iii) `method_sensitive` tags on the fields that depend on which submethod `/mechanism-skills` binds.
3. **Ctrl-A vs Ctrl-B logic** — clarified why *both* controls are load-bearing (Qi et al. 2023 shows benign SFT alone erodes safety → Ctrl-B subtracts that drift; Ctrl-A is the un-touched reference).
4. **Risk register** — enumerated 5 named risks with mitigations aligned to prior-work regularities (fragility from Schrodi; LoRA-artifact prediction; judge stability).

## What was NOT refined (and why)

- **The two claims themselves** — this is given-validation; the claims are captured verbatim from `task.md` and any "clarification" that alters their scope, direction, threshold, or metric is forbidden.
- **The M0 threshold (3 pp) and seed count (≥ 3)** — set by `task.md`; softening or tightening either would break the given contract.
- **The models, datasets, judge, GPU pin** — all HARD constraints from `task.md`; they are transcribed, not chosen.
- **The mechanism claim's altitude** — kept at "*some* internal component" per `/mechanism-explore` guidance; committing to a specific layer / feature at claim time would pre-empt the discovery work.

## Deliverables

- `refine-logs/FINAL_PROPOSAL.md`
- `refine-logs/EXPERIMENT_PLAN.md`
- `refine-logs/EXPERIMENT_TRACKER.md`
- `refine-logs/REVIEW_SUMMARY.md`
- `refine-logs/REFINEMENT_REPORT.md`
- `refine-logs/PIPELINE_SUMMARY.md`
