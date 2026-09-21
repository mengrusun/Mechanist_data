# Mechanism Audit Report — Claim C3

**Date**: 2026-07-22
**Auditor**: executor (no llm-chat reviewer call — early N/A exit for Check A; B-F reserved; additional mechanism checks performed manually per HARD CONSTRAINTS)
**Project**: Belief-Circuit Reproduction on Pythia
**Claim**: C3 — Formation Window (checkpoint-analysis with zero-ablation)
**Linked milestones**: M3

## Overall Verdict: N/A
*C3's mechanism-rigor verdict. N/A means Check A was not triggered — C3's intervention is zero-ablation (scale=0.0, same binary knockout as C2, not an additive steering intervention). B–F reserved/not_implemented.*

## Triggered checks (this run): (none — Check A not triggered)

## Checks

### A. Steering Coefficient Sweep: N/A
- Triggered: no
- C3 applies zero-ablation (scale=0.0, fixed binary knockout) of H*_personal and H*_attributed at each intermediate checkpoint — same mechanism as C2's zero-ablation, no tunable α scalar, no additive direction intervention. Check A trigger keywords absent from m3_formation.py.

### B–F. Reserved (not_implemented)

## Additional Mechanism Observations (HARD CONSTRAINTS — outside catalogue scope)

Per the orchestrator's HARD CONSTRAINTS, two C3-specific mechanism rigor criteria are noted:

1. **Same head indices used at all checkpoints (not re-derived per checkpoint)**: CONFIRMED — M3's implementation uses H*_personal and H*_attributed from M2 on pythia-1b step-143000, applied UNCHANGED at every earlier checkpoint. The `--hstar-personal` and `--hstar-attributed` arguments are fixed paths to the M2 output JSONs; they are NOT re-derived per checkpoint. This is methodologically correct per task.md and the Prakash et al. 2024 justification.

2. **Pre-registered emergence thresholds applied exactly as declared**: CONFIRMED — thresholds are defined in EXPERIMENT_PLAN.md (behavioral acc≥0.60 with 2-of-next-3 persistence; causal Δ≥0.20 with 2-of-next-3 persistence) and applied exactly in `m3_aggregate.py` BEFORE the sweep. formation/summary.json confirms the exact threshold values (behav_thresh=0.6, causal_thresh=0.2). No threshold tuning post-hoc.

## Action Items
None — N/A is appropriate; manual mechanism checks pass.
