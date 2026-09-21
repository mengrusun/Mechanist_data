# Mechanism Audit Report — Claim C2

**Date**: 2026-07-22
**Auditor**: executor (no llm-chat reviewer call — early N/A exit for Check A; B-F reserved; additional mechanism checks performed manually per HARD CONSTRAINTS)
**Project**: Belief-Circuit Reproduction on Pythia
**Claim**: C2 — Belief Heads Localization (Fisher-information-matrix + zero-ablation)
**Linked milestones**: M2.1, M2.2, M2.3, M2.4

## Overall Verdict: N/A
*C2's mechanism-rigor verdict. N/A means Check A (steering coefficient sweep) was not triggered — C2 uses zero-ablation (scale=0.0, a fixed parameter, not a tunable scalar α in the steering/CAA/RepE sense). Checks B–F are reserved/not_implemented.*

## Triggered checks (this run): (none — Check A not triggered)

## Checks

### A. Steering Coefficient Sweep: N/A
- Triggered: no
- Trigger evaluation: C2's mechanism is "Fisher-information-matrix + zero-ablation". The zero-ablation sets identified head outputs to exactly 0 (scale=0.0 in `install_head_scaling_hooks`). This is NOT an additive activation intervention with a tunable scalar α — it is a binary knockout (identity scale=1.0 or ablation scale=0.0). No steering vector, no CAA, no RepE, no direction is added. The `alpha` parameter does not appear in M2 scripts. Check A trigger keywords (steer, CAA, RepE, DAS, ROME, activation_patch) are absent from M2 scripts (m2_1_fisher.py, m2_2_masks.py, m2_3_search.py, m2_4_control.py). The code pattern `activations += alpha * v` does not appear. Early N/A exit taken.

### B–F. Reserved (not_implemented)
Status: not yet implemented.

## Additional Mechanism Observations (HARD CONSTRAINTS — outside catalogue scope)

Per the orchestrator's HARD CONSTRAINTS, three C2-specific mechanism rigor criteria are noted (these fall under reserved checks B–F, not Check A):

1. **Deterministic greedy-add/remove discipline**: CONFIRMED — M2.3's search_log confirms greedy-add stops at first {C2a, C2c, C2d}-passing set; greedy-remove iterates until no removal holds (fixed-point). The |S|≤30 cap is enforced. Implementation in `scripts/m2_3_search.py` is deterministic (no randomness, no threshold relaxation).

2. **Head-parameter aggregation over fused QKV+dense**: CONFIRMED — per-head Fisher is aggregated over `{W_Q^h, W_K^h, W_V^h}` (fused query_key_value matrix sliced per head via GPTNeoXAttention._split_heads convention) plus `W_O^h` (dense output projection head-column slice). The hook sanity check (`scripts/_test_hooks.py`) verified that α=0 knockout reduces target-task accuracy by ≥20pp while α=1 is exactly identity. Per-head slicing correctness validated.

3. **Exact 4-criteria evaluation**: CONFIRMED — acceptance JSON shows explicit per-criterion boolean evaluation (C2a, C2b, C2c, C2d all True for every localized pair). Thresholds applied verbatim from task.md: drop≥0.30, >mean+2σ, off-target/WK drop≤0.10, PPL≤1.05×. No threshold relaxation.

## Action Items
None — N/A is appropriate for zero-ablation (non-additive) mechanisms; manual mechanism checks all pass.
