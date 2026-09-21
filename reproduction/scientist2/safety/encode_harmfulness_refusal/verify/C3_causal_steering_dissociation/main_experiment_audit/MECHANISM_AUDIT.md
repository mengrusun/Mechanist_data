# Mechanism Audit Report — Claim C3

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP; model=gpt-5.4)
**Project**: encode_harmfulness_refusal
**Claim**: C3 — Additive steering dissociates the effects: h flips internal harmfulness readout with refusal unchanged; r flips refusal with harmfulness readout unchanged
**Linked milestones**: M3

## Overall Verdict: FAIL
*This is C3's mechanism-rigor verdict. FAIL means the interpretability mechanism was not tuned with the necessary controls for the steering coefficient sweep.*

## Triggered checks (this run): A

## Checks

### A. Steering Coefficient Sweep: FAIL
- Triggered: yes — via scripts/m3_claim3_steering.py (alpha parameter multiplied to direction: `hs[b, p, :] = hs[b, p, :] + alpha * direction`)
- Intervention type: CAA (contrastive activation addition / diff-mean additive hook)
- Sweep grid: [-2, -1, -0.5, 0, 0.5, 1, 2] (7 points)
- sigma_proj scaling used: NO — alpha expressed in raw direction-norm units (not sigma_proj units). sigma_proj_h=1.58, sigma_proj_r=1.96 are stored in directions.json but NOT used to express alpha. In sigma_proj units the grid covers approx [-1.27, 1.27] for h and [-1.02, 1.02] for r.
- Capability metric logged: mean_logp_completion + rep_rate (both logged at every sweep point) — YES, capability metric IS present
- Plateau identification: FAIL — h-readout is monotone across the full 7-point grid (no interior plateau visible); r-refusal effect appears only at alpha=+2 (threshold-like, not plateau)
- Locked alpha: 2.0 (the grid boundary — not selected as a mid-plateau point)
- Alpha position in plateau: EDGE (or outside, since no plateau was established for r)
- Random-direction control: run=true, n_random=1 (ONE matched-norm vector, not >=30)
- Sign pattern: preserved (positive alpha increases h-readout for h direction; preserved)
- Output-case spot-check: cases_available=false (only aggregate per-cell metrics in steering_metrics.json; no per-prompt completion text sampled at each alpha in audit-accessible logs)

**Failure rationale** (FAIL, not just WARN):
1. **No usable plateau established**: For the r direction, the refusal effect is essentially zero at alpha<=+1 and jumps at alpha=+2 — this is not a "plateau in the middle" but a threshold at the boundary. Selecting alpha=+2 as the operative coefficient places the intervention at exactly the largest tested alpha, which is the edge of the tested range, not a mid-plateau. The FAIL criterion "alpha chosen where target effect is within baseline-noise floor" applies at alpha<+2 for r, and the effective alpha for r's stated result IS the boundary point alpha=+2.
2. **Span < 3 orders of magnitude in sigma_proj units**: The alpha grid in sigma_proj units spans approximately [-1.27, +1.27] for h — less than a factor of 10 between the smallest non-zero alpha (0.32 sigma_proj) and the largest (1.27 sigma_proj). The required >=3 orders of magnitude is not met.
3. **Random-direction baseline n_random=1**: The stated minimum is n_random>=30 for a statistical comparison; running one random direction does not establish that the learned direction's effect statistically beats random.

Evidence: scripts/m3_claim3_steering.py lines 59 (alpha parameter), 322-327 (hook_block_idx assignment); results/m3/claim3_verdict.json (r_delta_at_pos1=0.0, r_delta_at_pos2=0.52); refine-logs/EXPERIMENT_RESULTS.md M3 section.

### B–F. Reserved (not_implemented)

## Action Items
- **Required before C3 can be PASS**: re-sweep alpha in sigma_proj units spanning >=3 orders of magnitude (e.g., [0.03, 0.1, 0.3, 1.0, 3.0] x sigma_proj); run n_random>=30 matched-norm directions and report statistical comparison; identify and lock alpha mid-plateau (where both target effect is stable and capability metrics are within tolerance).
- Consider running the secondary Llama Guard 3 8B judge on the 20% random subset as planned (--secondary_judge flag) to corroborate the string-match refusal signal.
