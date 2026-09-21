# Mechanism Audit Report — Claim C3

**Date**: 2026-07-14
**Auditor**: external LLM reviewer (gpt-5.4 via dmxapi, cross-model)
**Project**: Unified Verification of the Four-Claim Language-Agnostic/Specific Subspace Hypothesis on Qwen-3-4B-Thinking + MGSM
**Claim**: C3 — The steering coefficient α in h ← h + α·Π_lang·h produces MGSM accuracy monotone-decreasing in α over the signed sweep α ∈ [−1.5, +1.5] with A(−1) > A(0) > A(+1) (signed dose-response, negative correlation).
**Linked milestones**: M3

## Overall Verdict: WARN

## Triggered checks (this run): A (steering coefficient sweep)

## Checks

### A. Steering Coefficient Sweep: WARN
- Triggered: yes — same intervention as M2: `new_hidden = hidden + self.alpha * proj` (mlr/m2_projection_eval.py:95)
- Intervention type: projection-based steering (same as M2)
- Sweep grid: [−1.5, −1.0, −0.5, −0.25, 0, +0.25, +0.5, +1.0, +1.5] — 9 values, both signed, includes α=0 baseline ✓
- σ_proj scaling used: no — raw units
- Capability metric logged: GlotLID fidelity at all 9 α values (partial proxy; not independent perplexity/val-acc)
- Plateau range: N/A — no usable capability-preserving plateau; accuracy peaks at α=0 and drops both directions
- Locked α: no single locked α for C3 (the sweep IS C3's claim — all 9 values are the evidence)
- Alpha position: N/A — M3 IS the sweep, not a downstream use of a previously locked α
- Random-direction control: yes, run at ALL 9 α values with n=1 random subspace each (WARN: should be n≥30 at the key comparison points)
- Sign pattern: preserved — A(negative α) > A(positive α) at equal magnitude; V_lang collapses faster than random at positive α
- Output-case spot-check: α=0 sanity (macro_acc=0.744 ≈ independent baseline 0.762) confirms hooks inactive at α=0; α=+0.25 shows catastrophic collapse (0.009) vs random at +0.25 (0.729); V_lang direction is specifically disruptive
- Evidence: mlr/m2_projection_eval.py:79-95 (SteeringHooks), results/m3/vlang_alpha*_s42_summary.json (all 9), results/m3/random_alpha*_s42_summary.json (all 9)
- Verdict reason: WARN (not FAIL): sweep is genuine (9 values, both signs), includes α=0, random control at all grid points — these are positives. WARN triggers: (1) span < 3 orders of magnitude (|0.25| to |1.5|, factor 6); (2) no σ_proj scaling; (3) GlotLID fidelity is only a partial capability proxy, not a strict independent metric; (4) random baseline n=1 per α, not n≥30. Not FAIL because: genuine sweep performed, no single hardcoded α, random control exists at all grid points.

### B–F. Reserved (not_implemented)
Status: not yet implemented.

## Action Items
- [WARN-A] Express α grid in σ_proj units for cross-paper comparability.
- [WARN-A] Log a strictly independent capability metric (e.g., perplexity on English held-out text) at each α.
- [WARN-A] Run ≥30 random-subspace controls at the key comparison α values (e.g., −1.0, +0.25) for statistical validity.
