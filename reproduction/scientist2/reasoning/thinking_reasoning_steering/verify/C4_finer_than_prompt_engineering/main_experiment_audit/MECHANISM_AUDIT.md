# Mechanism Audit Report — Claim C4

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, GPT-5.4 via DMX, T=0)
**Project**: Linear Steering of Reasoning Behaviours in DeepSeek-R1-Distill
**Claim**: C4 — Steering-vector control offers strictly more distinct (behaviour-rate, accuracy) operating points than NL-instruction prompt engineering and Thinking Intervention.
**Linked milestones**: M4 (primary), M3 (α operating range)

## Overall Verdict: WARN
*C4's mechanism-rigor verdict — M4 uses a fixed set of 4 α values reusing M3's operating region rather than a fresh α sweep.*

## Triggered checks (this run): A

## Checks

### A. Steering Coefficient Sweep: WARN
- Triggered: yes — M4 applies steering at α ∈ {-2, -1, +1, +2}σ_proj using the same CAA additive hook as M3
- Intervention type: CAA (same as M3)
- Sweep grid: M4 uses a fixed 4-point subset {-2, -1, +1, +2} × σ_proj — not a fresh α sweep. This is by design (M4's purpose is to compare controllers at the operating region, not re-sweep α). The full 7-point sweep was done in M3.
- σ_proj scaling used: yes (inherits from M3/M1)
- Capability metric logged: accuracy and coherence logged for every M4 controller instance
- Plateau range: inherited from M3 analysis; M3 did not establish a clean mid-plateau (see C3 MECHANISM_AUDIT.json). M4 reuses an under-validated operating region.
- Locked α: {-2, -1, +1, +2}σ — M4 tests all four, not a single locked point. The comparison is at these four points.
- Position: covers the coherent range (coherence ≥ 0.65 at ±2σ) but no mid-plateau was established in M3. M4's α=±1σ is likely the most stable point (coherence ≈ 0.98) and corresponds to the largest on-target effect with acceptable coherence.
- Random-direction control: NO — not in M3 (from which M4 inherits), not in M4
- Sign pattern: preserved for uncertainty (positive α amplifies, negative suppresses); not meaningful for other 3 behaviours (zero rate)
- Output-case spot-check: M4 preview chains at α=−1 and +1 show coherent outputs matching expected behaviour; at α=±2 some coherence degradation (0.63–0.80) visible.
- Evidence: src/run_M4_control_compare.py (steering logic inherits from M3), runs/M4_control_compare/results_summary.json, runs/M3_steer/part_A/results_summary.json (operating region)
- Verdict reason: WARN because (1) M4 inherits M3's under-validated operating region (no random-direction control, α not in clean mid-plateau); (2) however, M4's purpose is a multi-controller Pareto comparison, not a fresh dose-response sweep — the design intent of reusing M3's α range is reasonable and disclosed. Not FAIL because M4 does add capability metrics (accuracy), the 4-point sampling across the coherent region is appropriate for the Pareto comparison task, and the coherence threshold (≥ 0.9) is applied during M3's analysis.

### B–F. Reserved (not_implemented)
Future checks will cover direction-extraction quality, site / layer selection, n_effective sufficiency, probe-vs-causal disentanglement, intervention scope.

## Action Items
- The main mechanism concern for C4 is inherited from C3: without a cleaner dose-response plateau established in M3 (via random-direction control + more tasks), the claim that steering's granularity advantage is due to the direction rather than general perturbation remains unverified.
- For a future round: once M3 is upgraded (n=300+, random-direction control), M4 can be run with the same validated α range and the comparison becomes more trustworthy.
