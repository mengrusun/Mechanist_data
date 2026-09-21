# Mechanism Audit Report — Claim C2

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, GPT-5.4 via DMX, T=0)
**Project**: Linear Steering of Reasoning Behaviours in DeepSeek-R1-Distill
**Claim**: C2 — Each behaviour direction is extractable from a small pool (n_pairs ≤ 200).
**Linked milestones**: M2 (also M1 for reference)

## Overall Verdict: WARN
*C2's mechanism-rigor verdict — the M2 stability sweep applies an additive steering hook at the fixed M3/M1-derived operating α for steering-effect ratio measurement.*

## Triggered checks (this run): A

## Checks

### A. Steering Coefficient Sweep: WARN
- Triggered: yes — M2 applies the mean-difference direction at a fixed α (the M3 operating α, or a mid-grid α) to a 30-task subset to compute steering-effect ratio. The α used is reused from M3 / M1; there is no fresh α sweep in M2.
- Intervention type: CAA (mean-difference additive steering)
- Sweep grid: M2 does NOT perform its own α sweep — it reuses the M3 sweep result (α_op = 0.5σ for uncertainty). This is by design (M2 tests direction stability at a fixed α, not the α itself).
- σ_proj scaling used: yes (direction stored with sigma_proj from M1; M3 sweep uses coef = alpha * sigma_proj)
- Capability metric logged: coherence_rate logged at each M3 α point; for M2's fixed-α steering-effect subset, coherence is logged via the same LLM judge.
- Plateau range: not established within M2 (M2 borrows M3's α_op); M3 shows coherence drops to 0.65 at α=±2σ, nominally clean at α≤1σ.
- Locked α: 0.5σ (from M3; used in M2 for ratio measurement)
- Position: edge — the M3 evidence shows that at α=0.5σ, the on-target Δrate (+0.017) is within noise at n=30–60, and the plateau is not robustly established.
- Random-direction control: no
- Sign pattern: preserved for uncertainty at α_op=0.5σ; not tested in M2 directly.
- Output-case spot-check: no dedicated output case logging for M2's steering-effect subset; the preview_chains in M3 show coherent chains at small α.
- Evidence: src/run_M2_smallpool.py (steering-effect calls borrow M3 hook logic), runs/M2_smallpool/results_summary.json (ratio_to_large_pool fields), runs/M3_steer/part_A/results_summary.json (analysis.expressing_uncertainty.operating_alpha = 0.5)
- Verdict reason: WARN because M2 uses a borrowed α_op that is not robustly established (effect at 0.5σ is within noise at n=30–60, no random-direction control, α not in mid-plateau of a confirmed dose-response curve). Not FAIL because the sweep did exist in M3, σ_proj scaling is used, and coherence is logged.

### B–F. Reserved (not_implemented)
Future checks will cover direction-extraction quality, site / layer selection, n_effective sufficiency, probe-vs-causal disentanglement, intervention scope.

## Action Items
- To upgrade to PASS: establish a clean dose-response plateau in M3 (requires n ≥ 200+ tasks for adequate statistical power), then use the mid-plateau α in M2's steering-effect ratio.
- Add a random-direction control run at the locked α to verify that the measured effect is direction-specific.
