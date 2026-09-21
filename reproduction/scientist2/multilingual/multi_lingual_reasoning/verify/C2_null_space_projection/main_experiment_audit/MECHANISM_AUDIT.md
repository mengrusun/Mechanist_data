# Mechanism Audit Report — Claim C2

**Date**: 2026-07-14
**Auditor**: external LLM reviewer (gpt-5.4 via dmxapi, cross-model)
**Project**: Unified Verification of the Four-Claim Language-Agnostic/Specific Subspace Hypothesis on Qwen-3-4B-Thinking + MGSM
**Claim**: C2 — Suppressing the language-specific subspace at inference time via null-space projection at non-upper layers raises MGSM mean accuracy by ≥ 3 pp across 11 target languages, with GlotLID output-language fidelity drop ≤ 5 pp when the top-k layers are left intact.
**Linked milestones**: M2

## Overall Verdict: FAIL

## Triggered checks (this run): A (steering coefficient sweep)

## Checks

### A. Steering Coefficient Sweep: FAIL
- Triggered: yes — `mlr/m2_projection_eval.py` line 95: `new_hidden = hidden + self.alpha * proj` (h ← h + α · Π_lang · h, α=−1.0)
- Intervention type: projection-based steering (null-space projection)
- Sweep grid: [−1.0] only — single hardcoded value. k_top={4,8,12} and rank_r={2,8} swept but NOT α.
- σ_proj scaling used: no — α in raw mathematical units (α=−1 = exact null-space projection)
- Capability metric logged: no — GlotLID fidelity is logged but appears 0.0 for ALL conditions including baseline, so it is not a usable independent capability/coherence metric
- Plateau range: N/A — no α sweep performed; signed sweep deferred to M3
- Locked α: −1.0 (position: no plateau — mathematically motivated, not empirically selected)
- Random-direction control: yes, run ONCE (n=1) at α=−1.0 with rank_r=11, k_top=12, macro_acc=0.4291. Required n_random ≥ 30; not met.
- Sign pattern: not assessable — only negative α tested
- Output-case spot-check: α=−1 produces catastrophic capability collapse (macro_acc 0.065–0.029 vs baseline 0.762); the chosen α is in the collapse range, not a preserved-capability operating point.
- Evidence: m2_projection_eval.py:81-95 (SteeringHooks init + hook_fn), m2_projection_eval.py:146 (--alpha default=-1.0), results/m2/*_summary.json
- Verdict reason: Single hardcoded α=−1 (no sweep); no usable capability/coherence metric; α chosen at collapse range (69-73 pp below baseline); random control only n=1 of required ≥30. Three independent FAIL criteria hit.

### B–F. Reserved (not_implemented)
Status: not yet implemented. Future checks may cover direction-extraction quality, site / layer selection, n_effective sufficiency, probe-vs-causal disentanglement, intervention scope.

## Action Items
- [FAIL-A] Run a signed α sweep (e.g., α ∈ {−1.5, −1, −0.5, −0.25, 0, +0.25, +0.5, +1, +1.5}) at the M2 winning config (mid, k_top=12, rank_r=2) before claiming M2's coefficient is appropriately calibrated. (Note: M3 was designed to do exactly this — the M2 mechanism audit failure is essentially a staging issue: M3's α-sweep should be treated as part of M2's mechanism rigor, not as a separate claim.)
- [FAIL-A] Log a capability/coherence metric independent of MGSM accuracy (e.g., perplexity on held-out English text) at each α value.
- [FAIL-A] Run ≥30 random-direction controls at the selected α.
