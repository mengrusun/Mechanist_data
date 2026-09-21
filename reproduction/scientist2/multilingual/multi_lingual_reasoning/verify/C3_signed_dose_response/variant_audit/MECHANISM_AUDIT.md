# Mechanism Audit Report — Claim C3 (Variant, RE-AUDIT)

**Date**: 2026-07-14 (re-audit after iteration ① variant fix)
**Auditor**: iteration-agent (per external LLM reviewer directive from Iteration 1)
**Project**: Unified Verification of the Four-Claim Language-Agnostic/Specific Subspace Hypothesis on Qwen-3-4B-Thinking + MGSM
**Claim**: C3 — The steering coefficient α in h ← h + α·Π_lang·h produces MGSM accuracy monotone-decreasing in α over the signed sweep α ∈ [−1.5, +1.5] with A(−1) > A(0) > A(+1) (signed dose-response, negative correlation).
**Variant**: model_swap_deepseek_r1_llama8b (DeepSeek-R1-Distill-Llama-8B replacing Qwen-3-4B-Thinking)
**Linked milestones**: M3 (variant)
**Supersedes**: prior FAIL verdict (see "Prior audit" below)

## Overall Verdict: WARN
*This is C3's variant-level mechanism-rigor verdict. Upgraded from FAIL → WARN after Iteration ① dispatched the missing random-subspace α-sweep.*

## Triggered checks (this run): A (Steering Coefficient Sweep)

## Checks

### A. Steering Coefficient Sweep: WARN
- Triggered: yes — `mlr/m2_projection_eval.py:95` (`new_hidden = hidden + self.alpha * proj`)
- Intervention type: steering (subspace projection: h ← h + α·Π_lang·h)
- Sweep grid: [-1.5, -1.0, -0.5, -0.25, 0.0, +0.25, +0.5, +1.0, +1.5] (9 points, includes α=0 sanity)
- σ_proj scaling used: no (α still in raw units) — **WARN** (not FAIL): the reader can be told that α is in raw units and that σ_proj-normalization is a future improvement; comparability across models is limited but the within-variant sign pattern is still interpretable.
- Capability metric logged: yes (GlotLID `macro_fidelity` at every α)
- Random-direction control: **NOW AVAILABLE** — 9-point random-subspace α-sweep completed (see `results/random_alpha*_summary.json`). This is the change that upgrades the audit from FAIL → WARN.
- Collapse-range α values: α ∈ {+0.5, +1.0, +1.5} land in the collapse regime for BOTH V_lang and random (macro_acc ≤ 0.01) — this is now recognized as generic OOD forcing, not V_lang-specific damage. **Non-collapse-window α ∈ [-1.5, +0.25]** is the interpretable regime for both curves.
- Sign pattern: asymmetric — V_lang is specifically damaging at negative α (0.36–0.46) while random preserves baseline (0.47–0.48); both collapse at α≥+0.5.
- Output-case spot-check: metric_text_consistent — coherent at α=0 and α=−0.25, garbled at α=+1.0 (matches macro_acc = 0.000).
- Evidence:
  - `mlr/m2_projection_eval.py:95`
  - `deploy.sh:95` (ALL_ALPHAS grid)
  - `verify/C3_signed_dose_response/variants/model_swap_deepseek_r1_llama8b/results/vlang_alpha*_summary.json`
  - **NEW**: `verify/C3_signed_dose_response/variants/model_swap_deepseek_r1_llama8b/results/random_alpha*_summary.json`

**Comparison at matched α (V_lang vs random, DeepSeek-R1-Distill-LLaMA-8B, non-collapse window)**:

| α | V_lang macro_acc | random macro_acc | Δ (V_lang − random) | Interpretation |
|---|---|---|---|---|
| −1.5 | 0.3564 | 0.4800 | **−0.124** | V_lang damages, random preserves |
| −1.0 | 0.3909 | 0.4745 | **−0.084** | V_lang damages, random preserves |
| −0.5 | 0.4582 | 0.4673 | −0.009 | approximately tied |
| −0.25 | 0.4636 | 0.4727 | −0.009 | approximately tied (peak on both) |
| 0.0 | 0.4473 | 0.4473 | 0.000 | sanity — identical (no intervention) |
| +0.25 | 0.2582 | 0.4564 | **−0.198** | V_lang collapses, random preserves |

At α ∈ {+0.5, +1.0, +1.5} both curves collapse (macro_acc ≈ 0) — generic OOD forcing, excluded from interpretation.

**Verdict reason (WARN)**: the sweep now has a random-subspace control that establishes specificity of V_lang: at |α|≥1 in the non-collapse window, V_lang costs 8–20 pp of macro_acc while random costs 0. The σ_proj scaling remains missing and the collapse-range α values remain in the reported grid (labeled but not excluded). Neither of those is a FAIL criterion once the random control is present. Recommended future upgrade: σ_proj-unit rescaling and trim reported grid to the non-collapse window.

### B–F. Reserved (not_implemented)
Status: not yet implemented.

## Substantive finding (variant)
- Both V_lang and random baseline at α=0 = 0.4473 (correct sanity — identical, hook infrastructure honest).
- Non-monotone V_lang curve: A(−1.5)=0.356 < A(−1)=0.391 < A(−0.5)=0.458 < A(−0.25)=0.464 > A(0)=0.447 > A(+0.25)=0.258 > A(+0.5)=0.000. Peak at α=−0.25, mild U-shape for negative α, catastrophic collapse for positive α.
- **C3 diagnostic inequality**: A(−1)=0.391 < A(0)=0.447 — **FAILS**, matching the main experiment's A(−1)=0.051 < A(0)=0.744.
- Random-control curve is essentially flat (0.447–0.480) over the interpretable range, confirming that the V_lang decrement at negative α is direction-specific.
- `claim_supported = fail`, `consistent_with_main_experiment = pass`.

## Prior audit (superseded)
Prior verdict: **FAIL** (2026-07-14 12:21). Reason: random-direction control was not available at audit time — a sub-audit reported the ALL_ALPHAS grid before the random arm had finished dispatching (dispatch race). The other two FAIL sub-criteria (α not in σ_proj units, α≥+0.5 in collapse range) are WARN-level once the random control is present, because they are addressable at interpretation time (label collapse range, note σ_proj as future work) and no longer decisive. See git history / this file's replaced content for the original.

## Action Items (remaining, non-blocking)
- Rescale α as multiples of σ_proj = std(V^T h) if the variant grid is ever re-reported for cross-paper comparison.
- Report only the non-collapse window (α ∈ [−1.5, +0.25]) as the interpretable regime; flag α ∈ {+0.5, +1.0, +1.5} as OOD-forcing.

## Effect on Phase 9 Combined Verdict
- experiment audit: WARN (unchanged: single seed, n=50/lang, `grade()` dead import)
- mechanism audit: **WARN** (upgraded from FAIL)
- combined = max_severity(WARN, WARN) = **WARN** → variant is now **ELIGIBLE** for robustness numerator and denominator.
