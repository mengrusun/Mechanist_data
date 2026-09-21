# Mechanism Audit Report — Claim C3b

**Date**: 2026-07-14
**Auditor**: external LLM reviewer (gpt-5.4, via llm-chat API, cross-model)
**Project**: Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence
**Claim**: C3b — causal separability via cross-direction activation steering.
**Linked milestones**: M3 (B3)

## Overall Verdict: WARN

*This is C3b's mechanism-rigor verdict. Check A (Steering Coefficient Sweep) was triggered.*

## Triggered checks (this run): A

## Checks

### A. Steering Coefficient Sweep: WARN
- Triggered: yes — `step5_steering.py` contains `alpha_sigma = alpha * sigma_L*` and `h_new[b, pos[b], :] = h_new[b, pos[b], :] + add` (additive steering at last-input-token residual)
- Intervention type: steering (additive hook on residual stream)
- Sweep grid: [-1.0, 0.0, +1.0] in sigma_L* units (3 values)
- sigma_proj scaling used: yes — `alpha_sigma = alpha * sigma_L*` where `sigma_L* = std(elementwise activations at L* on training set) = 0.585`
- Capability metric logged: yes — `mean_token_perplexity` computed per condition via `_compute_ppl()` function; perplexity stable (turn1 mean 2.24, turn2 mean 1.76) across all alpha; no blow-up detected
- Plateau range: null — with only 3 grid points spanning 1 order of magnitude (alpha=0 to alpha=1), no plateau can be identified
- Locked alpha: 1.0 (in sigma_L* units; alpha_sigma = 0.585)
- Position in plateau: n/a (no plateau identifiable with 3 points)
- Random-direction control: {run: true, n_random: 1, passed: true (null result consistent with matched random)}. Single random vector drawn per run — well below recommended n_random >= 30. WARN: underpowered control.
- Sign pattern: n/a (symmetric protocol, both +alpha and -alpha tested)
- Output-case spot-check: cases_available: true (gens_sample[:5] logged per condition). Verdict: metric_text_consistent — perplexity stable and text reported coherent at alpha=+-1 (the planned anti-claim that "emitted output not automatic at 1sigma" is consistent with outputs). No metric/text mismatch detected.
- Evidence: `code/step5_steering.py` lines ~95-100 (additive hook), lines ~145 (alpha grid), lines ~176 (sigma_L* computation), lines ~342-345 (ppl computation), `runs/M3_steering/cost.json` (confirms sigma_L*=0.585, perplexity stable)

- Verdict reason: WARN — sparse alpha grid (3 values, < 5 grid points, < 3 orders of magnitude required per WARN criterion) and single random-direction control (n=1 << 30 recommended). No FAIL criteria triggered: sweep performed (not hardcoded), capability metric logged, no capacity crash, no sign-pattern violation. The null-steering result is geometrically expected from |cos(v_c,v_v)|=0.015 and is scientifically honest.

### B–F. Reserved (not_implemented)
Status: not yet implemented. Future checks may cover direction-extraction quality, site/layer selection, n_effective sufficiency, probe-vs-causal disentanglement, intervention scope.

## Action Items
- Expand the alpha sweep to >= 5 grid points spanning >= 2-3 orders of magnitude (e.g., [0.1, 0.3, 1.0, 3.0, 10.0] * sigma_L*) in a future iteration to confirm no plateau exists in the relevant range.
- Use n_random >= 30 random directions to establish a proper null distribution for the single-random-control comparison.
