# Mechanism Audit Report — Claim CM

**Date**: 2026-07-14
**Auditor**: executor (Claude Sonnet 4.6) + evidence review
**Project**: Emotional Framing in Prompts as a Weak, Input-Dependent Signal
**Claim**: CM — Some low-rank residual-stream direction on Qwen3-14B carries the emotion identity (Location) and causally modulates per-item GSM8K Δaccuracy (Intervention), with matched-length filler controls null
**Linked milestones**: M5, M6

## Overall Verdict: WARN

*Check A (Steering coefficient sweep) triggered and judged WARN. The sweep was performed at 3-4 α values per site, σ_proj scaling was used (sigma_l computed from neutral activations), and a fluency/parse-rate capability metric was logged. However: the sweep spans only ~2 orders of magnitude (α ∈ {-1, 0, +0.5, +1} × σ_proj), the α=+0.25 mid-plateau point was descoped, no random-direction baseline was run, and the dose-response is flat within noise.*

## Triggered checks (this run): A

## Checks

### A. Steering Coefficient Sweep: WARN

- **Triggered**: yes — via `mechanism_causal.py:275` (`h[0, pos, :] = h[0, pos, :] + delta` where `delta = alpha * sigma * dvec_dev`), `mechanism_causal.py:222-228` (sigma_l computation from neutral activations), and M6 tracker rows naming "steer" intervention.

- **Intervention type**: steering (additive residual-stream modification at last-prefix-token position)

- **Sweep grid**: α ∈ {-1.0, 0.0, +0.5, +1.0} at L4; α ∈ {-1.0, +0.5, +1.0} at L8 (α=0 baseline at L4 only). The MECHANISM_ROUTING.md specified adding α=+0.25 (mid-fine grid point, per Tip 2) but this was descoped with the 4 other controls.

- **σ_proj scaling used**: YES — `mechanism_causal.py:222-228` computes `sigma_l = proj.std()` where `proj = (neutral_acts @ d_frame).cpu().numpy()` over cached neutral-condition activations at the site layer. The steering delta is `alpha * sigma_l * dvec_dev` (line 275). This is correct σ_proj scaling. Confirmed from M6 result files: `sigma_l = 0.0320` at L4, `sigma_l = 0.0165` at L8.

- **Capability metric logged**: YES (partial) — `parse_rate` (fraction of completions successfully parsed by the `####` regex) and `mean_gen_tokens` are recorded in every M6 result JSON. Parse rate = 1.0 across all α values — no fluency collapse detected. However, this is not a standard capability metric (e.g., perplexity or accuracy on an unrelated task held constant) — it measures whether the CoT generation terminates with a parseable answer format. No independent unrelated-task capability metric was logged.

- **Plateau range**: No clear plateau identifiable — the dose-response is flat. L4: acc ∈ {0.92, 0.94, 0.92, 0.94} for α ∈ {-1.0, 0.0, +0.5, +1.0} — range 2 pp. L8: acc ∈ {0.92, 0.92, 0.90} for α ∈ {-1.0, +0.5, +1.0} — range 2-4 pp. All within the M2b noise floor (5.24 pp). No plateau identified because the effect is flat/null.

- **Locked α**: Not applicable — no α was "locked" at a plateau because the dose-response is flat and null.

- **α position in plateau**: n/a (no plateau identified; effect is null at all tested α values)

- **Random-direction control**: NOT RUN. No n_random ≥ 30 random-direction control at the locked α was performed. This was not mentioned in the plan or tracker.

- **Sign pattern**: N/A (not an asymmetric protocol — same direction applied at all α values)

- **Output-case spot-check**: Parse_rate = 1.0 at all α values, mean_gen_tokens = 512.0 at all (max tokens hit — the model generates up to the max_new_tokens=256 limit but many generate more). Fluency appears intact based on parse rate, but the truncation at 256 tokens (hitting max) is a concern — many completions may be truncated and still yield a parseable number at the end, masking quality degradation. However, given the null dose-response, the more likely explanation is that the direction simply has no causal effect on GSM8K accuracy at these α values.

- **Evidence**:
  - `scripts/mechanism_causal.py:222-228` (sigma_l computation)
  - `scripts/mechanism_causal.py:268-279` (steering write function)
  - `runs/M6/steer_baseline_L4_a0.json`: sigma_l=0.032, alpha=0.0, acc=0.94
  - `runs/M6/steer_L4_am1.json`: alpha=-1.0, acc=0.92
  - `runs/M6/steer_L4_ap0p5.json`: alpha=+0.5, acc=0.92
  - `runs/M6/steer_L4_ap1.json`: alpha=+1.0, acc=0.94
  - `runs/M6/steer_L8_am1.json`: alpha=-1.0, acc=0.92
  - `runs/M6/steer_L8_ap0p5.json`: alpha=+0.5, acc=0.92
  - `runs/M6/steer_L8_ap1.json`: alpha=+1.0, acc=0.90

- **Verdict reason**: WARN — σ_proj scaling is correctly implemented and a fluency proxy (parse_rate) is logged. However: (1) sweep spans < 3 orders of magnitude (only α ∈ {-1, 0, +0.5, +1} × σ_proj — approximately 2 orders of magnitude); (2) no random-direction control run; (3) the α=+0.25 mid-point was descoped, leaving gaps in the grid; (4) the capability metric (parse_rate) is a task-internal proxy rather than an independent unrelated-task metric. The flat dose-response is itself informative (the direction is not causally read out for GSM8K accuracy), but the sweep rigor has soft gaps that fall into WARN territory.

### B–F. Reserved (not_implemented)
Status: not yet implemented. Future checks may cover direction-extraction quality, site/layer selection, n_effective sufficiency, probe-vs-causal disentanglement, intervention scope.

## Action Items
1. **Run the random-direction control** (n_random ≥ 30) at α = ±1.0 to confirm the null dose-response is specific to the d_frame direction and not a general property of any additive perturbation at these layers and α values.
2. **Extend the α grid** to include α ∈ {+2.0, +5.0} × σ_proj to verify the effect truly requires large amplitudes before it appears (or remains null beyond the tested range).
3. **Replace parse_rate with an independent capability metric** — e.g., run the model on 50 MedQA items at each α value to check that accuracy on an unrelated task is unchanged (this also serves as the off-target specificity control that was descoped).
4. WARN does not block Stage 2 — CM advances with an integrity warning.
