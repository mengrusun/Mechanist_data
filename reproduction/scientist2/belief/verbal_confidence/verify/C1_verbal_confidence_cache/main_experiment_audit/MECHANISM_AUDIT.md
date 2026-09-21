# Mechanism Audit Report — Claim C1

**Date**: 2026-07-13
**Auditor**: External LLM reviewer (gpt-5.4 via dmxapi, cross-model)
**Project**: Verbal-Confidence Cache Hypothesis (Gemma-3-27B + TriviaQA)
**Claim**: C1 — post-answer hidden states carry a retrievable representation of the model's self-assessed score value; confidence is cached mid-generation, not computed on demand
**Linked milestones**: M1, M2, M3, M4, M5, M6

## Overall Verdict: FAIL

*This is C1's mechanism-rigor verdict. The steering experiment (M5) has several hard failures: no real capability/coherence metric (only a coarse collapse proxy); target effect is within the baseline-noise floor across all runs; no locked α; no random-direction control.*

## Triggered checks (this run): A

## Checks

### A. Steering Coefficient Sweep: FAIL

- Triggered: yes — via `scripts/m5_steer.py:13` (`h_L[pos] ← h_L[pos] + α · σ_L · u_hat`), `scripts/m5_steer.py:63` (alpha grid `[-4,-2,-1,0,1,2,4]`), `EXPERIMENT_PLAN.md:M5`
- Intervention type: activation steering (Representation and Parameter Analysis / Steering Vectors)
- Sweep grid: α ∈ {-4, -2, -1, 0, 1, 2, 4} × σ_proj (7 grid points, includes α=0)
- σ_proj scaling used: **yes** — `sigma_proj = std(X_train @ u)` at `scripts/m5_steer.py:240`; `alpha_scaled = alpha * sigma_proj` at line 250
- Capability metric logged: **none** — only `off_digit_rate` (argmax outside digit-vocabulary at C0), which is a coarse collapse proxy, not an independent capability metric (no perplexity, val-acc, or fluency score)
- Plateau range: **null** — no plateau identified; effect span ≈ 1–1.5 verbal-conf units across all α, while per-item std ≈ 41. Target effect is within baseline-noise floor.
- Locked α: **none** — experiment reports full sweep but selects no mid-plateau operating point
- α position in plateau: n/a (no plateau)
- Random-direction control: **no** — not run
- Sign pattern: n/a (no meaningful dose-response to evaluate sign)
- Output-case spot-check: cases not available — no raw post-steering text samples were logged; coherence/behavior verification is not possible
- Evidence:
  - `scripts/m5_steer.py:13` — additive hook `h_L[pos] += alpha_scaled * direction_unit`
  - `scripts/m5_steer.py:63` — alpha grid definition `default="-4,-2,-1,0,1,2,4"`
  - `scripts/m5_steer.py:240` — `sigma_proj = float(np.std(projections))`
  - `scripts/m5_steer.py:250` — `alpha_scaled = alpha * sigma_proj`
  - `scripts/m5_steer.py:284-287` — per_alpha record: `{mean, std, n, off_digit_rate}`
  - `scripts/m5_steer.py:305` — collapse_flag via off_digit_rate threshold
  - `results/m5/m5_diff_of_means_seed42.json` — span 1.27 units (47.00→45.73), std≈41
  - `results/m5/m5_lda_seed42.json` — span ~1.2 units, std≈41
  - `results/m5/m5_diff_of_means_seed123.json` — span <1 unit (essentially flat)
  - `results/m5/m5_lda_seed123.json` — non-monotone, span <2 units

- Verdict reason: FAIL on three hard criteria: (1) no independent capability/coherence metric (only `off_digit_rate` coarse proxy — cannot detect OOD drift or fluency collapse); (2) target effect is drowned by baseline noise (span ≈ 1–1.5 units << per-item std ≈ 41 — no plateau can be identified); (3) no locked α and no random-direction control. σ_proj scaling is correctly implemented. Sweep spans 7 points but only ~1.5 orders of magnitude in absolute α (0→4), not ≥3 orders of magnitude.

### B–F. Reserved (not_implemented)

Status: not yet implemented. Future checks may cover direction-extraction quality, site/layer selection rationale, n_effective sufficiency, probe-vs-causal disentanglement, intervention scope.

## Action Items

1. **Add a proper capability metric** (e.g., per-item fluency score, answer token log-prob on an unrelated subset, or coherence classifier) logged at every α point alongside `verb_conf_mean`. `off_digit_rate=0` is not sufficient to rule out OOD drift at large |α|.
2. **Log raw post-steering text samples** (≥5 items per α value) so reviewers can spot-check whether metric "wins" correspond to actual behavior changes.
3. **Report effect relative to within-seed std**: the span of ~1–1.5 verbal-conf units across α ∈ [-4×σ_proj, +4×σ_proj] is smaller than 1 standard deviation of baseline verbal-conf (std≈41). This means the steering direction at E4L10 has negligible causal effect; the "effect" is noise. This is a genuine negative finding for P4, not a methodology failure — but it also means there is no identifiable plateau to lock α in.
4. **Run random-direction control** (n_random ≥ 30 random unit vectors at the same α) to confirm the trained direction does not statistically beat random. Given the near-zero effect size, this would likely confirm the negative finding.
5. **Extend α sweep range** or try other sites to confirm the negative P4 finding is robust across the layer grid.
