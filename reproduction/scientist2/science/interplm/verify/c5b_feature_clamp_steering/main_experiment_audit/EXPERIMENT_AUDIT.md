# Experiment Audit Report — Claim C5b

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP; model=gpt-5.4)
**Project**: Reproduction of Five SAE-on-ESM-2 Interpretability Claims
**Claim**: C5b — Clamping a labeled SAE feature during ESM-2 sequence generation steers generation toward the target biological property with monotone dose-response, above no-steering / random-clamp / mean-activation-addition baselines.
**Linked milestones**: M6

## Overall Verdict: WARN
*This is C5b's integrity verdict.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
- External property checkers: SignalP heuristic (N-terminal hydrophobicity rule), Kyte-Doolittle sliding-window for TM, ProSite-style regex for Zn-finger patterns, N-glycosylation regex N[^P][ST][^P]. These are rule-based, independent of ESM-2 model outputs.
- Evidence: `scripts/m6_feature_clamp_steering.py:63-115` — `check_signal_peptide()`, `check_transmembrane()`, `check_zinc_binding()`, `check_n_glycosylation()` are all rule-based functions.

### B. Score Normalization: PASS
- Yield = fraction of completions carrying target property (raw count/total, no model-max normalization). Alpha scaling uses sigma_f (feature activation std) as a biologically-motivated units choice, not a score normalization.
- Evidence: `scripts/m6_feature_clamp_steering.py:376-383`: `eff_alpha = alpha * sigma` for each arm.

### C. Result File Existence: PASS
- `runs/m6/yields.parquet` and `runs/m6/plausibility.parquet` exist. `runs/m6/summary.md` reports per-feature × arm × dose yield values.
- EXPERIMENT_RESULTS.md: three features (3998 TM, 4209 Zn, 1240 SP); no_steer yields 0.333/0.133/0.167 vs. SAE-clamp 0.200/0.100/0.133 — these numbers match the summary.md table.
- Tracker row m6: status=done.

### D. Dead Code Detection: PASS
- All four arms (no_steer, sae_clamp, mean_add, random_clamp) are implemented and executed in the `main()` loop. The `SteeringHook` class is instantiated and registered per generation call. `compute_mean_activation_direction()` is called for mean_add arm. No unreached evaluation code found.

### E. Scope Assessment: WARN
- Planned: 4 features × 4 doses × 4 arms × 3 seeds × 25 seqs = 4800 generations.
- Actual: 3 features × 4 doses × 4 arms × 3 seeds × 10 seqs = 1440 generations.
- Reductions: 3/4 features (1 target-property feature not found by M4), 10/25 seed sequences per batch. Transparently disclosed.
- Dose sweep spans {0.5, 1, 2, 4} × sigma_f — 4 data points covering 8-fold range.
- The result is a null outcome (steering reduces yield) with an adequate arm-set design (no-steer, sae_clamp, mean_add, random_clamp all present). Plausibility band tracked and maintained.
- Assessment: WARN (scope reduction honestly disclosed), not FAIL.

### F. Evaluation Type: WARN
- Yield measurement: rule-based property checkers (external, not model-derived) = real_gt.
- Plausibility: pseudo-perplexity under the same ESM-2 MLM = self_supervised_proxy (model scores its own generation).
- Dual-type; the yield measurement is the primary criterion for the claim.

## Action Items
- Extend to 25 seed sequences/batch for more statistical power on the null result.
- Broaden alpha range: try {0.1, 0.5, 1, 2, 4, 8, 16} × sigma_f (≥3 orders of magnitude) to detect any effect at higher magnitudes.
- Investigate whether the feature clamping approach is correctly implemented: SteeringHook adds to the post-layer residual stream at all positions, which may differ from the clamping described in the reference (which may target the SAE's input, not the residual).
