# Experiment Audit — C1: Generative Accuracy on Main Pair
## Scope: M1 (features_L4/L12/L20.jsonl, aggregate.json), M0.5 (sanity.json)
## Auditor: auto-verify Phase 2

**overall_verdict: WARN**

### A. GT Provenance
- `true_max` in m1_main_pair.py (line ~215) is read from `s['max_val']` where `s` is from
  `NeuronpediaClient.parse_activation_snippets()` which reads `maxValue` from Neuronpedia's cached
  API response. This is Neuronpedia's pre-computed SAE activation — NOT re-computed from the model.
- **PASS**: Ground truth is from the external dataset (Neuronpedia), not the model being evaluated.

### B. Score Normalization
- `normalize_activations()` in common.py: min-max within-feature on the held-out `true_max` array.
- Normalization denominator is `hi - lo` of the held-out Neuronpedia activation values.
- This is NOT dividing by the model's own max output; it is a per-feature range normalization on
  the external ground truth. All three methods (SAGE, NP, GPT5-1shot) are scored against the
  SAME normalized GT.
- **PASS**: No self-referential normalization.

### C. Result File Existence (claim-scoped)
- `results/m1/features_L4.jsonl`: exists, 15 records, 77678 bytes — PASS
- `results/m1/features_L12.jsonl`: exists, 15 records, 78596 bytes — PASS
- `results/m1/features_L20.jsonl`: exists, 14 records, 70598 bytes — PASS
- `results/m1/summary.json`: exists — PASS
- `results/aggregate.json`: exists — PASS
- Numbers cited in EXPERIMENT_RESULTS.md (Δ=+0.018, CI[+0.005,+0.036], p=0.045) verified against
  raw data by re-computation: mean_delta=0.0182, CI=[0.0045,0.0364], Wilcoxon p=0.0455. Match confirmed.
- **PASS**: All cited results exist and are reproducible from the JSONL files.

### D. Dead Code
- `scripts/m1_main_pair.py` is present and contains all cited codepaths:
  `run_sage_loop()`, `write_probes()`, `compute_feature_activations_on_texts()`,
  `score_activation()`, `_paired_bootstrap_ci()` in `aggregate_files()`.
- All functions in `scripts/common.py` referenced in m1_main_pair.py are actually used.
- **PASS**: No dead eval code found.

### E. Scope — Claim-Scoped
- C1 states: SAGE beats Neuronpedia on generative accuracy (paired 95% CI lower bound > 0).
- **Metric correct**: gen_acc = hits/5 where hits = count(probe_activation > tau_f); this matches
  "success rate of triggering the feature with text written from the explanation".
- **Independence**: write_probes() receives only the final explanation string (not SAGE's internal
  candidates or round history). Same probe writer used for SAGE, NP, and GPT5-1shot explanations.
- **Scope WARN**: `gen_acc` has only 5 probes per feature per method — coarse resolution (possible
  values: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0). With 44 features, only 4 of 44 paired differences
  are non-zero (all = +0.2). Wilcoxon's effective n = 4; p=0.0455 is at the limit with n_eff=4.
  The bootstrap CI [+0.005, +0.036] is based on the full n=44 but reflects a coarse discrete
  distribution. The significance is real but fragile — one feature flipping would move the p-value
  substantially. This is a power/resolution concern, not a methodology error.
- **Under-power**: realized n=44/300 (15% of plan). All four claim verdicts are provisional.
- **WARN**: Scope language in EXPERIMENT_RESULTS.md says "partial-support" which is accurate; the
  n_eff concern is noted but not a fatal flaw in the evaluation methodology itself.

### F. Evaluation Type
- Generative accuracy uses a "probe-writing → activation-check" protocol consistent with
  Paulo et al. 2024's intervention scoring convention.
- Probe writer is an independent GPT-5 session (not one of SAGE's internal roles for the same feature).
- Ground-truth threshold τ_f computed from training-corpus token-level activations (99th percentile).
- **PASS**: Real-GT evaluation (feature activations are empirical, not synthetic proxies).

## Summary
| Check | Verdict | Notes |
|-------|---------|-------|
| A. GT provenance | PASS | Neuronpedia cached activations, not model output |
| B. Score normalization | PASS | Within-feature min-max on GT, same for all methods |
| C. Result file existence | PASS | All cited files exist; numbers reproducible |
| D. Dead code | PASS | All referenced functions used |
| E. Scope | WARN | n_eff=4 for Wilcoxon (90.9% zero differences); 5 probes/feature is coarse |
| F. Evaluation type | PASS | Real GT (empirical SAE activations) |

**overall_verdict: WARN** (Check E: scope — statistical resolution concern due to coarse gen_acc metric)
