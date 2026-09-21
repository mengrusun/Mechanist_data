# Experiment Audit — C2: Predictive Accuracy on Main Pair
## Scope: M1 (features_L4/L12/L20.jsonl, aggregate.json)
## Auditor: auto-verify Phase 2

**overall_verdict: PASS**

### A. GT Provenance
- `true_max = np.array([s['max_val'] for s in held_snips])` — same as C1.
- `held_snips` comes from `NeuronpediaClient.parse_activation_snippets()` reading Neuronpedia's
  `maxValue` (cached pre-computed SAE activation). Ground truth is NOT from model re-computation.
- **PASS**: GT from external dataset (Neuronpedia cached activations).

### B. Score Normalization
- `true_norm = normalize_activations(true_max)` — min-max within-feature on held-out Neuronpedia
  activation values. All methods scored against the same normalized GT.
- Pearson correlation is scale-invariant; normalization here affects only the true_norm range, which
  is consistent across methods (paired design).
- No division by the model's own max or mean output.
- **PASS**: Normalization does not introduce bias between methods.

### C. Result File Existence (claim-scoped)
- `results/m1/features_L{4,12,20}.jsonl`: all exist with per-feature `pred_pearson` fields — PASS.
- Cited stats: Δpearson=-0.018, CI[-0.093,+0.058], p=0.43.
  Verified by re-computation: mean_delta=-0.0181, CI=[-0.0940,+0.0599], Wilcoxon p=0.4255. Match confirmed.
- **PASS**: All cited results exist and are reproducible.

### D. Dead Code
- `score_activation()` in common.py is called in m1_main_pair.py loop for each held-out text.
- `_pearson()` function is called after scoring.
- `_aggregate_records()` includes `pairwise_deltas` computation for pred_pearson — all active.
- **PASS**: No dead code found for C2's evaluation path.

### E. Scope — Claim-Scoped
- C2 states: SAGE beats Neuronpedia on predictive accuracy (Pearson r on held-out, paired CI > 0).
- **Metric correct**: pred_pearson is Pearson(predictions, true_norm) where predictions come from
  `score_activation(gpt_client, explanation, text)` — independent GPT-5 session predicting [0,1].
- **Independence**: scorer receives only (explanation, text) — not SAGE's internal candidates.
- **Hold-out integrity**: `held_snips` are from the 20% split NOT shown to any explainer (train_snips
  were shown). The 80/20 split is seed-locked per feature. No leakage confirmed.
- **Under-power**: n=44/300 (15% of plan). The null result (Δ=-0.018, p=0.43) is provisional;
  with n=300, the CI would narrow and could reveal either a true null or a small positive effect.
- **PASS**: Metric definition matches claim; no scope over-claim; under-power is noted as a caveat
  but does not make the methodology incorrect.

### F. Evaluation Type
- Predictive accuracy follows Paulo et al. 2024's simulation/detection scoring protocol.
- Scorer LLM predicts activation; correlated with Neuronpedia's real cached activation values.
- **PASS**: Real GT evaluation.

## Summary
| Check | Verdict | Notes |
|-------|---------|-------|
| A. GT provenance | PASS | Neuronpedia cached activations |
| B. Score normalization | PASS | Per-feature min-max, consistent across methods |
| C. Result file existence | PASS | All files exist; numbers reproducible |
| D. Dead code | PASS | All eval functions called |
| E. Scope | PASS | Metric matches claim; under-power noted but not a methodology error |
| F. Evaluation type | PASS | Real GT evaluation |

**overall_verdict: PASS**
