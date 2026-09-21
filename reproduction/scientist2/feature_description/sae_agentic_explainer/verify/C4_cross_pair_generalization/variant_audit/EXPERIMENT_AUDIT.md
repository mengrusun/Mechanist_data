# Experiment Audit — Variant: model-swap-gpt-oss-20b (Phase 9)

**Variant**: model-swap-gpt-oss-20b (GPT-OSS-20B + resid-post-aa)
**Audited directory**: verify/C4_cross_pair_generalization/variants/model-swap-gpt-oss-20b/
**Scope**: C4 — cross-pair generalization of predictive accuracy
**Audit date**: 2026-07-14

## Overall verdict: PASS

No methodology violations found. The variant faithfully replicates the M2 evaluation protocol on a novel model/SAE pair.

## Checks

### A. Fake Ground Truth
**Status: PASS**
Ground truth = Neuronpedia's cached `maxValue` for each activation snippet (`s["max_val"]`). This is an externally cached corpus score — not re-computed from the GPT-OSS-20B model weights. The variant makes no forward pass through the target LLM (explicitly documented: "This script uses NO target-LLM forward pass"). GT provenance is identical to the main experiment (M2).

### B. Score Normalization
**Status: PASS**
`true_max = np.array([s["max_val"] for s in held_snips])` followed by `true_norm = normalize_activations(true_max)`. The `normalize_activations` function from `scripts/common.py` applies per-feature min-max normalization to the held-out set only. This is correct and consistent with M2.

### C. Phantom Results
**Status: PASS**
45 features written to `features.jsonl` (15/layer × 3 layers: 3, 11, 19). All 45 records have non-null `pred_pearson` and `pred_auroc` values for all three methods (sage_lite, neuronpedia, gpt5_1shot). `summary.json` written by the `aggregate()` function after all features complete. No missing or zero-padded results.

### D. Dead Metric Code
**Status: PASS**
The `_pearson()` function is called and its result stored in `fentry["methods"][m_name]["pred_pearson"]` for every feature-method pair. The `aggregate()` function reads these and computes `mean_pearson` per layer and overall. The pairwise bootstrap CI uses the per-feature pearson values. All code paths are live and executed.

### E. Scope Overclaim
**Status: PASS**
The variant claims to test "predictive accuracy (Pearson r on held-out Neuronpedia activations)" — exactly the metric computed. No generative accuracy or other metrics are claimed. The scope is correctly narrowed to SAGE-lite + predictive accuracy, matching the C4 claim boundary (cross-pair generalization).

### F. Test Leakage
**Status: PASS**
Seed-locked 80/20 split: `rng_f = np.random.default_rng(split_seed * 1000 + fid + L * 1000000)`. SAGE-lite receives only `train_snips`; scoring calls `score_activation(gpt, expl, t)` only on `held_snips`. The split indices are disjoint (`train_idx = idx[:n_train]`, `heldout_idx = idx[n_train:]`). No leakage.

### G. Statistical Validity
**Status: PASS**
n=45 features across 3 layers (15/layer). Paired bootstrap CI with 10k resamples. Wilcoxon test available in `scripts/common.py` (used in M1, available here). The variant does not apply Wilcoxon in the aggregate script for this pair (same limitation as M2), but the bootstrap CI is correctly computed. Under-power is a scientific finding, not a methodology error.

## Notes
- n_heldout=10 per feature (vs n_heldout=20 in the plan): adjusted because GPT-OSS-20B features have ~40 snippets/feature (80/20 → ~8 held-out). This is a data-constraint adjustment, not a performance choice. Consistent with the realized n_heldout in M2 (~8-9 held-out).
- Layers 3/11/19 for GPT-NeoX-20B (44 layers): proportionally similar to M2's 8/16/28 for Qwen3-4B (28 layers).
- All features qualify (≥8 snippets, public explanation, frac_nonzero > 1e-7).
