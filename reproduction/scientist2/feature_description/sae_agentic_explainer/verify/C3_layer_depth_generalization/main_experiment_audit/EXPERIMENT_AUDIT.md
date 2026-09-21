# Experiment Audit — C3: Layer-Depth Generalization
## Scope: M1 layer-stratified read-off from features_L{4,12,20}.jsonl
## Auditor: auto-verify Phase 2

**overall_verdict: WARN**

### A. GT Provenance
- Same as C1/C2 — Neuronpedia cached activations for both gen_acc and pred_pearson.
- **PASS**: GT from external dataset.

### B. Score Normalization
- Same as C2 for pred_pearson; same as C1 for gen_acc.
- **PASS**: No self-referential normalization.

### C. Result File Existence (claim-scoped)
- C3 reads from the same per-layer JSONL files as C1/C2.
- Cited layer-stratified stats verified by re-computation:
  - L4: ΔGenAcc=+0.027 CI[0.000,+0.067], ΔPearson=-0.065 CI[-0.161,+0.040]
  - L12: ΔGenAcc=+0.027 CI[0.000,+0.067], ΔPearson=-0.004 CI[-0.168,+0.157]
  - L20: ΔGenAcc=0.000 CI[0.000,0.000], ΔPearson=+0.018 CI[-0.086,+0.144]
  All match EXPERIMENT_RESULTS.md within rounding.
- **PASS**: Files exist; numbers reproducible.

### D. Dead Code
- C3's statistics are computed in `aggregate_files()` → `_aggregate_records()` with per-layer
  breakdown in `pairwise_deltas[...][per_layer]`. These code paths are active.
- **PASS**: No dead code.

### E. Scope — Claim-Scoped
- C3 states: gains hold at each of the three depths (early L4, mid L12, late L20) with per-depth
  paired 95% CI lower bound > 0 and Bonferroni α=0.05/6.
- **Metric correct**: same gen_acc and pred_pearson metrics as C1/C2 computed per layer.
- **WARN — Statistical limitations**:
  1. Per-layer n = 14-15 features. For gen_acc with 5 probes/feature, the effective resolution
     is very coarse. L20's gen_acc differences are ALL ZERO (zero-difference Wilcoxon cannot run).
     L4 and L12 each have only 2 non-zero differences.
  2. Bonferroni correction: EXPERIMENT_RESULTS.md mentions Bonferroni across 3 layers × 2 metrics
     = 6 tests. However, the actual scripts do NOT apply Bonferroni explicitly — the per-layer
     bootstrap CIs are computed without correction. The reported "no layer CI excludes zero"
     verdict is correct (all CIs include zero) but the Bonferroni framing is slightly misleading
     since the individual per-layer tests are not Bonferroni-adjusted in the code.
  3. Under-power: planned n=100/layer realized n=14-15/layer (15% of plan).
- **WARN**: The Bonferroni claim in EXPERIMENT_RESULTS.md is not explicitly implemented in the
  aggregate script (no `alpha/6` threshold applied), though it doesn't change the verdict
  (all CIs include zero regardless). This is a minor documentation mismatch, not a fabrication.

### F. Evaluation Type
- Same as C1/C2 — real GT evaluation.
- **PASS**

## Summary
| Check | Verdict | Notes |
|-------|---------|-------|
| A. GT provenance | PASS | Neuronpedia cached activations |
| B. Score normalization | PASS | Consistent with C1/C2 |
| C. Result file existence | PASS | Per-layer stats verified |
| D. Dead code | PASS | Per-layer aggregate code active |
| E. Scope | WARN | Bonferroni not explicitly coded in script; per-layer n=14-15 very small; L20 all-zero gen_acc differences |
| F. Evaluation type | PASS | Real GT |

**overall_verdict: WARN** (Check E: Bonferroni framing not implemented in code; per-layer sample too small to detect directional effects)
