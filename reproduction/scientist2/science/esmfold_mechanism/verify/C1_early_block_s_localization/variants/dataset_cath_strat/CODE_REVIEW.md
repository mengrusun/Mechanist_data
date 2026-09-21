# Code Review — Dataset Variant: CATH-stratified re-aggregation

**Script**: `verify/C1_early_block_s_localization/variants/dataset_cath_strat/run_variant.py`  
**Review date**: 2026-07-15

## Summary

Stats-only re-aggregation script. Reads existing M1 per-chain JSONL results and groups by `cath_label`. No ESMFold forwards.

## Findings

### PASS

1. **DSSP ground truth preserved**: No re-computation of DSSP or hairpin decisions — uses existing `is_hairpin` values from M1 results. Ground truth chain is unbroken.

2. **Correct aggregation**: Per-chain: mean over donors. Per-group: Wilcoxon paired test on (baseline, condition) pairs. Consistent with M1 methodology.

3. **Bootstrap CI**: Non-parametric bootstrap over chain pairs. Correct implementation (resample with replacement, compute mean delta per bootstrap).

4. **CATH-null handling**: Correctly maps null CATH labels to "cath_null" group. Single group = identical to M1 global result. Documented clearly.

5. **Predicate thresholds**: Same as M1 (Δ≥0.2pp, p<0.05). Correct.

### No WARN or FAIL findings.

## Verdict: PASS — script is correct. Execution status: COMPLETED (analytic from existing M1 data).
