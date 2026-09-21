## C3: robustness = 1.00 (threshold = 0.5, eligible = 1/1)  →  PASS

- swap_variants_run: true
- Main-experiment verdict on C3: supported
- Variant counts (over `consistent_with_main_experiment`): 1 pass, 0 fail
  (of 1 eligible; 0 excluded for integrity reasons)
- Model dimension (H-only specificity endpoint): matches the main experiment (consistent=pass,
  claim_supported=pass, 0/48 null dirs >= S under BOTH predictors, z=5.83/6.13, p=0.020 both)

**Swap axis:** model (specificity readout metric: pLDDT-weighted HGI → pLDDT-weighted H-only)
**Variant result:** ESMFold: 0/48 null dirs >= S (z=5.83, p=0.020), S-delta_h=+0.144,
  matched-control delta_h=-0.025; OmegaFold: 0/48 (z=6.13, p=0.020), S-delta_h=+0.147,
  matched-control delta_h=-0.021. z-scores marginally stronger than main (5.30→5.83 / 5.80→6.13),
  consistent with H-only reducing background noise.

Limitation noted: per-sample H-only values not stored in M3 files, so cluster-bootstrap CI
not available for the H-only endpoint. Empirical p-value and z-score test are valid
(aggregate-level N=48 null directions).

Interpretation: C3's specificity is robust to the helix labeling model. Under strict H-only,
S remains the clear outlier among 48 norm-matched random directions under both predictors.
The matched-control is negative. The specificity result (helix-axis specificity; β-arm documented
negative) holds under the model swap. C3 is robust.

**Artifacts:**
- verify/C3_specificity_causal_knob/variants/model-swap-h-only-specificity/result.json
- verify/C3_specificity_causal_knob/variants/model-swap-h-only-specificity/verdict.json
- verify/C3_specificity_causal_knob/variant_audit/EXPERIMENT_AUDIT.{md,json}
- verify/C3_specificity_causal_knob/variant_audit/MECHANISM_AUDIT.{md,json}
