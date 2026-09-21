## C2: robustness = 1.00 (threshold = 0.5, eligible = 1/1)  →  PASS

- swap_variants_run: true
- Main-experiment verdict on C2: supported
- Variant counts (over `consistent_with_main_experiment`): 1 pass, 0 fail
  (of 1 eligible; 0 excluded for integrity reasons)
- Model dimension (H-only helix fraction swap): matches the main experiment (consistent=pass,
  claim_supported=pass, ESMFold rho=0.867 p=0.0012 / OmegaFold rho=0.903 p=0.0003, both significant positive)

**Swap axis:** model (structural readout metric: pLDDT-weighted HGI helix fraction → pLDDT-weighted H-only helix fraction)
**Variant result:** ESMFold Spearman rho=0.867 (p=0.0012), delta at c*=+0.144;
  OmegaFold rho=0.903 (p=0.0003), delta at c*=+0.147. Both predictors significant,
  positive, and dual-agreeing. Delta slightly larger than HGI (+0.129/+0.135) —
  consistent with H-only removing G/I-helix background noise, sharpening the signal.

Interpretation: C2's dose-response trend is robust to the helix labeling model.
Under strict H-only definition (excluding 3-10 and pi helix types), the Spearman
trend remains strongly significant across both predictors, with interior c*=21.48
still the optimal dose. The monotone dose-response to interior c*, dual-predictor
agreement, and delta > 0.10 all hold under H-only. C2 is robust.

**Artifacts:**
- verify/C2_dose_response_helix/variants/model-swap-h-only-helix-frac/result.json
- verify/C2_dose_response_helix/variants/model-swap-h-only-helix-frac/verdict.json
- verify/C2_dose_response_helix/variant_audit/EXPERIMENT_AUDIT.{md,json}
- verify/C2_dose_response_helix/variant_audit/MECHANISM_AUDIT.{md,json}
