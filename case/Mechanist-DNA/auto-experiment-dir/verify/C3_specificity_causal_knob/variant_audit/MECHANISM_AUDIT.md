# Mechanism Audit — C3 Variant (model-swap-h-only-specificity)
**Scope:** variant verify/C3_specificity_causal_knob/variants/model-swap-h-only-specificity/

## Check A — Steering Coefficient Sweep
**Result: PASS (reuses main experiment data)**
The variant reanalyzes M3 arm + null files under a different endpoint metric. The underlying data was generated with the full mechanism-rigor protocol (48 norm-matched null directions, locked interior c*, split-sample, per-direction capability check). The endpoint change (helix_h_w instead of helix_hgi_w) does not affect the mechanism rigor of the underlying runs.

## Overall Verdict
**overall_verdict: pass**
