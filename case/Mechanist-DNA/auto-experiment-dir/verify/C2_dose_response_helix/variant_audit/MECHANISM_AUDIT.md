# Mechanism Audit — C2 Variant (model-swap-h-only-helix-frac)
**Scope:** variant verify/C2_dose_response_helix/variants/model-swap-h-only-helix-frac/

## Check A — Steering Coefficient Sweep
**Result: PASS (reuses main experiment sweep)**
The variant does not run a new steering sweep — it reanalyzes the M2 result files under a different endpoint metric. The steering coefficient sweep used to generate the data is the same as the main experiment (10-dose sigma_proj grid, interior c*=21.48, split-sample P2, capability metric logged). The variant's single change (endpoint helix_hgi_w → helix_h_w) does not affect the mechanism rigor of the underlying data.

## Overall Verdict
**overall_verdict: pass**
