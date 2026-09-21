# Experiment Audit — C2 Variant (model-swap-h-only-helix-frac)
**Scope:** verify/C2_dose_response_helix/variants/model-swap-h-only-helix-frac/

## Check A — GT Provenance
**Result: PASS**
Same as main experiment: helix fraction from ESMFold + OmegaFold structure prediction. Appropriate for generated sequences. helix_h_w_mean stored per run alongside helix_hgi_w_mean in the M2 result files — same prediction pipeline.

## Check B — Score Normalization
**Result: PASS**
Spearman rho is a rank statistic. helix_h_w is an unscaled fraction. No model-max normalization.

## Check C — Result File Existence
**Result: PASS**
30 M2 result files all present; result.json written with rho=0.867/0.903 and delta+0.144/+0.147; verdict.json confirms pass.

## Check D — Dead Code
**Result: PASS**
All code paths in analyze_c2_h_only.py executed.

## Check E — Scope
**Result: PASS**
Single endpoint swap (helix_hgi_w → helix_h_w); same dose grid, seeds, predictors.

## Check F — Evaluation Type
**Result: synthetic_proxy (appropriate)**
Same structure prediction as main experiment.

## Overall Verdict
**overall_verdict: pass**
