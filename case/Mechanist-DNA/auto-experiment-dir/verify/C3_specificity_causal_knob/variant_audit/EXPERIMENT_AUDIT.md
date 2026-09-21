# Experiment Audit — C3 Variant (model-swap-h-only-specificity)
**Scope:** verify/C3_specificity_causal_knob/variants/model-swap-h-only-specificity/

## Check A — GT Provenance
**Result: PASS**
Same structure prediction pipeline as main experiment. helix_h_w_mean stored per direction in M3 null files and per arm in M3 arm files — same ESMFold + OmegaFold pipeline.

## Check B — Score Normalization
**Result: PASS**
Empirical p-value is rank-based (fraction of null dirs >= S). No normalization by model max.

## Check C — Result File Existence
**Result: PASS**
- M3 arm files (S at c*, c=0, matched-control) for seeds 200+201: all present.
- 16 M3 null chunk JSON files covering 48 directions: all present.
- result.json written with 0/48 null >= S, z=5.83/6.13, p=0.020 both predictors; verdict.json confirmed.

## Check D — Dead Code
**Result: PASS**
All code paths in analyze_c3_h_only.py executed.

## Check E — Scope
**Result: PASS (minor caveat noted)**
Single endpoint swap. Caveat: per-sample H-only values not stored in M3 files, so cluster-bootstrap CI is not available for H-only; uses aggregate-level z-score. This is a limitation noted in DIFF.md and does not undermine the empirical p-value (which is the primary test). No overclaim.

## Check F — Evaluation Type
**Result: synthetic_proxy (appropriate)**
Same as main experiment.

## Overall Verdict
**overall_verdict: pass**
Caveat noted (no cluster-bootstrap CI for H-only) but does not constitute a FAIL; the empirical p-value test is valid.
