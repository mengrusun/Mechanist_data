# Experiment Audit — C1 Variant (model-swap-h-only-helix-def)
**Scope:** verify/C1_helix_feature_set_exists/variants/model-swap-h-only-helix-def/

## Check A — GT Provenance
**Result: PASS**
Same as main experiment: H-only labels come from the same experimental DSSP run on the same PDB structures. The H-only definition is a strict subset of the real DSSP output. No model-generated "ground truth."

## Check B — Score Normalization
**Result: PASS**
Same AUROC computation as main experiment. No model-max normalization.

## Check C — Result File Existence
**Result: PASS**
- analyze_c1_h_only.py reads 6 M0 H_only result files (all confirmed present).
- result.json written with prokaryote_mean_set_auroc=0.906, eukaryote_mean_set_auroc=0.868.
- verdict.json confirms claim_supported=pass.

## Check D — Dead Code
**Result: PASS**
The analysis script runs end-to-end; all code paths are executed. No dead evaluation code.

## Check E — Scope
**Result: PASS**
Single swap: helix_def HGI → H_only. Everything else frozen. No scope overclaim.

## Check F — Evaluation Type
**Result: PASS**
Same real_gt as main experiment — experimental DSSP labels.

## Overall Verdict
**overall_verdict: pass**
