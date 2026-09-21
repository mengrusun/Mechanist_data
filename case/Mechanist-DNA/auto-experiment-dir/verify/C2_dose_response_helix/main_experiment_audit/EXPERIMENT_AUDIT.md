# Experiment Audit — C2 (dose-response helix)
**Scope:** M1 + M2 (refine-logs/EXPERIMENT_PLAN.md#M1, #M2); result files results/m1_calibration.json, results/m2_dose_response_curve.json, results/m2_alpha_helix_S_c*.json

**Claim:** Amplifying the α-helix feature set (σ_proj-unit dosing) during autoregressive DNA generation increases the encoded protein's pLDDT-weighted α-helix fraction, monotonically up to a mapped interior optimum, reproduced across two structure predictors.

## Check A — GT Provenance
**Result: PASS (with transparency note)**
The "ground truth" for C2 is the pLDDT-weighted α-helix fraction from ESMFold + OmegaFold structure predictions on generated proteins. These are model-generated structures (not experimental PDB), which is appropriate because C2 tests generated sequences — no experimental ground truth exists for novel AI-generated sequences. The dual-predictor design (two independent predictors: ESMFold, OmegaFold) provides cross-model robustness, reducing single-model bias. Evaluated as PASS: this is the standard and appropriate methodology for assessing steered generation outputs; the evaluation model (structure predictor) is not the intervention model (Evo2-7B+SAE).

## Check B — Score Normalization
**Result: PASS**
The pLDDT-weighted helix fraction (helix_hgi_w) is computed as the sum of (helix residue indicator × pLDDT weight) / total residues. No normalization by model's own max or mean. The Spearman ρ correlation is a rank statistic, not a ratio.

## Check C — Result File Existence (claim-scoped)
**Result: PASS**
- results/m2_dose_response_curve.json: exists, contains "spearman_rho", "spearman_p", "delta_at_cstar_vs_c0", "interior_cstar": true for both ESMFold (rho=0.867, p=0.0012) and OmegaFold (rho=0.879, p=0.0008) — exactly matching cited values.
- Per-dose per-seed result files: all 30 m2_alpha_helix_S_c*_s*.json files verified present.
- results/m1_calibration.json: exists, contains sigma_proj=0.4104 and baseline stats.
- c* selection: seed 42 used for selection, heldout_seeds=[200,201] confirmed in dose_response_curve.json.

## Check D — Dead Code
**Result: PASS**
m2_dose_response.py generates sequences, folds with both predictors, computes pLDDT-weighted and hard-gated helix fractions. m2_consolidate.py aggregates into the summary file. All evaluation paths are exercised (30 result files produced + consolidated curve).

## Check E — Scope (claim-scoped)
**Result: PASS**
Claim says "monotonically up to a mapped interior optimum, reproduced across two structure predictors." The experiment: (1) demonstrates interior c*=21.48 within the capability-preserved region (P7; valid-ORF at c*=0.893 ≈ baseline=0.880); (2) reports both predictors with agreement on trend sign (dual_predictor_agree=True); (3) uses split-sample c* selection (P2). Scope matches.

## Check F — Evaluation Type
**Result: synthetic_proxy**
Helix fraction measured via structure prediction (ESMFold + OmegaFold) — synthetic proxy, not experimental DSSP on PDB structures. This is appropriate and by design for generated sequences. The dual-predictor design mitigates single-predictor bias.

## Overall Verdict
**overall_verdict: pass**

The synthetic_proxy evaluation is appropriate and by design. Both predictors agree. c* is a demonstrated interior maximum. The split-sample design prevents c* post-selection leakage. All result files exist with matching values.
