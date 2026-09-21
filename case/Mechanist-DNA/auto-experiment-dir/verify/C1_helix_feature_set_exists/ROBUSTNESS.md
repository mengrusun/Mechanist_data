## C1: robustness = 1.00 (threshold = 0.5, eligible = 1/1)  →  PASS

- swap_variants_run: true
- Main-experiment verdict on C1: supported
- Variant counts (over `consistent_with_main_experiment`): 1 pass, 0 fail
  (of 1 eligible; 0 excluded for integrity reasons)
- Model dimension (H-only helix definition swap): matches the main experiment (consistent=pass,
  claim_supported=pass, delta negligible: +0.005 prokaryote / -0.002 eukaryote set-AUROC)

**Swap axis:** model (DSSP helix-labeling model: HGI → H-only strict alpha helix)
**Variant result:** prokaryote set-AUROC 0.906 ± 0.002 (n=3 seeds), eukaryote 0.868 ± 0.004
  — both meet the ≥0.80 M0 gate threshold, shuffle-null ~0.48, confound-only ~0.50.
  No meaningful drop from HGI results (0.901/0.866) — difference within seed variance.

Interpretation: C1's set-AUROC is robust to the choice of DSSP helix definition model.
The frozen S features mark strict alpha-helix codons at essentially the same AUROC as
HGI-inclusive coding, confirming the features respond to the core alpha-helix signal
and not to 3-10 or pi helix types. C1 (M0 verdict `established`) is robust.

**Artifacts:**
- verify/C1_helix_feature_set_exists/variants/model-swap-h-only-helix-def/result.json
- verify/C1_helix_feature_set_exists/variants/model-swap-h-only-helix-def/verdict.json
- verify/C1_helix_feature_set_exists/variant_audit/EXPERIMENT_AUDIT.{md,json}
- verify/C1_helix_feature_set_exists/variant_audit/MECHANISM_AUDIT.{md,json}
