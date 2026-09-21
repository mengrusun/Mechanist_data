## C2 Variant Integrity Audit — EXPERIMENT_AUDIT

**Claim:** c2 (sae_concept_alignment_gap)
**Variant:** model-swap-esm2-8m
**Audit target:** variant results at `verify/c2_sae_concept_alignment_gap/variants/model-swap-esm2-8m/results/`
**Overall verdict: PASS**

| Check | Status | Note |
|---|---|---|
| A. GT provenance | PASS | Swiss-Prot external GT, same file, same 1500-seq test split |
| B. Score normalization | PASS | F1 ∈ [0,1], ratio = covered_sae / covered_neuron (inf when denom=0) |
| C. Result existence | PASS | coverage.json, sensitivity.json, per_concept_best_F1.parquet, cost.json all present non-empty |
| D. Dead code | PASS | All functions called; no dead paths |
| E. Scope | PASS | Disclosed: 1500 seqs, 387195 residues, 400 concepts, 6 layers, 255s total |
| F. Eval type | PASS | Real external GT (Swiss-Prot), per-residue F1, identical to main experiment |

**Findings:** none

**Integrity status:** CLEAN (no warnings, no failures)
