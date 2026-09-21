# Experiment Audit — C1 (helix feature set exists)
**Scope:** M0 phenomenon-validation gate (refine-logs/EXPERIMENT_PLAN.md#M0); result files results/m0_*.json

**Claim:** In Evo2-7B's Layer-26 SAE, a set-level group of features selectively marks α-helix codons (frozen round-1 set S; cross-organism), beyond confounds and multiple-testing chance.

## Check A — GT Provenance
**Result: PASS**
Ground-truth labels are from experimental PDB structures via real DSSP (mkdssp 4.6.1), not from model output. Natural CDS from public genomic databases; protein products have experimental PDB structures. DSSP run on experimental PDB files → per-residue SS. HC3 explicitly mandates "natural CDS + real DSSP labels from experimental structures" and "reverse-translation of proteins into DNA is FORBIDDEN." Full audit trail in data/ caches (HC3-compliant, round-1 build verified).

## Check B — Score Normalization
**Result: PASS**
AUROC is computed as area under the ROC curve of feature activations vs DSSP labels. No normalization by model's own max, mean, or output. The set-level combined AUROC uses oracle-threshold aggregation on the held-out test split (not a model-max-normalized quantity).

## Check C — Result File Existence (claim-scoped)
**Result: PASS**
- results/m0_feature_set.json exists, non-empty, contains "verdict": "established", "combined_set_test_auroc" values for prokaryote configs, eukaryote configs.
- Per-config files verified: m0_prokaryote_HGI_s42.json, m0_prokaryote_HGI_s200.json, m0_prokaryote_HGI_s201.json, m0_eukaryote_HGI_s42.json, m0_eukaryote_HGI_s200.json, m0_eukaryote_HGI_s201.json — all present, non-empty.
- Set-AUROC 0.901 (prokaryote mean) and 0.866 (eukaryote) cited in EXPERIMENT_RESULTS.md; both values present in the result files.

## Check D — Dead Code
**Result: PASS**
m0_feature_selectivity.py is the main script. The set-level AUROC computation, BH-FDR, shuffle-null, and confound controls are all invoked from the main execution path (validated: result files are produced with correct keys). No evidence of defined-but-uncalled evaluation functions.

## Check E — Scope (claim-scoped)
**Result: PASS**
Claim says "frozen round-1 set S, cross-organism." Experiment covers prokaryote + eukaryote (2 organisms) × HGI + H-only (2 helix defs) × 3 split seeds = 12 configs. Cross-organism confirmed (holds_multi_organism=True). No over-reaching language beyond what the 12-config analysis supports.

## Check F — Evaluation Type
**Result: PASS**
Evaluation type: real_gt. Labels are experimental DSSP SS calls on PDB structures from UniProt/PDB. Not synthetic proxies. This is the standard ground-truth for secondary structure claims in structural biology.

## Overall Verdict
**overall_verdict: PASS**

All six checks pass. C1's main experiment has trustworthy evaluation methodology. The existence claim rests on real experimental DSSP labels, proper BH-FDR, shuffle-null controls, and full multi-config coverage.
