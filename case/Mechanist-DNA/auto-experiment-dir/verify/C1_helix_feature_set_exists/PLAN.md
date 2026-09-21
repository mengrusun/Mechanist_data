# Verify Plan — C1: helix feature set exists

## Claim
In Evo2-7B's Layer-26 SAE, a set-level group of features selectively marks α-helix codons (frozen round-1 feature set S, cross-organism), beyond confounds and multiple-testing chance.

**Main-experiment verdict:** supported (M0 verdict `established`)
**Main-experiment metric:** set-AUROC 0.901 (prokaryote), 0.866 (eukaryote) vs confound-only ~0.51 and shuffle-null ~0.50

## Main experiment (from /auto-experiment)
- Method: per-codon set-level AUROC on frozen S vs DSSP HGI helix labels (experimental PDB structures)
- Dataset: natural CDS + experimental DSSP labels (round-1 caches, 531k codons), prokaryote + eukaryote
- Model (readout): DSSP HGI definition (H+G+I helix types) as the labeling model
- Metric: combined_set_test_auroc (oracle-threshold aggregation on test split)

## Variants

| # | Dimension | Swap | Replaces | Justification | Source |
|---|-----------|------|----------|---------------|--------|
| 1 | model | H-only helix definition (DSSP 'H' only, strict alpha helix, excludes 3-10 G and pi I) | HGI definition (H+G+I, includes all helix-like) | Tests if set-AUROC is robust to the DSSP helix labeling model — the only varying element in the selectivity computation. Strict H-only is a valid alternative definition used in structural biology. M0 already computed H-only configs alongside HGI, so results are on disk. | EXPERIMENT_PLAN.md §M0 "helix_def: [HGI, H_only]"; round-1 design note |

## Success Criterion (per variant)
The claim is supported if set-AUROC under H-only helix definition is ≥ 0.80 (the M0 gate threshold) for prokaryote, with eukaryote transfer ≥ 0.80 as well. The shuffle-null must remain ≈ 0.50 and confound-only AUROC must be substantially lower than S's AUROC (margin ≥ 0.1).

## Reviewer Notes (Phase 4)
The H-only swap is a genuine test of the helix-definition dependency: G-helices (3-10 helices) and I-helices (π helices) are rare but structurally distinct from alpha helices. If S's AUROC rests on features that respond to G/I helix codons rather than pure alpha helix, the H-only AUROC would drop. Conversely, if S genuinely captures alpha-helix structure, H-only should remain high. The fact that M0 pre-computed both definitions at the same quality (same caches, same DSSP run, same features) makes this a clean within-analysis model swap.

Implementation: read existing M0 H_only result files (no GPU, no re-run).
