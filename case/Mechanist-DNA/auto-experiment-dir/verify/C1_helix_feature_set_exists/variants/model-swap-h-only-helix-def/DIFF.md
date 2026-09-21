# DIFF — C1 model-swap-h-only-helix-def vs Main Experiment

## What changed
**Single change**: helix labeling model swapped from **HGI** (DSSP H+G+I helix types — standard alpha, 3-10, and pi helices) to **H-only** (strict DSSP 'H' alpha helix only, excluding G-helices and I-helices).

## What is frozen (identical to main experiment)
- Feature set S: frozen round-1 19 features (unchanged)
- Dataset: same M0 caches (natural CDS + experimental PDB DSSP labels, 531k codons)
- AUROC computation: same oracle-threshold aggregation on test split
- BH-FDR procedure: same
- Shuffle-null: same (20 permutations)
- Confound controls: same
- Train/val/test splits: same (same split seeds: 42, 200, 201)
- Both organisms: prokaryote + eukaryote (same)

## Why this is a valid model swap
HGI and H-only differ in which DSSP-assigned helices count as "helix." G-helices (3-10 helices, rare, ~2-5% of protein residues) and I-helices (pi helices, very rare) are included in HGI but excluded from H-only. If S's features genuinely capture alpha-helix structure, the H-only AUROC should remain high. If they rely on G/I-helix codon patterns, the H-only AUROC would drop.

## Implementation
Reads pre-computed M0 H_only files from results/ directory. No GPU required. The M0 script ran all 12 configurations (2 organisms × 2 helix defs × 3 seeds) simultaneously, so H_only results are on disk already.

## Config change summary
```
main experiment: --helix_def HGI    → variant: --helix_def H_only
```
All other flags unchanged.
