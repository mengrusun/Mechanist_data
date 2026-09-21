## C1 — Set-level AUROC of the frozen 19-feature set S vs DSSP α-helix labels

| Organism | Helix def | Frozen S (mean ± std, n=3 seeds) | Confound-only | Shuffle-null | M0 verdict |
|---|---|---|---|---|---|
| prokaryote | HGI | **0.8967 ± 0.0032** | 0.506 | 0.53 | established |
| prokaryote | H_only | **0.9055 ± 0.0018** | 0.506 | 0.53 | established |
| eukaryote | HGI | **0.8644 ± 0.0022** | 0.506 | 0.53 | established |
| eukaryote | H_only | **0.8676 ± 0.0040** | 0.506 | 0.53 | established |

_Source: `results/m0_feature_set.json`._ Frozen S = 19 features (round-1 provenance, no re-mining). Confound-only = GC3 + codon-position logistic AUROC on the same eval set. Shuffle-null = set-level AUROC after label shuffle. Best single-feature relaxed AUROC 0.62–0.68 (< 0.75) → **set-level, not single-feature**.
