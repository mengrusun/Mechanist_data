# Research-Lit — Raw Retrieval Dump (audit-only)

**Date**: 2026-08-19
**Query focus**: Evo2-7B; steering/activation editing for DNA/protein generation; SAE features for protein secondary structure (α-helix); α-helix-content evaluation of generated sequences.

## Retrieval provenance
- **WebSearch** (web + arXiv/bioRxiv surface): 5 queries executed (Evo2 model; steering genomic LMs; Goodfire SAE Evo2 α-helix; α-helix SS prediction from ORF; Evo2 autoregressive controllable generation).
- **mechanic-db cloud SEARCH** (`/mechanic-db-search`): skill loaded; the `search_papers` MCP tool is **not exposed** in this isolated claim-agent context (no `mcp__mechanic-db__*` in tool schema, no project `.mcp.json`). Treated as an unavailable source for this run per the skill's graceful-degradation contract; coverage supplied by WebSearch. Decomposed query that would be submitted is recorded below for audit.
- **arXiv PDF download**: not attempted (ARXIV_DOWNLOAD=false).

## Decomposed mechanic-db query (prepared; not submitted — tool unavailable)
```json
{
  "original_query": "mechanistic interpretability of genomic DNA foundation models Evo 2, sparse autoencoder features for protein secondary structure alpha helix, activation steering for controllable DNA sequence generation",
  "is_cross_domain": false,
  "sub_queries": [
    {
      "domain": "AI interpretability",
      "db": "interp_db",
      "semantic_query": "sparse autoencoder features and activation steering in genomic DNA foundation models, protein secondary structure alpha-helix beta-sheet directions, residual-stream feature clamping for controllable autoregressive sequence generation in Evo 2",
      "keywords": ["sparse autoencoder", "activation steering", "genomic language model", "protein secondary structure", "controllable generation"],
      "year_min": 2024, "year_max": null, "min_citations": null,
      "techniques": ["feature_dictionary_learning", "causal_attribution", "probing"],
      "components": ["residual_stream"],
      "task_scenarios": ["science"],
      "abilities": [],
      "target_models": ["Evo 2", "Evo2-7B"],
      "model_families": ["Evo"],
      "hyde_text": "We investigate how a genomic DNA foundation model internally represents protein secondary structure and whether these representations can be used to steer generation. Using sparse autoencoders trained on residual-stream activations, we analyze features that activate on alpha-helical and beta-sheet coding regions and validate them against predicted protein structures. We characterize a low-dimensional direction encoding helical propensity and test whether clamping the corresponding feature or adding a contrastive steering vector during autoregressive nucleotide generation increases the alpha-helical content of the encoded protein. Our analysis reports dose-response of the intervention coefficient, specificity against matched control directions, and off-target effects on sequence validity and coding structure. Findings support the hypothesis that a localizable internal component carries helical-structure information and causally influences generated sequence composition."
    }
  ]
}
```

## Key retrieved items (deduplicated, most relevant first)
1. Evo 2 — "Genome modelling and design across all domains of life" (Nature 2026; Arc Institute preprint 2025). Model card: `arcinstitute/evo2_7b` (HuggingFace). Autoregressive, StripedHyena-2, 7B/40B, temperature/top-k/top-p sampling.
2. Goodfire × Arc — "Interpreting Evo 2" (goodfire.com/research/interpreting-evo-2). BatchTopK SAEs on **Layer 26**; features for **α-helix, β-sheet**, tRNA, exon/intron; validated vs AlphaFold3. Steering = "early signs of life", "further research needed". Artifact: `Goodfire/Evo-2-Layer-26-Mixed`.
3. ARCADE — "Controllable Codon Design from Foundation Models via Activation Engineering" (bioRxiv 2025.08.19.668819). Training-free activation steering on (coding) foundation models.
4. "Steering Protein Language Models" (arXiv 2509.07983). Contrastive-set steering vectors; training-free property control.
5. FoldSAE — "Learning to Steer Protein Folding Through Sparse Representations" (arXiv 2511.22519).
6. "Sparse Autoencoders Reveal Interpretable Structure in Small Gene Language Models" (arXiv 2507.07486).
7. "Sparse autoencoders uncover biologically interpretable features in protein language model representations" (PNAS 2025; doi 10.1073/pnas.2506316122).
8. "Language Models for Controllable DNA Sequence Design" (arXiv 2507.19523); ATGC-Gen.
9. "Steering Vector Fields for Property-Controlled Molecular Generation with Chemical Language Models" (bioRxiv 2025.09.24.678080).
10. "Controlling Large Language Models Through Concept Activation Vectors" (AAAI) — CAV linear-concept control template.
11. Protein secondary-structure prediction / α-helix-content methods: DSSP (H/G/I→H); NetSurfP-3.0, SPOT-1D-LM, S4PRED, DML-SS; "End-to-End Deep Learning Model to Predict and Design Secondary Structure Content of Structural Proteins" (PMC9347213); ESMFold/AlphaFold2 → DSSP route.
