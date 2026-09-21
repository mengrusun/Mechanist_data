# Research Literature — Raw Retrieval Dump (audit-only)

**Direction**: Steering Evo2-7B to generate DNA sequences with high α-helical content
**Behavior-source**: given · **Mechanism**: discovery
**Date**: 2026-08-23
**Retrieval sources executed**: cloud `mechanic-db-search` (mandatory, executed) + web retrieval.

## mechanic-db-search (cloud SEARCH service)

- Executed via the documented local fallback (the `mechanic-db` MCP server is registered in the
  Mechanist repo's `.mcp.json`, not this project session; called `run_search` with the same
  `MECHANIC_DB_API_KEY` / `MECHANIC_DB_BASE_URL` env — registered tier).
- Decomposed cross-domain query: sub-query 1 `interp_db` (activation steering / representation
  control / probing / causal intervention for controllable generation), sub-query 2 `sciatlas_db`
  (genomic foundation models, controllable DNA/protein design, α-helix / secondary structure, ORF
  validity).
- Result: **299 papers** returned (registered tier). Full payload:
  `mechanic_db_cache/20260823_135908_evo2_helix_steer.json` (field `papers[]`).

### Most relevant retrieved papers (title — year)

Steering methods / causal intervention (interp_db):
- Steering Llama 2 via Contrastive Activation Addition (CAA) — 2023
- Improving Activation Steering in Language Models with Mean-Centring — 2023
- Extracting Latent Steering Vectors from Pretrained Language Models — 2022
- Multi-property Steering of LLMs with Dynamic Activation Composition — 2024
- Improving Steering Vectors by Targeting Sparse Autoencoder Features — 2024
- Can sparse autoencoders be used to decompose and interpret steering vectors? — 2024
- In-Distribution Steering: Balancing Control and Coherence in LM Generation — 2025 (validity/coherence trade-off)
- Steering Language Models in Multi-Token Generation (Tense/Aspect) — 2025 (multi-token generation control)
- Activation Steering for Masked Diffusion Language Models — 2025
- LinEAS: End-to-end Learning of Activation Steering with a Distributional Loss — 2025
- Understanding (Un)Reliability of Steering Vectors; On the Non-Identifiability of Steering Vectors — 2025/2026 (reliability caveats)
- Analysing the Generalisation and Reliability of Steering Vectors — 2024
- Towards Understanding Steering Strength; Understanding Steering Strength — 2026 (dose-response)

Domain — protein/genomic language models (interp_db ∩ sciatlas_db):
- Steering Protein Language Models — 2025
- Interpreting and Steering Protein Language Models through Sparse Autoencoders — 2025
- (sciatlas_db) genomic foundation models, controllable DNA/protein design, secondary-structure
  assignment (DSSP), ORF validity, codon usage — see full payload.

## Web retrieval (Evo2 specifics)

- Genome modelling and design across all domains of life with Evo 2 — Nature 2026 /
  bioRxiv 2025.02.18.638918. Evo 2: biological foundation model, 7B and 40B params, single-nucleotide
  resolution, up to 1M-token context, StripedHyena2 (short-explicit / medium-regularized /
  long-implicit hyena operators). Enables design/generation of long synthetic sequences.
- "Interpreting Evo 2" (Goodfire) — SAEs trained on multiple Evo 2 layers (Transformer and
  StripedHyena); layer ~26 (a StripedHyena layer) carries the most biologically-relevant features;
  SAE features used for **annotation, discovery, and steering of sequence generations**.
- ArcInstitute/evo2 GitHub — official weights (incl. evo2_7b) + generation API.

Source URLs:
- https://www.nature.com/articles/s41586-026-10176-5
- https://www.biorxiv.org/content/10.1101/2025.02.18.638918.full.pdf
- https://www.goodfire.com/research/interpreting-evo-2
- https://github.com/arcinstitute/evo2
