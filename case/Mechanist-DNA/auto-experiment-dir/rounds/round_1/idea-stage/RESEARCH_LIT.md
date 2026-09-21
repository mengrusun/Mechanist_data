# Research Literature — Raw Retrieval Dump (audit-only)

**Direction**: Validate that Evo2-7B Layer-26 SAE features selectively respond to protein α-helix, then causally steer/amplify those features during autoregressive DNA generation to raise α-helical content of the encoded protein (dose-response), verified by protein structure prediction + DSSP.
**Date**: 2026-07-18
**Channels queried**: cloud mechanic-db SEARCH service (mandatory, executed), web search. Zotero / Obsidian / local PDF channels: not configured in this project (only `papers/Evo2.pdf`, `papers/InterPLM.pdf` present locally — summarized separately).
**Raw cloud result JSON**: `idea-stage/mechanic_db_results.json` (300 papers, cross-domain: interp_db + sciatlas_db).

---

## Channel 1 — Cloud mechanic-db SEARCH (decomposed, cross-domain)

Query decomposition: sub-query A → `interp_db` (SAE features on genomic/protein LMs, concept-feature alignment, activation steering / feature amplification, dose-response); sub-query B → `sciatlas_db` (DSSP secondary structure, CDS↔codon alignment, ESMFold/AlphaFold, α-helix content). 300 papers returned. Most-relevant hits:

| # | Title | Year | Relevance |
|---|-------|------|-----------|
| 1 | Interpreting and Steering Protein Language Models through Sparse Autoencoders | 2025 | **Closest method analog.** SAE on ESM-2; intervene on latent components to steer generation toward structural motifs (transmembrane, binding sites, zinc-finger). In-modality (protein-LM → protein seq). |
| 2 | InterPLM: Discovering Interpretable Features in Protein Language Models via SAEs | 2024 | **Reference paper.** SAE on ESM-2; up to 2,548 interpretable features/layer vs ≤46 neurons; features↔143 biological concepts; concept-feature correlation; enables annotation-fill + steering. |
| 3 | ProtSAE: Disentangling and Interpreting PLMs via Semantically-Guided SAEs | 2025 | Semantically-guided SAE for cleaner concept features in PLMs. |
| 4 | Sparse Autoencoders Reveal Interpretable Structure in Small Gene Language Models | 2025 | Genomic-LM SAEs; explicitly notes "SAEs have been used to analyze genomics-focused models such as Evo 2, identifying interpretable features"; nucleotide-identity + TF-binding-motif features. |
| 5 | Sparse Autoencoders for Low-N Protein Function Prediction and Design | 2025 | Latent steering of SAE features designs top-fitness variants in 83% of cases; SAE latents encode interpretable structural/functional motifs. |
| 6 | Concept Bottleneck Language Models for Protein Design | 2024 | Concept-bottleneck control of protein generation — controllable-generation baseline framing. |
| 7 | Sparse Autoencoders Find Highly Interpretable Features in Language Models | 2023 | Foundational SAE-for-LM interpretability (dictionary learning, monosemanticity). |
| 8 | Mechanistic Interpretability of Antibody Language Models Using SAEs | 2025 | SAE features in a specialized biological LM. |
| 9 | Automated Neuron Labelling Enables Generative Steering and Interpretability in PLMs | 2025 | Neuron-activation-guided steering to generate proteins with targeted secondary/tertiary structural motifs; scales labeling to all neurons. |
| 11 | SAE-RNA: A Sparse Autoencoder Model for Interpreting RNA Language Model Representations | 2025 | SAE features localize to RNA secondary-structure regions (stems, hairpins). Cross-check for structural-feature specificity methodology. |
| 13 | CorrSteer: Generation-Time LLM Steering via Correlated SAE Features | 2025 | Selects steering features by correlating generation-time SAE activations with task outcome; inference-time only; reports side-effect (specificity) reduction. Relevant for feature-selection + specificity control. |
| 16 | Which SAE Features Are Real? Model-X Knockoffs for FDR Control | 2025 | Statistical control for false-discovery among "concept-selective" SAE features — relevant to M0 rigor (don't over-claim a feature set). |
| 19 | Emergence of Biological Structural Discovery in General-Purpose LMs | 2026 | Cross-modal transfer (syntax → protein structure) framing. |
| 27 | Measuring and Guiding Monosemanticity | 2025 | Metric for monosemanticity/steering quality. |
| 25 | Transformers trained on proteins can learn to attend to Euclidean distance | 2025 | Evidence PLMs encode structure geometry internally. |

Full 300-row list in `mechanic_db_results.json`.

---

## Channel 2 — Web search (frontier / recency check)

- **Evo 2 paper (Arc Institute + NVIDIA; Nature 2026 / bioRxiv 2025.02.18.638918)** — 7B & 40B genomic FM, single-nucleotide resolution, 1 Mb context, ~9.3T nt training. SAEs trained on Evo 2 activations **without biological labels** recover features for **intron/exon boundaries, TF motifs, CDS/UTR, and protein secondary structure (α-helices, β-sheets), RNA stem-loops, prophage/CRISPR elements**. The paper explicitly states feature **steering to guide DNA generation is promising but "still in its early stages."** → the *causal, dose-response* control targeted by this task is the open gap.
- **Goodfire "Interpreting Evo 2" (Arc × Goodfire collaboration)** — companion interpretability write-up; the released **Layer-26 mixed prokaryote/eukaryote SAE** (HF: `Goodfire/Evo-2-Layer-26-Mixed`, BatchTopK, expansion 8, k 64) is the exact artifact in `/data1/share_model/evo2/evo2_sae_layer26_mixed`. Shows α-helix / β-sheet / tRNA feature activation maps — **correlational** evidence only.
- **FoldSAE: Learning to Steer Protein Folding Through Sparse Representations (arXiv 2511.22519, 2025)** — steering protein folding via sparse features (protein-space analog).
- **Steered Generation via Gradient-Based Optimization on Sparse Query Features (arXiv 2605.23040)** — alternative steering-optimization framing.

**Sources**: [Evo 2 bioRxiv](https://www.biorxiv.org/content/10.1101/2025.02.18.638918v1.full), [Evo 2 Nature](https://www.nature.com/articles/s41586-026-10176-5), [Goodfire: Interpreting Evo 2](https://www.goodfire.ai/research/interpreting-evo-2), [Goodfire/Evo-2-Layer-26-Mixed](https://huggingface.co/Goodfire/Evo-2-Layer-26-Mixed), [FoldSAE](https://arxiv.org/pdf/2511.22519), [arcinstitute/evo2](https://github.com/arcinstitute/evo2).

---

## Reference-paper deep summaries
See `idea-stage/REF_PAPER_SUMMARY.md` (Evo2 + InterPLM methodology extraction used to author the M0 data methodology and the steering/verification protocol).
