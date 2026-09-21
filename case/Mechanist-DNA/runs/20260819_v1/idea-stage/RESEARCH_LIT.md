# Raw Literature Retrieval: SAE feature amplification steering of Evo2 to control α-helical content of generated DNA

**Date**: 2026-08-19
**Query**: SAE feature amplification steering of DNA language models (Evo2) to control protein secondary structure (α-helix content) of generated coding sequences; measuring secondary structure via ORF translation + structure prediction/DSSP
**Sources scanned**: mechanic-db (200 papers, registered tier) + WebSearch (5 formulations) + arXiv/web + local PDF (papers/Evo2.pdf). Zotero / Obsidian: not configured (skipped). Semantic-Scholar / DeepXiv / Exa: not requested.
**Query formulations used**:
- Evo 2 DNA language model sparse autoencoder SAE interpretable features Arc Institute Goodfire
- sparse autoencoder feature steering genomic language model DNA generation control biological property
- Evo2 SAE features alpha-helix beta-sheet protein secondary structure coding sequence steering generation
- measuring alpha-helix content generated DNA ORF translation ESMFold DSSP secondary structure prediction
- mechanic-db decomposed query (SAE interpretability + feature steering of genomic/protein LMs + secondary-structure eval)

---

## Retrieved Papers

### Paper 1: Genome modeling and design across all domains of life with Evo 2
- **Authors**: Brixi, Durrant, Ku, Poli, Hie et al. (Arc Institute / NVIDIA / Goodfire)
- **Year**: 2025 (biorxiv 2025.02.18.638918); published Nature 652, 30 April 2026 (s41586-026-10176-5)
- **Venue**: Nature
- **Source**: local PDF (papers/Evo2.pdf) + WebSearch
- **Identifier**: doi:10.1101/2025.02.18.638918 ; Nature s41586-026-10176-5
- **URL**: https://www.nature.com/articles/s41586-026-10176-5

**Key content (verbatim excerpts from the PDF and Arc/Goodfire materials)**:
- "Evo 2 latent representations capture a broad spectrum of biologically relevant signals" — BatchTopK SAEs trained on Evo 2 recover interpretable features: exon boundary features (f/1050, f/25666), prophage feature (f/19746), promoter motifs (70% of promoter-enriched HOCOMOCO v.12 motifs with q<0.01), and "SAE model-derived feature activations for α-helices, β-sheets, and tRNAs" appear in the E. coli genomic region encompassing the thrT and tufB genes.
- "These results demonstrate that an Evo 2 SAE learns features that transfer across species."
- SAE features "can be used for annotation, discovery and steering of sequence generations."
- Fig. panel: "Distribution of secondary structure from Evo 2-generated proteins compared to natural [genomes]" — the paper already measures secondary-structure distributions of generated sequences.
- Released: SAE models + visualizer (https://arcinstitute.org/tools/evo/evo-mech-interp) and Evo Designer generation tool.
- Architecture (from local config.yml): 32-layer StripedHyena-2, hidden_size 4096, vocab 512 (char-level nucleotide tokenizer), 262k context. Layer 26 is a hyena-medium (hcm) block.

### Paper 2: Goodfire/Evo-2-Layer-26-Mixed (SAE model card)
- **Authors**: Goodfire (in collaboration with Arc Institute)
- **Year**: 2025
- **Venue**: HuggingFace model repository
- **Source**: WebSearch + WebFetch + local checkpoint inspection
- **URL**: https://huggingface.co/Goodfire/Evo-2-Layer-26-Mixed

**Content**: "BatchTopK sparse autoencoders", training data "mixed prokaryote/eukaryote", hooked at Evo 2 layer 26. Local checkpoint `sae-layer26-mixed-expansion_8-k_64.pt` tensors: `W (4096, 32768)`, `b_enc (32768,)`, `b_dec (4096,)` → tied-weight TopK SAE, d_model=4096, d_sae=32768 (expansion 8), TopK k=64. This is the exact mechanism named in task.md.

### Paper 3: Interpreting and Steering Protein Language Models through Sparse Autoencoders (2025)
- **Source**: mechanic-db + WebSearch — arXiv 2502.09135
**Abstract (excerpt)**: "explores the application of sparse autoencoders (SAE) to interpret the internal representations of protein language models, specifically ... ESM-2 8M ... By performing a statistical analysis on each latent component's relevance to distinct protein annotations, we identify potential interpretations linked to various protein characteristics, including transmembrane regions, binding sites, and specialized motifs. We then leverage these insights to guide sequence [generation]." — Direct methodological analogue in protein space: identify annotation-selective latents by statistical association, then steer. Establishes the amplification-steering recipe we transfer to DNA/Evo2.

### Paper 4: InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders (2024)
- **Source**: mechanic-db
**Abstract (excerpt)**: "training SAEs on embeddings from the PLM ESM-2, we identify up to 2,548 human-interpretable latent features per layer that strongly correlate with up to 143 known biological concepts such as binding sites, structural motifs, and functional domains. In contrast, examining individual neurons ... reveals up to 46 neurons per layer with clear conceptual alignment." — Establishes SAE >> raw neurons for concept selectivity; motivates the feature-selectivity confirmation step (F1 / correlation-based selection) that opens our plan.

### Paper 5: Sparse Autoencoders Reveal Interpretable Structure in Small Gene Language Models (2025)
- **Source**: mechanic-db + WebSearch — arXiv 2507.07486
**Abstract (excerpt)**: "SAEs have been used to analyze genomics-focused models such as Evo 2, identifying interpretable features in gene sequences. However, it remains unclear whether SAEs can extract meaningful representations from small gene language models." — Confirms Evo2-SAE feature discovery as established prior art; small-GLM extension is the frontier.

### Paper 6: Decode-gLM: Tools to Interpret, Audit, and Steer Genomic Language Models (2025)
- **Source**: WebSearch — biorxiv 2025.10.31.685860
**Content**: Framework for interpreting, auditing, and steering genomic LMs via latent intervention; reports latents activating selectively on puromycin-resistance genes and CMV enhancers, used to steer generation. Closest concurrent "steer a genomic LM via latent features" work.

### Paper 7: Concept Bottleneck Language Models for Protein Design (CB-pLM) (2024)
- **Source**: mechanic-db
**Abstract (excerpt)**: "a generative masked language model with a layer where each neuron corresponds to an interpretable concept ... intervene on concept values to precisely control the properties of generated proteins, achieving a 3 times larger change in desired concept values compared to baselines." — Establishes concept-amplification → property-control causal expectation (monotone dose–response), a design reference for our coefficient sweep.

### Paper 8: SAE-RNA: A Sparse Autoencoder Model for Interpreting RNA Language Model Representations (2025)
- **Source**: mechanic-db
**Abstract (excerpt)**: Maps RiNALMo representations to known biological features; concept discovery in pretrained embeddings without retraining. — Sibling biomolecular-LM SAE interpretability; supports cross-molecule generality of the recipe.

### Paper 9: Sparse Autoencoders for Low-N Protein Function Prediction and Design (2025)
- **Source**: mechanic-db
**Abstract (excerpt)**: Evaluates SAEs on fine-tuned ESM2 embeddings for low-N function prediction and design; SAE latents "capture structural and functional features." — Evidence SAE latents carry structure signal usable for design.

### Paper 10: Steered Generation via Gradient Descent on Sparse Features (2025)
- **Source**: WebSearch — arXiv 2502.18644
**Content**: Optimizes over sparse SAE features to steer generation. Alternative steering-mechanics reference (gradient over features vs. our fixed-coefficient amplification during autoregressive decoding).

### Supporting method references
- **Scaling and evaluating sparse autoencoders** (2024, OpenAI; TopK SAEs) and **BatchTopK Sparse Autoencoders** — the SAE family used here (cited in Evo2 paper refs).
- **Sparse Autoencoders Find Highly Interpretable Features in Language Models** (Cunningham et al. 2023) — foundational SAE-interpretability.
- **DSSP** (Kabsch & Sander) — standard secondary-structure assignment; "H" = α-helix (≥4-residue 4-turn helix); yields per-residue helix/sheet/coil → % helix. ESMFold ~60× faster than AlphaFold2 for large-scale structure prediction of translated ORFs.
