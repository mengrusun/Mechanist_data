# Literature Landscape — Steering Evo2-7B toward higher α-helical content

**Date**: 2026-08-19
**Direction**: Generate DNA sequences with higher α-helical content using Evo2-7B (behavior-source: given; mechanism: discovery)
**Retrieval sources this run**: WebSearch (web + arXiv/bioRxiv surface); mechanic-db cloud SEARCH attempted via `/mechanic-db-search` but the `search_papers` MCP tool is not exposed in this isolated agent context (no `mcp__mechanic-db__*` tool, no project `.mcp.json`) — per the skill's graceful-degradation contract, proceeded with the remaining sources. arXiv PDF download not attempted (ARXIV_DOWNLOAD=false).

## 1. The target model — Evo 2 (Arc Institute)

- **Evo 2** is an autoregressive DNA foundation model at single-nucleotide resolution, StripedHyena-2 architecture, released 2025; sizes **7B** and 40B; context up to 1M bp. The 7B model (`arcinstitute/evo2_7b` on HuggingFace) was trained on OpenGenome2 (~2.4T tokens for 7B). It is a **generalist generative model across DNA/RNA/protein-coding regions** and exposes standard autoregressive sampling controls (temperature, top-k, top-p) for de novo sequence design. (Nature 2026; Arc Institute manuscripts; ArcInstitute/evo2 GitHub.)
- Generation mode extends a DNA prompt autoregressively; generated sequences contain coding + noncoding elements and can produce coherent genes maintaining structural/sequence similarity to natural ones. Inference-time guidance has been used to steer long-sequence generation toward controllable chromatin accessibility — precedent that **inference-time control of Evo 2 generation is feasible**.

## 2. Interpretability of Evo 2 — where α-helical structure lives (KEY)

- **Goodfire × Arc "Interpreting Evo 2"** trained **BatchTopK sparse autoencoders (SAEs)** on Evo 2 activations and found individual features aligning with **phage insertions, operon/exon–intron structure, and protein secondary structure — explicitly α-helices, β-sheets, and tRNAs**. SAEs were trained on **Layer 26 (a StripedHyena layer)**, chosen for the richest biologically-relevant features; the α-helix / β-sheet features were **validated against AlphaFold3** predictions in an E. coli region. Public artifact: `Goodfire/Evo-2-Layer-26-Mixed` (HuggingFace).
- **Critical open gap**: Goodfire report only **"early signs of life" on steering Evo 2 to engineer protein structures**, and state steering Evo 2 is **"considerably more complex than steering a language model"** and that **"further research is needed."** So the α-helix concept is **located** but **causal, dose-controlled steering of generated α-helix content is underdeveloped** — the natural mechanism-discovery opening.
- Related SAE-on-bio-model work: SAEs reveal interpretable structure in small gene language models (arXiv 2507.07486); SAEs uncover biologically interpretable features in **protein** LM representations (PNAS 2025); InterPLM; FoldSAE ("Learning to Steer Protein Folding Through Sparse Representations", arXiv 2511.22519) — steering protein folding via sparse features (protein-LM analogue of our goal).

## 3. Controllable generation / activation steering for bio sequence models

- **ARCADE: Controllable Codon Design from Foundation Models via Activation Engineering** (bioRxiv 2025.08.19.668819) — training-free **activation engineering / steering vectors** applied to (genomic/coding) foundation models to control codon usage at inference; establishes the steering-vector recipe (derive from contrasting sets, perturb activations during generation) for this exact model class.
- **Steering Protein Language Models** (arXiv 2509.07983) — derive steering vectors from **contrasting protein sets** and perturb PLM activations at inference for **training-free, precise control** of generated properties. Direct methodological template for a contrastive high-α vs low-α direction.
- **Steering Vector Fields for Property-Controlled Molecular Generation** (bioRxiv 2025.09.24) — property-controlled generation with chemical LMs via steering fields.
- **Language Models for Controllable DNA Sequence Design** (arXiv 2507.19523) and **ATGC-Gen** — controllable DNA generation, mostly via conditioning/cross-modal encoding or encoder-only gLMs; note that decoder-only gLMs (like Evo 2) that emit raw sequence are the ones where activation-level control applies cleanly.
- **Concept Activation Vectors** (AAAI) — CAV-style linear-concept control, general template for a linear α-helix concept direction.

## 4. Evaluating α-helical content of generated DNA (metric design)

- Generated DNA has no ground-truth structure, so α-helix content must be **predicted** from the translated protein: find ORF → translate CDS to amino acids → predict secondary structure → fraction of residues assigned α-helix (H).
- **DSSP** is the reference SS assignment (8-state; H/G/I → helix in the 3-state reduction). Structure-based route: fold generated protein with **ESMFold / AlphaFold2** → run **DSSP** → % helix. Rigorous but heavier.
- **Sequence-based SS predictors** (NetSurfP-3.0, SPOT-1D-LM, S4PRED, DML-SS) predict per-residue H/E/C directly from sequence using protein-LM embeddings — fast primary metric. Deep-learning models can predict/design α-helix vs β-sheet content with good agreement to DSSP-labeled PDB data.
- Recommended: **fast sequence-based %H as the primary online metric**, **ESMFold+DSSP on a held-out subset** as the structure-grounded confirmation.

## 5. Structural gaps → where a new contribution sits

1. **No published causal, dose-controlled steering of Evo 2 generation toward a specific protein secondary-structure target.** The α-helix feature is located (Goodfire) but not causally exploited for generation.
2. **Method comparison is open**: SAE-feature clamping (Goodfire layer-26 features) vs contrastive mean-difference steering vectors (ARCADE / Steering-PLM style) vs linear-probe directions — which localizes and steers α-helix best, and at which layer/site, is unknown for Evo 2.
3. **Specificity is unaddressed**: whether raising α-helix content can be done without collapsing sequence validity / coding-ness or trivially trading off β-sheet content.
4. **Evaluation for generated DNA α-helix content is not standardized** (ORF→translate→SS-predict→%H, with structure-based confirmation).

## 6. Implications for this project (given behavior + discovery mechanism)

- The behavior ("Evo2-7B can produce DNA whose protein has higher α-helical content") is **taken as given**; the work's job is to find and causally exploit the **internal component** that controls it.
- Mechanism strategy indicated by the landscape: **Location → Causal Intervention → Tuning & Editing** — locate an α-helix direction/feature (SAE feature, contrastive vector, or probe direction), causally confirm it via a steering dose-response with specificity controls, then use it to generate higher-α-helix DNA. Concrete family (SAE clamp vs steering vector vs probe direction) is left to the experiment-stage `/mechanism-skills` routing.
- Model is pinned to **Evo2-7B** (HARD constraint). 8×A800-80GB comfortably runs 7B inference on short coding windows, so plan at full model scale.
