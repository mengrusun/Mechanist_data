# Raw Literature Retrieval: pLDDT-aware structural evaluation + multi-predictor robustness for generated-protein secondary-structure claims (Evo2 SAE α-helix steering, Round 2 rigor)

**Date**: 2026-07-19
**Query**: Fold predicted-structure confidence (pLDDT) into secondary-structure statistics (not just a gate); multi-predictor (ESMFold/OmegaFold/ColabFold-AF2) robustness of DSSP secondary-structure readouts; supporting SAE/activation steering dose-response, Evo2 genomic-LM protein-structure control, DSSP assignment robustness.
**Sources scanned**: mechanic-db (cloud SEARCH, 300 papers) ✔ | arXiv API ✔ | WebSearch ✔ | local PDFs (papers/) ✔ (Evo2, InterPLM) | Zotero ✘ (not configured) | Obsidian ✘ (not configured) | Semantic Scholar ✘ (not requested) | Exa ✘ | DeepXiv ✘
**ARXIV_DOWNLOAD**: false (metadata only, no PDFs fetched)

**Query formulations used**:
- (mechanic-db, temporal=recent, top_k=300) "pLDDT structure-prediction-confidence-aware evaluation of generated/designed proteins: confidence-weighted DSSP metrics, conditioning/stratifying on pLDDT, confidence-threshold sensitivity, pLDDT confounding a structural effect; multi-predictor robustness ESMFold vs OmegaFold vs ColabFold/AlphaFold2 (DSSP agreement); SAE feature steering / activation steering dose-response; Evo2 genomic LM protein α-helix control; DSSP assignment robustness."
- (web) "pLDDT confidence weighted secondary structure evaluation generated proteins DSSP confounding"
- (web) "ESMFold OmegaFold AlphaFold2 agreement secondary structure DSSP multiple predictors robustness comparison"
- (web) "pLDDT as confounder structure prediction confidence bias protein design evaluation metric"
- (web) "sparse autoencoder feature steering dose response language model activation steering coefficient sweep"
- (web) "Evo2 genomic language model protein secondary structure steering sparse autoencoder alpha helix"
- (web) "generative protein model evaluation alpha helix fraction pLDDT filter threshold designability self-consistency scTM RMSD"
- (web) "DSSP secondary structure assignment robustness STRIDE DSSP disagreement helix definition sensitivity"
- (arXiv API) "pLDDT confidence protein structure prediction evaluation design"; "ESMFold OmegaFold AlphaFold secondary structure comparison protein"; "sparse autoencoder steering protein language model secondary structure"

---

## A. Local PDFs (papers/ — background/given, read first pages)

### Paper A1: Genome modelling and design across all domains of life with Evo 2
- **Authors**: Brixi, Durrant, Ku, Naghipourfar, Poli, ... Goodarzi, Hsu, Hie et al.
- **Year**: 2026 (Nature 652:1349) — **Source**: local PDF (papers/Evo2.pdf)
- **Abstract (verbatim, opening)**: "All of life encodes information with DNA... Here we introduce Evo 2, a biological foundation model trained on 9 trillion DNA base pairs from a highly curated genomic atlas spanning all domains of life to have a 1 million token context window with single-nucleotide resolution. Evo 2 learns to accurately predict the functional impacts of genetic variation... Mechanistic interpretability analyses reveal that Evo 2 learns representations associated with biological features, including exon–intron boundaries, transcription factor binding sites, protein structural elements and prophage genomic regions. The generative abilities of Evo 2 produce mitochondrial, prokaryotic and eukaryotic sequences at genome scale with greater naturalness and coherence than previous methods... We have made Evo 2 fully open, including model parameters, training code, inference code and the OpenGenome2 dataset."
- **Relevance**: The base model of this project (Evo2-7B). Establishes that Evo 2 internally represents protein structural elements — the substrate for Layer-26 SAE α-helix features.

### Paper A2: InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders
- **Authors**: Elana Simon, James Zou (Stanford) — **Year**: 2024 (arXiv 2412.12101) — **Source**: local PDF (papers/InterPLM.pdf) + mechanic-db
- **Abstract (verbatim)**: "Protein language models (PLMs) have demonstrated remarkable success... yet their internal mechanisms for predicting structure and function remain poorly understood. Here we present a systematic approach to extract and analyze interpretable features from PLMs using sparse autoencoders (SAEs). By training SAEs on embeddings from the PLM ESM-2, we identify up to 2,548 human-interpretable latent features per layer that strongly correlate with up to 143 known biological concepts such as binding sites, structural motifs, and functional domains. In contrast, examining individual neurons in ESM-2 reveals up to 46 neurons per layer with clear conceptual alignment across 15 known concepts, suggesting that PLMs represent most concepts in superposition... we demonstrate how these latent features can fill missing annotations in protein databases and enable targeted steering of protein sequence generation."
- **Relevance**: Methodological template for SAE feature discovery + steering in a biological sequence model; supports the "distributed, set-level structural code (superposition)" framing used for the α-helix feature set S.

---

## B. Structure-prediction confidence / pLDDT-in-statistics (Focus 1)

### Paper B1: Random, de novo, and conserved proteins: How structure and disorder predictors perform differently
- **Year**: 2024 — **Venue**: Proteins: Structure, Function, and Bioinformatics (Wiley, prot.26652) — **Source**: WebSearch
- **Key content (verbatim from retrieval)**: Examines Spearman correlations of pLDDT with secondary elements (DSSP) and disorder. "AF2 predictions of de novo proteins have higher pLDDT values than those of random sequences despite being predicted to be more disordered" — a direct confounding signal. Warns that "using DSSP... to determine disordered regions leads to a dramatic overestimation in disorder content and represents a potential pitfall," recommending pLDDT (correlated to prediction confidence) as the better order/disorder metric. Kernel-density of pLDDT is strongly left-skewed for most SS classes except coils; β-bridge > helical > H-bond-stabilized turns for high-pLDDT propensity.
- **Relevance**: DIRECT evidence that pLDDT co-varies with the target's structural class and sequence origin (natural vs de novo/random) — i.e. exactly the pLDDT→SS confound the round must rule out via conditioning/stratification.

### Paper B2: Evaluating zero-shot prediction of monomeric protein design success by AlphaFold, ESMFold, and ProteinMPNN
- **Year**: 2025/2026 — **Venue**: bioRxiv 2025.07.29.667290 (also PMC12817478) — **Source**: WebSearch
- **Key content**: pLDDT provides little discrimination between experimentally stable and unstable designs; many de novo designs achieve highly confident model metrics yet fail to express/fold. Cross-checks AF vs ESMFold vs ProteinMPNN self-consistency.
- **Relevance**: Motivates NOT treating pLDDT as ground truth and folding it into statistics with caution; motivates multi-predictor + threshold sensitivity rather than a single confident metric.

### Paper B3: pLDDT Values in AlphaFold2 Protein Models Are Unrelated to Globular Protein Local Flexibility
- **Year**: 2023 — **Venue**: Crystals 13(11):1560 (MDPI), 35 citations — **Source**: mechanic-db
- **Relevance**: pLDDT is a prediction-confidence signal, not a physical-flexibility/order readout for globular regions — caution against over-interpreting per-residue pLDDT as a structural property; supports treating pLDDT as a covariate/nuisance rather than a direct structural label.

### Paper B4: Categorizing prediction modes within low-pLDDT regions of AlphaFold2 structures
- **Year**: 2025 — **Venue**: bioRxiv 2025.06.06.658382 — **Source**: mechanic-db
- **Relevance**: Low-pLDDT regions are heterogeneous (not uniformly "disordered"); supports pLDDT-stratified analysis and threshold sensitivity rather than one hard gate.

### Paper B5: Fold or flop: quality assessment of AlphaFold predictions on whole proteomes
- **Year**: 2025 — **Venue**: bioRxiv 2025.12.19.695427 — **Source**: mechanic-db
- **Relevance**: Proteome-scale pLDDT-based quality assessment; per-structure confidence distributions and gating practice at scale — informs how to report mean pLDDT per dose.

### Paper B6: pLDDT-Predictor: High-speed Protein Screening Using Transformer and ESM2
- **Year**: 2024 — **Venue**: arXiv 2410.21283 — **Source**: arXiv API
- **Key content**: Predicts pLDDT directly from sequence via ESM2 for high-throughput screening; notes label bias — predictors trained on AF2 pLDDT inherit AF2 calibration biases; pLDDT reports LOCAL confidence, does not assess global topology.
- **Relevance**: Reinforces pLDDT limitations that a confidence-weighted metric must account for.

### Paper B7: CONFIDE: Hallucination Assessment for Reliable Biomolecular Structure Prediction and Design
- **Year**: 2025 — **Venue**: arXiv 2512.02033 — **Source**: WebSearch + arXiv API
- **Relevance**: Frames high-confidence-but-wrong predictions as "hallucinations"; argues for confidence assessment beyond raw pLDDT in generation/design — supports sensitivity analyses and not gating on pLDDT alone.

### Paper B8: DISPROTBENCH: Uncovering the Functional Limits of Protein Structure Prediction Models in Intrinsically Disordered Regions
- **Year**: 2025 — **Venue**: arXiv 2507.02883 — **Source**: WebSearch
- **Relevance**: Benchmarks structure predictors in disordered regions where pLDDT is systematically low; DSSP-vs-confidence interplay in low-confidence regimes.

### Paper B9: Confidence Without Verification: Screening pLDDT Unreliability in AlphaFold2 Fold-Switching Predictions
- **Year**: 2026 — **Venue**: bioRxiv 2026.02.19.706878 — **Source**: WebSearch
- **Relevance**: pLDDT can be confidently wrong for fold-switchers; a stress case for confidence-weighted SS metrics.

### Paper B10 (methodology, generative-protein evaluation standards): designability / self-consistency conventions
- **Sources**: "An all-atom protein generative model" (PNAS 2024, PMC11228509); ProteinBench (arXiv 2409.06744); "Generative Modeling in Protein Design: ... Evaluation Standards" (arXiv 2603.26378); SHAPES (PMC11761634).
- **Key content (verbatim from retrieval)**: Standard designability = **self-consistency RMSD (scRMSD) < 2 Å AND pLDDT > 70**; scTM variant uses self-consistency TM > 0.5 (e.g. Genie 2). Per-residue pLDDT, pTM, PAE used as **filters** to discard low-quality candidates. Folding-oracle scRMSD folds the designed sequence and compares to the intended backbone. Filtering trims low-quality tails of scRMSD/pLDDT distributions.
- **Relevance**: Establishes the field-standard pLDDT>70 gate this project already uses; the round's contribution (confidence-WEIGHTED SS + pLDDT stratification + threshold sweep) is a rigor step BEYOND this hard-gate convention.

---

## C. Multi-predictor robustness (Focus 2)

### Paper C1: Balancing speed and precision in protein folding: a comparison of AlphaFold2, ESMFold, and OmegaFold
- **Year**: 2025 — **Venue**: Frontiers in Genetics 2025:1715037 (also bioRxiv 2025.06.20.660709; PubMed 41608648) — **Source**: WebSearch + WebFetch
- **Key content (verbatim from retrieval + fetch)**: Dataset of 1,300+ PDB structures resolved July 2022–July 2024 (post-training, unbiased). Median pLDDT: **AF2 92.65, OmegaFold 89.00, ESMFold 87.40**. Incorrect-prediction rate: AF2 8.9% < ESMFold 13.0% < OmegaFold 16.8%; **failure overlap between tools is LIMITED (complementary strengths)**. On **de novo designed proteins**, ESMFold and OmegaFold achieve significantly lower RMSD; AF2 shows lower TM-scores — alignment-free LM predictors excel where evolutionary homology is scarce. DSSP was run on AF2 and ESMFold models (ASA + SS extracted); a *separate* related analysis reported **no significant difference between α-helix and β-sheet fractions of ESMFold vs AF2**. (Note: the Frontiers article itself reports RMSD/TM/GDT-TS/pLDDT, not DSSP SS fractions.)
- **Relevance**: PRIMARY reference for the round's second-predictor axis. Key operational takeaways: (i) predictors DISAGREE most on designed/low-homology sequences — exactly the regime of Evo2-generated proteins; (ii) failure sets are complementary, so cross-predictor agreement is a meaningful robustness test; (iii) a second predictor changes the pLDDT scale (medians differ by ~3–5 points), so any pLDDT gate/weight must be predictor-specific.

### Paper C2: Evaluation of the structural models of the human reference proteome: AlphaFold2 versus ESMFold
- **Year**: 2025 — **Venue**: Computational and Structural Biotechnology Reports / ScienceDirect S2665928X25000042 — **Source**: WebSearch
- **Relevance**: Proteome-scale AF2-vs-ESMFold agreement/disagreement; systematic confidence and model-quality differences at scale.

### Paper C3: Proteins with alternative folds reveal blind spots in AlphaFold-based protein structure prediction
- **Year**: 2024 — **Venue**: arXiv 2410.14898 — **Source**: arXiv API
- **Relevance**: Single-predictor blind spots (fold-switching, alternative conformations) — a mechanism by which a single predictor's SS readout can be non-robust; argues for multi-predictor corroboration.

### Paper C4: Secondary structure assignment for conformationally irregular peptides: Comparison between DSSP, STRIDE and KAKSI
- **Year**: 2014 — **Venue**: Journal of Molecular Graphics and Modelling (ScienceDirect) — **Source**: WebSearch
- **Key content (verbatim from retrieval)**: DSSP vs STRIDE agreement ~81.0%; Segment-Overlap agreement ranges 49–70% for irregular peptides; major disagreement on turns, α vs 3-10 helix distinction, and short parallel-sheet segments. STRIDE (H-bonds + backbone dihedrals) tends to assign longer kinked helices; KAKSI assigns several short helices. DSSP is sensitive to the H-bond energy threshold; multi-threshold weighted averaging is possible.
- **Relevance**: The DSSP SS-assignment layer itself is a source of variance independent of the predictor. The α-vs-3-10 helix ambiguity directly affects "α-helix fraction." Supports the DSSP-swap robustness already done in Round 1 and motivates reporting which DSSP settings/definitions are used.

### Paper C5: Protein secondary structure assignment revisited: a detailed analysis of different assignment methods
- **Year**: 2005 — **Venue**: BMC Bioinformatics (PMC1249586) — **Source**: WebSearch
- **Relevance**: Foundational quantification of inter-method SS-assignment disagreement; baseline for how much SS-fraction variance is method-induced vs signal.

---

## D. Supporting: SAE / activation steering dose-response, Evo2/PLM structure control (Focus 3)

### Paper D1: Improving Steering Vectors by Targeting Sparse Autoencoder Features
- **Year**: 2024 — **Venue**: arXiv 2411.02193 — **Source**: mechanic-db
- **Abstract (verbatim, excerpt)**: "...steering methods attempt to ensure that outputs of the model satisfy specific pre-defined properties... it can be difficult to anticipate the effects of steering vectors produced by methods such as CAA or the direct use of SAE latents. In our work, we address this issue by using SAEs to measure the effects of steering vectors, giving us a method that can be used to understand the causal effect of any steering vector intervention."
- **Relevance**: Best-practice for causal, measured SAE steering; supports σ_proj-unit dosing and quantifying effect per intervention.

### Paper D2: A Comparative Analysis of Sparse Autoencoder and Activation Difference in Language Model Steering
- **Year**: 2025 — **Venue**: arXiv 2510.01246 — **Source**: mechanic-db + WebSearch
- **Abstract (verbatim, excerpt)**: "...many dimensions among the top-k latents capture non-semantic features... we propose focusing on a single, most relevant SAE latent (top-1)... We further identify a limitation in constant SAE steering, which often produces degenerate outputs such as repetitive single words. To mitigate this, we introduce a token-wise decaying steering strategy..."
- **Relevance**: Documents dose degeneracy at high steering strength — direct support for mapping an INTERIOR dose-response plateau (Round-2 improvement #4) rather than a grid-edge optimum.

### Paper D3: Steering Protein Language Models
- **Year**: 2025 — **Venue**: arXiv 2509.07983 — **Source**: mechanic-db
- **Abstract (verbatim, excerpt)**: "...we investigate the potential of Activation Steering... to direct PLMs toward generating protein sequences with targeted properties. We propose a simple yet effective method that employs activation editing to steer PLM outputs, and extend this approach to protein optim[ization]..."
- **Relevance**: Closest analogue: activation steering for targeted protein properties; peer method for the causal-knob claim.

### Paper D4: Interpreting and Steering Protein Language Models through Sparse Autoencoders
- **Year**: 2025 — **Venue**: arXiv 2502.09135 — **Source**: mechanic-db + arXiv API + WebSearch
- **Abstract (verbatim, excerpt)**: "...application of sparse autoencoders (SAE) to interpret the internal representations of protein language models, specifically... ESM-2 8M... we identify potential interpretations linked to various protein characteristics, including transmembrane regions, binding sites, and specialized motifs. We then leverage these insights to guide sequence [generation]."
- **Relevance**: Direct methodological sibling — SAE features → interpretation → steering in a PLM.

### Paper D5: ProtSAE: Disentangling and Interpreting Protein Language Models via Semantically-Guided Sparse Autoencoders
- **Year**: 2025 — **Venue**: arXiv 2509.05309 — **Source**: mechanic-db
- **Abstract (verbatim, excerpt)**: "...SAE suffers from semantic entanglement, where individual neurons often mix multiple nonlinear concepts... we propose a semantically-guided SAE, called ProtSAE... we guide semantic disentanglement during training using both annotation datasets and domain [knowledge]."
- **Relevance**: Entanglement/monosemanticity caveats relevant to the "distributed set S vs single feature" framing.

### Paper D6: Automated Neuron Labelling Enables Generative Steering and Interpretability in Protein Language Models
- **Year**: 2025 — **Venue**: arXiv 2507.06458 — **Source**: mechanic-db + WebSearch
- **Abstract (verbatim, excerpt)**: "...first automated framework for labeling every neuron in a PLM... individual neurons are selectively sensitive to diverse biochemical and structural properties. We then develop a novel neuron activation-guided steering method to generate proteins with desired traits, enabling convergence to target biochemical properties..."
- **Relevance**: Steering-to-target-property with a dose/convergence notion; peer to the causal-knob dose-response.

### Paper D7: Measuring and Guiding Monosemanticity (Feature Monosemanticity Score)
- **Year**: 2025 — **Venue**: arXiv 2506.19382 — **Source**: mechanic-db
- **Relevance**: Quantifies feature isolation reliability; supports honest treatment of set-level (distributed) rather than single-feature claims (C1).

### Paper D8: Sparse Autoencoders for Low-N Protein Function Prediction and Design
- **Year**: 2025 — **Source**: mechanic-db
- **Relevance**: SAE latents of ESM2 for design in data-scarce regimes — corroborates SAE features carry structural/functional signal usable for control.

### Paper D9: Towards Interpretable Protein Structure Prediction with Sparse Autoencoders
- **Year**: 2025 — **Venue**: arXiv 2503.08764 — **Source**: WebSearch
- **Relevance**: SAEs applied to structure-prediction models; ties SAE features to structural readouts.

### Paper D10: Interpreting Evo 2 (Goodfire / Arc Institute)
- **Year**: 2025 — **Venue**: goodfire.ai research report — **Source**: WebSearch
- **Key content (verbatim from retrieval)**: SAEs trained on Evo 2 decompose it into sparse, human-interpretable latents associated with biological function usable for annotation, discovery, and **steering of sequence generation**; SAE features uncover structural motifs including **α-helices** and RNA stem-loops. Steering toward α-helices via semantic search ("alpha") from a neutral 75-residue sequence produced a protein with 5 α-helix domains spanning 34% of the protein; "steering this model is considerably more complex than steering a language model."
- **Relevance**: The most direct external precedent for THIS project's exact behavior (Evo2 SAE α-helix steering). Establishes prior art but at a demonstration level — the round's quantitative rigor (dose-response CIs, pLDDT-aware SS, multi-predictor) goes beyond it.

### Paper D11: Controllable protein design with particle-based Feynman-Kac steering
- **Year**: 2025 — **Venue**: arXiv 2511.09216 — **Source**: arXiv API
- **Relevance**: Alternative inference-time steering formulation for controllable protein generation; comparative context for the steering mechanism.

### Paper D12: Steering Large Language Model Activations in Sparse Spaces (Sparse Activation Steering, SAS)
- **Year**: 2025 — **Venue**: arXiv 2503.00177 — **Source**: WebSearch
- **Relevance**: General SAE-space steering with contrastive feature selection and clamp-value tuning (dose); side-effect/coefficient trade-off — supports capability-preservation checks in the dose sweep.

### Paper D13 (supporting SAE-eval): Pre-Intervention Prediction of Sparse Autoencoder Steering Side Effects (arXiv 2606.08365); Feature-Guided Activation Additions/FGAA (OpenReview swRxS7s4rB); Steering Language Model Refusal with SAEs (arXiv 2411.11296)
- **Relevance**: Collectively establish that (a) steering strength trades off target-effect vs side-effects/degeneracy, (b) effects should be measured not assumed — motivating the σ_proj-unit dosing, interior-plateau mapping, and capability-preservation controls of Round 2.
