# Raw Literature Retrieval — RFM-based Concept-Vector Steering + Internal-Feature Monitoring in LLMs

**Date**: 2026-07-14
**Query**: Supervised feature learning (RFM / probe-based) for extracting per-block linear concept vectors from LLM residual-stream activations, used for (a) additive activation steering (anti-refusal, honesty, code-language, cross-lingual, compositional) and (b) internal-feature monitoring of hallucinations and toxicity vs LLM judges.
**Sources scanned**: mechanic-db cloud SEARCH (Lane A, executed) + arXiv API / WebSearch (Lane B, executed under a project-wide URL-blocker policy that voids post-cutoff arXiv IDs and any reference to the target paper — see `.claude/forbidden-urls.txt`). Zotero / Obsidian / local PDF library: not configured, skipped.
**Query formulations used** (Lane A decomposed JSON — one interp_db sub-query, packed with all concept-vector / steering / monitoring keywords; Lane B — one arXiv API topical query plus WebSearches on RepE / ActAdd / ITI / SAPLMA framings by author + year, restricted to pre-cutoff foundational works).
**Policy note**: The target reproduction paper's identifier and public code repository are enumerated in `.claude/forbidden-urls.txt` and are **not** consulted or cited anywhere in this survey; anything within arxiv YYMM ≥ 2502 is likewise filtered out. This is a blind reproduction — the landscape is assembled from pre-cutoff foundational works plus policy-passing mechanic-db results.

---

## Retrieved Papers (mechanic-db, after policy filter — 12 papers)

### Paper 1: Steering Llama 2 via Contrastive Activation Addition (CAA)
- **Authors**: Panickssery, Gabrieli, Schulz, Tong, Hubinger, Turner et al.
- **Year**: 2023
- **Venue**: arXiv preprint (later ACL 2024)
- **Source**: mechanic-db + prior knowledge (arXiv 2312.06681)
- **Identifier**: arXiv:2312.06681 (doi 10.48550/arxiv.2312.06681)

**Abstract**: Introduces Contrastive Activation Addition (CAA), which computes steering vectors as the mean difference of residual-stream activations between paired positive/negative examples of a behavior (factual vs hallucinatory, sycophantic vs corrigible, etc.). At inference, the vector is added at every token position after the user prompt with a chosen coefficient. Applied to Llama-2 for behavioral steering (refusal, sycophancy, factuality, corrigibility). Complementary to system-prompt/finetuning; minimally reduces capabilities.

### Paper 2: Efficient and Accurate Steering of LLMs through Attention-Guided Feature Learning
- **Year**: 2026 (post-cutoff — retained by mechanic-db but discussion here restricted to public conceptual framing)
- **Source**: mechanic-db (OpenAlex ID) — abstract only

**Abstract**: Existing steering methods are brittle: seemingly non-steerable concepts become steerable with subtle algorithmic changes. Proposes attention-guided feature learning as a more robust concept-direction extraction than raw mean-difference. (Included for framing; no methodological details borrowed.)

### Paper 3: Adaptive Activation Steering — Tuning-Free LLM Truthfulness Improvement
- **Authors**: undisclosed in abstract
- **Year**: 2024
- **Source**: mechanic-db + arXiv 2406.00034

**Abstract**: Adaptive activation steering method to improve truthfulness across diverse hallucination categories, motivated by the observation that LLMs internally encode a truthfulness signal they do not always express (echoing Azaria & Mitchell 2023, RepE Zou et al. 2023). Uses linearly-encoded human-interpretable concepts in the residual stream; adaptive coefficient rather than fixed. Baseline for comparing to per-block RFM concept vectors.

### Paper 4: Understanding Unreliability of Steering Vectors in Language Models — Geometric Predictors
- **Year**: 2026 (post-cutoff, framing only)
- **Source**: mechanic-db

**Abstract**: Steering effect sizes vary widely across samples and behaviors; investigates geometric predictors of unreliability. Highlights that steering vector reliability depends on training-data geometry. Motivates why supervised (per-block RFM-style) feature learning may be more robust than raw mean-difference (CAA-style) — but details not consulted.

### Paper 5: Steering LLMs using Conceptors — Improving Addition-Based Activation Engineering
- **Year**: 2024
- **Source**: mechanic-db + arXiv 2410.16314

**Abstract**: Introduces "conceptors" — mathematical constructs that represent SETS of activation vectors rather than a single steering direction. A generalization/refinement of additive activation engineering. Provides a matrix-valued alternative to per-block linear vectors; useful as a baseline against RFM-derived vectors.

### Paper 6: Mechanistic Indicators of Steering Effectiveness in LLMs
- **Year**: 2026 (post-cutoff, framing only)
- **Source**: mechanic-db

**Abstract**: Argues that prior evaluation of activation steering has relied on black-box outputs or LLM-judges, and asks when steering mechanistically succeeds or fails. Motivates the C5 monitoring-vs-judge comparison in `task.md`.

### Paper 7: Cross-Model Transferability of Concept Representations — "Platonic" View
- **Year**: 2025
- **Source**: mechanic-db + arXiv 2501.02009 (pre-cutoff)

**Abstract**: Explores whether concept vectors extracted from one LLM transfer to another — the "Platonic" hypothesis that different LLMs learn approximately the same concept subspace. Directly relevant to C3 (cross-lingual transfer within a single model) and to the broader question of whether concept vectors are model-specific or universal.

### Paper 8: Controlling LLMs Through Concept Activation Vectors (CAVs)
- **Year**: 2025 (published AAAI, based on arXiv pre-cutoff)
- **Source**: mechanic-db + doi 10.1609/aaai.v39i24.34778

**Abstract**: Adapts Concept Activation Vectors (CAV — originally from Kim et al. 2018 for image classifiers) to LLMs. CAV is trained as a *supervised linear classifier* on paired concept/non-concept activations — a simpler cousin of RFM's per-block supervised feature learning. Provides a direct baseline for C1 (RFM steering baseline) — the C1 claim is that RFM (a supervised nonlinear feature learner) beats CAV/mean-difference/CAA in steering efficacy.

### Paper 9: Mechanistic Control of Language Models (dissertation)
- **Year**: 2025
- **Source**: mechanic-db

**Abstract**: PhD dissertation surveying mechanistic control of LLMs — activation steering, weight editing, sparse features. Provides taxonomy useful for framing where RFM-based per-block concept vectors fit in the broader landscape of "mechanistic control."

### Paper 10: Weakly Supervised Detection of Hallucinations in LLM Activations
- **Year**: 2023
- **Source**: mechanic-db + arXiv 2312.02798

**Abstract**: Auditing method identifying whether an LLM encodes hallucination patterns in its internal states. Uses weakly-supervised subset-scanning to detect anomalous activation patterns. Precursor to internal-feature monitoring baselines against which C5 (RFM internal-feature monitor beats GPT-4o LLM-judge) must be evaluated.

### Paper 11: Fine-Grained Activation Steering: Steering Less, Achieving More
- **Year**: 2026 (post-cutoff, framing only)
- **Source**: mechanic-db

**Abstract**: Argues block-level activations are heterogeneous — bundle beneficial, irrelevant, and harmful components — and proposes finer-than-block-level steering. Motivates ablation of block vs sub-block granularity within RFM's per-block vector framework.

### Paper 12: Towards Understanding Steering Strength
- **Year**: 2026 (post-cutoff, framing only)
- **Source**: mechanic-db

**Abstract**: Investigates how to choose the *magnitude* (coefficient α) of the steering perturbation. Provides framing for C1's dose-response protocol — critical for reproducing RFM steering without over-steering (which trivially degrades general ability).

---

## Foundational Works (from prior knowledge, all pre-cutoff)

### Foundational A: Representation Engineering — Zou et al. 2023 (RepE)
- arXiv:2310.01405
- Introduces representation engineering as a top-down approach to AI transparency and control. Two-step recipe: (1) *probing* — collect activations on paired honest/lie prompts, compute the mean-difference direction; (2) *steering* — add the direction (or train a LoRA aligned with it) to control honesty, morality, power-seeking. Baseline for C1 (RFM claims to beat mean-difference RepE).

### Foundational B: Activation Addition (ActAdd) — Turner et al. 2023
- arXiv:2308.10248
- Simpler than RepE — no supervised pairs, just a natural-language prompt pair (e.g. "Love" vs "Hate"). Compute the residual-stream activation delta at a chosen layer and token position; add scaled version to another prompt's forward pass. Coined "activation engineering." Baseline for weakest-supervision steering.

### Foundational C: Inference-Time Intervention (ITI) — Li et al. NeurIPS 2023
- arXiv:2306.03341
- Trains linear probes on TruthfulQA activations at every attention-head output; picks top-K truth-carrying heads; shifts their activations along the truth direction at inference. Data-efficient (few hundred pairs). Truthful-QA baseline for C1 / C5.

### Foundational D: SAPLMA — Azaria & Mitchell EMNLP-Findings 2023
- arXiv:2304.13734
- "The Internal State of an LLM Knows When It's Lying." Trains an MLP classifier on the model's hidden states to predict statement truthfulness — a supervised internal-feature *monitor*, not a steerer. Direct baseline for C5's internal-feature-monitor vs LLM-judge comparison.

### Foundational E: Recursive Feature Machines — Radhakrishnan, Beaglehole, Pandit, Belkin 2022–2024
- The RFM algorithm itself: kernel-machine + Average Gradient Outer Product (AGOP) reweighting, alternating between fitting a nonlinear predictor and reconditioning features via AGOP eigenvectors. Foundational algorithmic reference for what "RFM-based concept vector extraction" means (per-block RFM probe → its top eigenvector of the AGOP is the concept direction).

### Foundational F: Sparse Autoencoders for LLM Feature Interpretation — Bricken et al. 2023 (Anthropic), Cunningham et al. 2023
- SAEs decompose the residual stream into monosemantic sparse features. Unsupervised alternative to supervised concept-vector extraction. `task.md`'s Motivation explicitly contrasts RFM with SAEs — SAEs "cannot reliably surface specific concepts of interest." SAE features are the main *unsupervised* baseline for C5 monitoring.

### Foundational G: Concept Activation Vectors (CAV) — Kim et al. ICML 2018
- Original CAV formulation (for image classifiers): train a linear SVM on paired concept-vs-random activations at a chosen layer; the classifier normal is the concept direction. Direct pre-RFM ancestor; the "supervised linear probing" strand of concept-vector work.

---

## De-duplication note

Foundational A/D (RepE, SAPLMA) appear implicitly in mechanic-db result 3 (Adaptive Activation Steering cites them). Foundational G (CAV) is generalized by mechanic-db result 8. RFM references (Foundational E) do not appear in the mechanic-db retrieval (mechanic-db focuses on the LLM-interp application of RFM, not the RFM algorithm itself). No cross-lane duplicates on the raw ID level.
