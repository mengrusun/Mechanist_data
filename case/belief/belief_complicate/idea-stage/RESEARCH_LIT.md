# Raw Literature Retrieval: Belief localization in pretrained language models

**Date**: 2026-07-10
**Query**: Fisher-information-matrix localization of belief-related attention heads in Pythia; causal zero-ablation; scale-dependent emergence of personal vs attributed belief; formation-window analysis over intermediate checkpoints; inference-time dynamic head amplification with a frame classifier.
**Sources scanned**: WebSearch (Google, arXiv titles surfaced), arXiv API (via WebSearch snippets). `mechanic-db` was **skipped** — the `search_papers` MCP tool is not exposed in this claim-agent's runtime tool set (server exists on disk but is not wired). Zotero / Obsidian / local PDFs — not configured / absent.
**Query formulations used**:
- `"Sensitivity Meets Sparsity" Fisher information theory of mind large language models`
- `theory of mind LLM belief attribution attention head localization interpretability`
- `Pythia intermediate checkpoints emergence developmental interpretability 2024 2025`
- `false belief task language model false-belief personal belief attributed belief false belief representation`
- `attention head zero ablation causal circuit discovery interpretability Pythia GPT-2`
- `activation steering dynamic amplification attention head inference-time control language model`
- `"belief representation" language model probing internal representation truthful world model`
- `indirect object identification IOI circuit GPT-2 attention head interpretability`
- `"Language Models Represent Beliefs of Self and Others" Zhu Chen ICML 2024`
- `Pythia theory of mind emergent abilities small language model scaling belief`
- `Fisher information matrix parameter localization neural network attention`
- `"induction head" emergence Pythia training checkpoint developmental circuit`
- `activation patching path patching mechanistic interpretability circuit analysis 2024`

---

## Retrieved Papers

### Paper 1: Sensitivity Meets Sparsity: The Impact of Extremely Sparse Parameter Patterns on Theory-of-Mind of Large Language Models
- **Authors**: (per arXiv 2504.04238)
- **Year**: 2025 (arXiv v1, April 2025); published in *npj Artificial Intelligence* (2025)
- **Venue**: arXiv preprint 2504.04238; npj Artif Intell (Nature)
- **Source**: WebSearch (arXiv + Nature + ResearchGate)
- **Identifier**: arXiv:2504.04238
- **URL**: https://arxiv.org/abs/2504.04238

**Abstract (paraphrased from snippets)**:
The paper investigates the mechanistic emergence of Theory-of-Mind (ToM) in LLMs via *extremely sparse parameter patterns*. It uses the **Fisher information matrix (FIM)** over model parameters to derive a binary mask isolating ToM-sensitive parameters, combined with a *language-modeling performance mask* to guarantee that perturbations impair ToM specifically without collapsing general LM ability. Findings: perturbing as little as 0.001% of the ToM-sensitive parameters significantly degrades ToM while also impairing contextual localization and language understanding. The sparse pattern is concentrated in W_Q and W_K matrices and links tightly to the positional-encoding (RoPE) module — perturbations disrupt the angle between queries and keys under rotary encoding, disrupting dominant-frequency activations critical for contextual belief processing. **This is the reference paper explicitly cited by `task.md`** and provides the methodological anchor for the Fisher-mask localization in Claim 2.

---

### Paper 2: Language Models Represent Beliefs of Self and Others (ICML 2024)
- **Authors**: Wentao Zhu, Zhining Zhang, Yizhou Wang
- **Year**: 2024
- **Venue**: ICML 2024 (PMLR 235:62638-62681)
- **Source**: WebSearch (arXiv, PMLR, GitHub)
- **Identifier**: arXiv:2402.18496
- **URL**: https://arxiv.org/abs/2402.18496 ; https://walter0807.github.io/RepBelief/ ; https://github.com/Walter0807/RepBelief

**Abstract (from snippet)**:
Demonstrates that LLMs internally represent beliefs of self and other agents in a **linearly decodable** form from residual-stream activations. Training linear classifier probes on hidden states recovers the belief status from various agents' perspectives. Crucially, **manipulating** (steering) these belief representations produces dramatic changes in ToM benchmark performance, while random-direction manipulation is negligible — evidence that the internal representation is causally used, not merely correlated. Generalizes across social-reasoning tasks with different causal-inference patterns. This is the closest prior work on *probing* self/other beliefs in LLMs and shows that a *representation-level* switch exists; Claim 4 in `task.md` scales this insight up to a *head-level* dynamic amplifier.

---

### Paper 3: Interpretability in the Wild — a Circuit for Indirect Object Identification in GPT-2 small
- **Authors**: Wang, Variengien, Conmy, Shlegeris, Steinhardt
- **Year**: 2022
- **Venue**: ICLR 2023
- **Source**: WebSearch (arXiv, ar5iv, OpenReview)
- **Identifier**: arXiv:2211.00593
- **URL**: https://arxiv.org/abs/2211.00593

**Abstract (from snippet)**:
The reference example of end-to-end reverse-engineering of a "natural" behavior (IOI: e.g., "When John and Mary went to the store, John gave a drink to ___") into a circuit in GPT-2 small. Uses **causal interventions** — path patching, activation patching, and zero/mean ablation — to isolate 26 attention heads in 7 classes: Name-Mover Heads, S-Inhibition Heads, Duplicate-Token Heads, Previous-Token / Induction Heads, Backup Name-Movers, and Negative Name-Movers. The methodology (cheap correlational screen → causal patch → matched-control baseline) is exactly the template Claim 2 in `task.md` will follow, transposed from names → belief frames.

---

### Paper 4: Interpreting Negation in GPT-2: Layer- and Head-Level Causal Analysis
- **Authors**: (arXiv 2603.12423)
- **Year**: 2026 (recent)
- **Venue**: arXiv preprint 2603.12423
- **Source**: WebSearch
- **Identifier**: arXiv:2603.12423
- **URL**: https://arxiv.org/html/2603.12423

**Abstract (from snippet)**:
Layer- and head-level causal analysis of negation processing in GPT-2. Ablating specific attention-head components directly disrupts the model's negation sensitivity, confirming those heads carry the signal. Methodologically related — negation is another *frame*-shifting operation that pivots the model's output.

---

### Paper 5: LLM Circuit Analyses Are Consistent Across Training and Scale (NeurIPS 2024)
- **Authors**: Curt Tigges et al.
- **Year**: 2024
- **Venue**: NeurIPS 2024
- **Source**: WebSearch (NeurIPS proceedings)
- **Identifier**: NeurIPS 2024 paper
- **URL**: https://proceedings.neurips.cc/paper_files/paper/2024/file/47c7edadfee365b394b2a3bd416048da-Paper-Conference.pdf

**Abstract (from snippet)**:
Tracks how model mechanisms (circuits) emerge and evolve across **300B tokens of training** in decoder-only LMs, using Pythia (70M–2.8B). Finds task abilities and the functional components that support them emerge consistently at similar token counts across scale. Direct methodological precedent for Claim 3 (formation-window analysis on Pythia intermediate checkpoints).

---

### Paper 6: When Do Attention Circuits Form? Developmental Trajectories of Capability and Attention-Sink Emergence Across Three 1B-Class Architectures
- **Authors**: (arXiv 2606.02378)
- **Year**: 2026
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2606.02378
- **URL**: https://arxiv.org/pdf/2606.02378

**Abstract (from snippet)**:
Examines attention heads across 8 intermediate training checkpoints in 7 Pythia (and comparable) models to track when key circuits form. Findings: **induction heads emerge early (~step 1,000 out of 143,000); FV (factual value) heads emerge substantially later (~step 16,000)**. Larger models tend to develop these circuits slightly earlier. Very direct precedent for Claim 3 (belief-circuit formation window).

---

### Paper 7: Which Attention Heads Matter for In-Context Learning?
- **Authors**: (arXiv 2502.14010)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2502.14010
- **URL**: https://arxiv.org/abs/2502.14010

**Abstract (from snippet)**:
Distinguishes function-vector (FV) heads from induction heads in supporting in-context learning; provides taxonomy of specialized attention roles. Relevant to head-level localization framing.

---

### Paper 8: Towards Best Practices of Activation Patching in Language Models: Metrics and Methods (Heimersheim & Nanda 2024)
- **Authors**: Heimersheim, Nanda
- **Year**: 2024
- **Venue**: arXiv:2309.16042 (ICLR 2024 blog / preprint)
- **Source**: WebSearch
- **Identifier**: arXiv:2309.16042
- **URL**: https://arxiv.org/pdf/2309.16042

**Abstract (from snippet)**:
Best-practices guide for activation patching in LLMs — metrics (logit-diff vs. probability-diff), interpretation, and pitfalls. Companion piece to *How to use and interpret activation patching* (arXiv:2404.15255). Baseline reference for the causal-intervention half of Claim 2.

---

### Paper 9: Localizing Model Behavior with Path Patching (Goldowsky-Dill et al.)
- **Authors**: Goldowsky-Dill et al.
- **Year**: 2023
- **Venue**: arXiv:2304.05969
- **Source**: WebSearch
- **Identifier**: arXiv:2304.05969
- **URL**: https://arxiv.org/pdf/2304.05969

**Abstract (from snippet)**:
Formalizes path patching for causal localization of specific behaviors to specific attention-head paths in transformers. Directly relevant if Claim 2's Fisher-based candidate set needs a downstream path-level refinement.

---

### Paper 10: Pattern Selectivity is Not Task-Causal Structure: A Cross-Architecture Mechanistic Study of Composed-Task Circuits in 1B-Class Language Models
- **Authors**: (arXiv 2606.05378)
- **Year**: 2026
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2606.05378
- **URL**: https://arxiv.org/pdf/2606.05378

**Abstract (from snippet)**:
Warns that **attention-pattern selectivity ≠ causal task structure** — patterns that *look* task-specific may fail causal tests. Reinforces the necessity of the 20-random-head baseline + off-target control in Claim 2's selection criteria.

---

### Paper 11: Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling (Biderman et al. 2023)
- **Authors**: Biderman, Schoelkopf et al. (EleutherAI)
- **Year**: 2023
- **Venue**: ICML 2023
- **Source**: WebSearch (PMLR, GitHub, Semantic Scholar)
- **Identifier**: arXiv:2304.01373
- **URL**: https://proceedings.mlr.press/v202/biderman23a/biderman23a.pdf ; https://github.com/EleutherAI/pythia

**Abstract (from snippet)**:
Introduces the Pythia suite: 16 decoder-only LMs (70M–12B params), all trained on the same data in the same order, with **154 intermediate checkpoints per model** released. Enables controlled scale, memorization, and training-dynamics analyses. Foundational infrastructure for Claims 1 (scale) and 3 (formation window).

---

### Paper 12: A Review of Developmental Interpretability in Large Language Models (2025)
- **Authors**: (arXiv 2508.15841)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2508.15841
- **URL**: https://arxiv.org/pdf/2508.15841

**Abstract (from snippet)**:
Survey of developmental interpretability methods that use intermediate checkpoints (Pythia and comparable suites) to track when circuits, features, and capabilities form during pretraining. Relevant methodological context for Claim 3.

---

### Paper 13: Activation Steering / Representation Engineering: Survey and Research Challenges (2025)
- **Authors**: (arXiv 2502.17601)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2502.17601
- **URL**: https://arxiv.org/pdf/2502.17601

**Abstract (from snippet)**:
Surveys activation steering / representation engineering — inference-time interventions on internal activations to alter output along interpretable axes (sentiment, topic, safety, truthfulness, ToM) without weight modification. Discusses steering vectors, head-level PASTA-style amplification, dynamic composition. Direct backdrop for Claim 4's dynamic head amplifier.

---

### Paper 14: Dynamic Activation Composition (PASTA family)
- **Authors**: (from Activation Steering survey and Emergent Mind topic)
- **Year**: 2024–2025
- **Venue**: arXiv preprints; NeurIPS-adjacent
- **Source**: WebSearch
- **Identifier**: multiple
- **URL**: (see survey 2502.17601 for pointers)

**Abstract (paraphrased)**:
Two key methodological patterns for dynamic control:
1. **PASTA** — profiles attention heads to identify those improving task performance and suppresses attention on non-instruction tokens.
2. **Semantics-Adaptive Dynamic Intervention (SADI)** — identifies and steers the most critical neurons or attention heads (via binary masks) for the *current* semantic context, allowing per-input, element-wise adaptation.
3. **Dynamic Activation Composition** — adjusts steering intensity via information-theoretic metrics (KL divergence between steered vs. unsteered token distributions).

All three prefigure Claim 4's frame-classifier-driven head amplifier. The novelty in Claim 4 is (a) *belief-frame* classification (not sentiment / safety), (b) using Claim-2 Fisher-localized heads as the amplification target, (c) preserving `world_knowledge` frames unchanged.

---

### Paper 15: Belief in the Machine — Investigating Epistemological Blind Spots of Language Models
- **Authors**: (arXiv 2410.21195)
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2410.21195
- **URL**: https://arxiv.org/pdf/2410.21195

**Abstract (from snippet)**:
Behavioral evaluation of LMs on belief vs. knowledge separation. Reports gaps between belief attribution and factual recall accuracies. Background for the *behavioral* half of Claim 1.

---

### Paper 16: Understanding Social Reasoning in Language Models with Language Models (BigToM)
- **Authors**: Gandhi et al.
- **Year**: 2023
- **Venue**: NeurIPS 2023
- **Source**: WebSearch
- **Identifier**: arXiv:2306.15448
- **URL**: https://arxiv.org/pdf/2306.15448

**Abstract (from snippet)**:
LLM-generated benchmark for social reasoning (BigToM). Frame-level split of forward-belief, backward-belief, forward-action, etc. Related benchmark space; complementary to the belief_core dataset used here.

---

### Paper 17: OmniToM — Benchmarking Theory of Mind in LLMs via Explicit Belief Modeling
- **Authors**: (arXiv 2605.26322)
- **Year**: 2026
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2605.26322
- **URL**: https://arxiv.org/pdf/2605.26322

**Abstract (from snippet)**:
Benchmark explicitly modeling belief-tracking in ToM tasks. Recent benchmark work; provides broader ToM landscape context but is not used directly (task.md pins to belief_core / belief_holdout).

---

### Paper 18: Language Statistics and False Belief Reasoning: Evidence from 41 Open-Weight LMs (2026)
- **Authors**: (arXiv 2602.16085)
- **Year**: 2026
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2602.16085
- **URL**: https://arxiv.org/pdf/2602.16085

**Abstract (from snippet)**:
Behavioral analysis across 41 open-weight LMs on false-belief tasks: reports that LMs on average score ~80.7% on third-person false-belief scenarios (James/Mary style) vs. ~54.4% on personal-belief tasks, indicating a *processing gap* between attributed and personal belief — directly relevant to Claim 1's expected direction.

---

### Paper 19: Cognitive Mirrors — Exploring the Diverse Functional Roles of Attention Heads in LLM Reasoning
- **Authors**: (arXiv 2512.10978)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2512.10978
- **URL**: https://arxiv.org/pdf/2512.10978

**Abstract (from snippet)**:
Head-level functional taxonomy in reasoning; validates specialization of heads across roles.

---

### Paper 20: Decomposing Theory of Mind — How Emotional Processing Mediates ToM Abilities in LLMs (2025)
- **Authors**: (arXiv 2511.15895)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2511.15895
- **URL**: https://arxiv.org/html/2511.15895v1

**Abstract (from snippet)**:
Decomposition of ToM abilities in LLMs including emotional-processing mediation. Broader context for ToM localization.

---

### Paper 21: Standards for Belief Representations in LLMs (2024)
- **Authors**: (arXiv 2405.21030)
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2405.21030
- **URL**: https://arxiv.org/pdf/2405.21030

**Abstract (from snippet)**:
Position paper on how to define / evaluate belief representations in LLMs. Frames the definitional question underlying Claim 4's frame classifier.

---

### Paper 22: How to Use and Interpret Activation Patching (Nanda et al. 2024)
- **Authors**: Nanda et al.
- **Year**: 2024
- **Venue**: arXiv:2404.15255
- **Source**: WebSearch
- **Identifier**: arXiv:2404.15255
- **URL**: https://arxiv.org/abs/2404.15255

**Abstract (from snippet)**:
Practical guide to applying activation patching correctly, discussing pitfalls (mean vs. zero ablation, resample ablation, metric choice). Baseline reference alongside Paper 8.
