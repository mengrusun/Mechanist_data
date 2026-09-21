# Raw Literature Retrieval: Belief representation & false-belief circuits in Pythia LMs (reproduction of "Sensitivity Meets Sparsity")

**Date**: 2026-07-22
**Query**: Belief representation and false-belief / theory-of-mind circuits in pretrained Pythia language models; ground the fixed methods for reproduction (Fisher-information mask, zero-ablation of attention heads, intermediate-checkpoint formation-window analysis, probe-and-amplify controller).
**Sources scanned**: mechanic-db (cloud SEARCH, 150 papers returned) + arXiv API + WebSearch. Zotero / Obsidian / local PDFs: skipped (not configured / no local library).
**Query formulations used**:
- "Sensitivity Meets Sparsity theory of mind LLM Fisher information sparse parameters"
- "false belief attention head circuit language model interpretability Pythia GPT"
- "Pythia intermediate checkpoints developmental analysis language model pretraining trajectory"
- "zero ablation attention head causal circuit discovery IOI transformer"
- "activation steering head amplification controllable generation belief representation LLM"
- Decomposed mechanic-db query (interp_db): Belief representation, ToM circuits, Fisher-information sparse parameter attribution, zero-ablation attention heads, Pythia checkpoint developmental analysis, activation steering (packed into a single interp_db sub-query with techniques ∈ {circuit_discovery, causal_attribution, probing, gradient_detection}, components ∈ {attention, circuit, residual_stream}, task_scenarios ∈ {social_computation_and_communication, fact_knowledge})
- Decomposed mechanic-db query (sciatlas_db): Theory of Mind / false-belief tasks / Sally-Anne (cognitive-science grounding)

---

## Retrieved Papers

### Paper 1: Sensitivity Meets Sparsity: The Impact of Extremely Sparse Parameter Patterns on Theory-of-Mind of Large Language Models
- **Authors**: Chen et al.
- **Year**: 2025
- **Venue**: arXiv:2504.04238
- **Source**: mechanic-db + WebSearch + arXiv API
- **Identifier**: arXiv:2504.04238 / doi:10.48550/arxiv.2504.04238
- **URL**: https://arxiv.org/abs/2504.04238

**Abstract**:
This paper investigates the emergence of Theory-of-Mind (ToM) capabilities in large language models (LLMs) from a mechanistic perspective, focusing on the role of extremely sparse parameter patterns. We introduce a novel method to identify ToM-sensitive parameters and reveal that perturbing as little as 0.001% of these parameters significantly degrades ToM performance while also impairing contextual localization and language understanding. To identify these parameters, we use the Fisher information matrix to derive a binary mask that captures the parameters most sensitive to ToM-related computation. Our analysis shows that these sensitive parameters are closely linked to the positional encoding module, particularly in models using Rotary Position Embedding (RoPE), where perturbations disrupt dominant-frequency activations critical for contextual processing. Further inspection reveals that they exhibit strong sparsity and low-rank structure, with significant perturbations concentrated in the W_Q and W_K matrices.

**Notes**: **THIS IS THE REPRODUCTION PAPER.** Provides the Fisher-mask method the project reuses in Claim 2. Also names the specific mask construction (top-fraction of ToM-signal AND-NOT top-fraction of control-signal) that maps directly to `Mask_attributed / Mask_personal` in task.md.

---

### Paper 2: How large language models encode theory-of-mind: a study on sparse parameter patterns
- **Authors**: (same team as Paper 1)
- **Year**: 2025
- **Venue**: npj Artificial Intelligence
- **Source**: mechanic-db + WebSearch
- **Identifier**: doi:10.1038/s44387-025-00031-9
- **URL**: https://www.nature.com/articles/s44387-025-00031-9

**Abstract**:
Peer-reviewed venue version of Paper 1. Introduces a novel method to identify ToM-sensitive parameters using the Fisher information matrix; perturbing as little as 0.001% of these parameters significantly degrades ToM performance while impairing contextual localization and language understanding. Sensitive parameters concentrate in W_Q / W_K matrices and connect to RoPE positional encoding.

**Notes**: Authoritative published version; grounds the Fisher-based localization methodology.

---

### Paper 3: Brittle Minds, Fixable Activations: Understanding Belief Representations in Language Models
- **Authors**: Bortoletto, Ma, et al.
- **Year**: 2024
- **Venue**: arXiv:2406.17513
- **Source**: mechanic-db
- **Identifier**: arXiv:2406.17513

**Abstract**:
Despite growing interest in Theory of Mind (ToM) tasks for evaluating language models (LMs), little is known about how LMs internally represent mental states of self and others. Understanding these internal mechanisms is critical — not only to move beyond surface-level performance, but also for model alignment and safety, where subtle misattributions of mental states may go undetected. Uses probing, activation intervention, and steering vectors to study belief representations.

**Notes**: Directly relevant to Claim 4 (probe + steer controller). Shows probes + activation edits can *fix* ToM behavior.

---

### Paper 4: Language Models Represent Beliefs of Self and Others
- **Authors**: Zhu, Jian, Rathee, et al.
- **Year**: 2024
- **Venue**: arXiv:2402.18496 (later ICML 2024)
- **Source**: mechanic-db
- **Identifier**: arXiv:2402.18496

**Abstract**:
Understanding and attributing mental states, known as Theory of Mind (ToM), emerges as a fundamental capability for human social reasoning. In this study, we discover that it is possible to linearly decode the belief status from the perspectives of various agents through neural activations of language models, indicating the existence of internal representations of self and others' beliefs. Manipulating these representations causally alters social reasoning performance, underscoring the causal role of belief representations in ToM.

**Notes**: Directly supports Claim 4 methodology (linear probe → causal intervention). Confirms that a lightweight probe *can* pick up personal vs attributed belief from residual-stream activations.

---

### Paper 5: Language Models use Lookbacks to Track Beliefs
- **Authors**: (2025)
- **Year**: 2025
- **Venue**: arXiv:2505.14685
- **Source**: mechanic-db
- **Identifier**: arXiv:2505.14685

**Abstract**:
How do language models represent characters' beliefs, especially when those beliefs may differ from reality? We analyze LMs' ability to reason about characters' beliefs using causal mediation and abstraction. We construct a dataset, CausalToM, and identify specific attention-head-level mechanisms ("lookback" heads) that route past belief-relevant information.

**Notes**: Corroborates the existence of *distinct* attention-head circuits for tracking own-vs-other beliefs — exactly the phenomenon Claim 2 seeks to localize.

---

### Paper 6: Unveiling Theory of Mind in Large Language Models: A Parallel to Single Neurons in the Human Brain
- **Authors**: Cross, Xiang, et al.
- **Year**: 2023
- **Venue**: arXiv:2309.01660
- **Source**: mechanic-db
- **Identifier**: arXiv:2309.01660

**Abstract**:
With their recent development, large language models (LLMs) have been found to exhibit a certain level of Theory of Mind (ToM), a complex cognitive capacity that is related to our conscious mind and that allows us to infer another's beliefs and perspective. Draws parallels between LLM neurons that respond selectively to ToM prompts and single-neuron recordings from dorsomedial prefrontal cortex in humans.

**Notes**: Neuroscience-grounded motivation for localization: ToM computations in humans concentrate in a small, identifiable neural population — a hypothesis the sparse-Fisher-mask literature adopts computationally.

---

### Paper 7: Evaluating Contrast Localizer for Identifying Causal Units in Social & Mathematical Tasks in Language Models
- **Authors**: (2025)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: mechanic-db
- **Identifier**: (2025)

**Abstract**:
Adapts a neuroscientific contrast localizer to pinpoint causally relevant units for Theory of Mind (ToM) and mathematical reasoning tasks in LLMs and VLMs. Across 11 LLMs and 5 VLMs (3B-90B), localizes top-activated units using contrastive stimulus sets and assesses their causal role via targeted ablations.

**Notes**: Methodologically close to Claim 2 (contrastive signal → localized units → causal ablation). Contrastive design (target-signal AND NOT control-signal) mirrors the Fisher-mask AND-NOT construction in task.md.

---

### Paper 8: Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small (IOI)
- **Authors**: Wang, Variengien, Conmy, Shlegeris, Steinhardt
- **Year**: 2022
- **Venue**: ICLR 2023 / arXiv:2211.00593
- **Source**: mechanic-db + WebSearch + arXiv
- **Identifier**: arXiv:2211.00593  (49 cites)

**Abstract**:
Reverse-engineers a circuit of ~26 attention heads across 5 categories (Name Mover, Negative Name Mover, S-Inhibition, Duplicate Token, Induction, Backup) responsible for IOI in GPT-2 Small. Introduces the pipeline: candidate discovery → path patching / knockout → controlled ablation → validation on distribution.

**Notes**: **Methodological reference for Claim 2** — the "identify candidate heads → zero-ablate → measure task-specific accuracy drop" workflow originates here. Random-head baseline design also standard in this line.

---

### Paper 9: What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation
- **Authors**: (2024)
- **Year**: 2024
- **Venue**: arXiv:2404.07129
- **Source**: mechanic-db
- **Identifier**: arXiv:2404.07129  (2 cites)

**Abstract**:
Induction heads (IH), which perform a match-and-copy operation, emerge around the same time as a notable phase change in the loss during training of large transformers. Uses **synthetic data + Pythia checkpoints** to identify a diverse set of interacting sub-circuits whose interaction dynamics govern IH emergence during pretraining.

**Notes**: **Direct methodological grandparent of Claim 3** — same design pattern: probe intermediate checkpoints, measure behavioral emergence, then localize the causal sub-circuit at each checkpoint. Uses Pythia.

---

### Paper 10: LLM Circuit Analyses Are Consistent Across Training and Scale
- **Authors**: (2024)
- **Year**: 2024
- **Venue**: arXiv:2407.10827
- **Source**: WebSearch
- **Identifier**: arXiv:2407.10827

**Abstract**:
Uses Pythia checkpoints (across sizes and training steps) to test whether circuits identified at one checkpoint/scale remain valid at another. Finds circuit structure is remarkably stable across training and scale.

**Notes**: Directly supports Claim 3's premise that circuits identified at the final checkpoint remain identifiable / analyzable at intermediate ones — justifies re-using the Claim-2 pythia-1b head set to intervene at every checkpoint.

---

### Paper 11: In-context Learning and Induction Heads
- **Authors**: Olsson et al. (Anthropic)
- **Year**: 2022
- **Venue**: Transformer Circuits Thread
- **Source**: mechanic-db
- **Identifier**: (2022)  (83 cites)

**Abstract**:
Landmark study identifying induction heads as the specific attention-head mechanism responsible for in-context learning in transformer LMs, and showing that IH formation coincides with a discrete phase change in the training loss (formation window).

**Notes**: Foundational formation-window paradigm for Claim 3 — motivates the "there is a discrete step-count at which the ability appears" hypothesis. Also references Pythia-style checkpoint analysis.

---

### Paper 12: Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling
- **Authors**: Biderman, Schoelkopf, Anthony, et al. (EleutherAI)
- **Year**: 2023
- **Venue**: ICML 2023 / arXiv:2304.01373
- **Source**: WebSearch + arXiv
- **Identifier**: arXiv:2304.01373

**Abstract**:
Pythia is a suite of decoder-only autoregressive transformer language models spanning from 14M to 12B parameters together with 144 intermediate checkpoints. Models trained on the Pile (~300B tokens). Each model has 154 checkpoints: 11 at 0, 1, 2, 4, ..., 512 steps and 143 at 1000, 2000, ..., 143000 steps.

**Notes**: The models and checkpoint schedule the project reproduces. Provides the checkpoint list that must be sampled for Claim 3.

---

### Paper 13: Knowledge Circuits in Pretrained Transformers
- **Authors**: Yao, Zhang, Chen, et al.
- **Year**: 2024
- **Venue**: NeurIPS 2024 / arXiv:2405.17969
- **Source**: mechanic-db
- **Identifier**: arXiv:2405.17969

**Abstract**:
Isolates knowledge circuits (subgraphs of attention heads + MLPs) in GPT-2 Medium/Large for various factual-recall tasks. Uses attention-head level intervention (knockout, patch) to demonstrate the circuit's causal sufficiency for the target knowledge. Analyzes the role of intermediate MLPs as "knowledge accumulators".

**Notes**: Supports the world-knowledge control side of Claim 2 — provides a reference for how "factual knowledge circuits" differ structurally from ToM/belief circuits.

---

### Paper 14: Knowledge Neurons in Pretrained Transformers
- **Authors**: Dai, Dong, Hao, Sui, Chang, Wei
- **Year**: 2022
- **Venue**: ACL 2022 / arXiv:2104.08696
- **Source**: mechanic-db
- **Identifier**: arXiv:2104.08696  (125 cites)

**Abstract**:
Identifies "knowledge neurons" in the feed-forward layers of pretrained transformers that store specific factual relations. Uses a knowledge attribution method (gradient × activation) to score each neuron's importance and demonstrates that suppressing or amplifying these neurons targeted-modifies the factual output.

**Notes**: Foundational gradient-based attribution paper — the technical ancestor of Fisher-information-based parameter attribution (Fisher = expectation of gradient² under the model distribution).

---

### Paper 15: Transformers represent belief state geometry in their residual stream
- **Authors**: (2024)
- **Year**: 2024
- **Venue**: arXiv:2405.15943
- **Source**: mechanic-db
- **Identifier**: arXiv:2405.15943

**Abstract**:
Uses the theory of optimal prediction to anticipate and then confirm that transformer LMs represent "belief states" over hidden data-generating states in a low-dimensional linear subspace of the residual stream.

**Notes**: Corroborates that belief-like variables are linearly decodable from residual stream — the assumption underlying the Claim 4 frame classifier.

---

### Paper 16: Constrained belief updates explain geometric structures in transformer representations
- **Authors**: (2025)
- **Year**: 2025
- **Venue**: arXiv:2502.01954
- **Source**: mechanic-db
- **Identifier**: arXiv:2502.01954

**Abstract**:
Follow-up to Paper 15. Provides evidence that transformers implement a constrained Bayesian belief-updating computation, dictated by architectural sparsity, and shows the resulting geometry in the residual stream.

**Notes**: Reinforces the linear-belief-representation assumption for Claim 4's probe.

---

### Paper 17: The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets
- **Authors**: Marks, Tegmark
- **Year**: 2023
- **Venue**: ICLR 2024 / arXiv:2310.06824
- **Source**: mechanic-db
- **Identifier**: arXiv:2310.06824  (16 cites)

**Abstract**:
Trains linear probes on hidden states of GPT-family models on labelled true/false statements; finds a low-dimensional "truth direction" that transfers across topics and can be used to steer generations.

**Notes**: Corollary: since a truth direction is linearly decodable, a *belief-frame* direction (personal-vs-attributed) is at least plausible to be similarly decodable — supports the frame classifier design in Claim 4.

---

### Paper 18: Emergence of Minimal Circuits for Indirect Object Identification in Attention-Only Transformers
- **Authors**: (2025)
- **Year**: 2025
- **Venue**: arXiv:2510.25013
- **Source**: WebSearch + mechanic-db
- **Identifier**: arXiv:2510.25013

**Abstract**:
Trains attention-only transformers from scratch on a symbolic IOI task; observes emergence of minimal head-level circuits during training and analyzes the developmental trajectory.

**Notes**: Formation-window methodology from a *from-scratch* angle — provides a comparison point for Claim 3's pretraining-checkpoint analysis.

---

### Paper 19: Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice Capabilities in Chinchilla
- **Authors**: Lieberum, Rahtz, Kramár, et al.
- **Year**: 2023
- **Venue**: arXiv:2307.09458
- **Source**: mechanic-db
- **Identifier**: arXiv:2307.09458  (6 cites)

**Abstract**:
Applies circuit-analysis methodology (path patching, ablation) to a 70B Chinchilla model on multiple-choice QA. Finds circuits at scale are similar in structure to those in small models but harder to fully specify.

**Notes**: Scale-relevant validation that the circuit-discovery workflow used in Claim 2 remains applicable at the multi-billion-parameter regime (which pythia-2.8b touches).

---

### Paper 20: Belief in the Machine: Investigating Epistemological Blind Spots of Language Models
- **Authors**: (2024)
- **Year**: 2024
- **Venue**: arXiv:2410.21195
- **Source**: WebSearch + arXiv
- **Identifier**: arXiv:2410.21195

**Abstract**:
Systematic behavioural evaluation of how LLMs (GPT-4, Claude, Pythia among others) handle epistemically loaded statements about belief, knowledge, and truth — including false-belief conditions.

**Notes**: Behavioural background for Claim 1 (Scale-Dependent Emergence) — establishes that belief-related behaviours differ systematically across model scales and families.

---

### Paper 21: Circuit Component Reuse Across Tasks in Transformer Language Models
- **Authors**: Merullo, Eickhoff, Pavlick
- **Year**: 2023
- **Venue**: arXiv:2310.08744
- **Source**: mechanic-db
- **Identifier**: arXiv:2310.08744  (3 cites)

**Abstract**:
Provides evidence that circuit components discovered on one task (IOI) participate in circuits for structurally similar tasks. Motivates task-agnostic head roles.

**Notes**: Supports the specificity criterion in Claim 2 — the "off-target task must NOT be broken by ≥ 0.10" bar guards against generic head damage vs task-specific belief-head damage.

---

### Paper 22: Thinking Fast and Slow in Large Language Models
- **Authors**: Hagendorff, Fabi, Kosinski
- **Year**: 2022
- **Venue**: arXiv:2212.05206
- **Source**: mechanic-db
- **Identifier**: arXiv:2212.05206  (38 cites)

**Abstract**:
Behavioural study of dual-process cognition in LLMs; establishes that model scale meaningfully changes cognitive-style outputs.

**Notes**: General motivation for Claim 1 (scale-dependent emergence of cognitive capacities in LMs).

---

### Paper 23: Sparse Autoencoders Enable Scalable and Reliable Circuit Identification in Language Models
- **Authors**: (2024)
- **Year**: 2024
- **Venue**: arXiv:2405.12522
- **Source**: mechanic-db
- **Identifier**: (2024)

**Abstract**:
Uses sparse autoencoders trained on residual-stream / attention output activations to identify sparse causal circuits at scale.

**Notes**: Alternative sparsity-based localization approach that competes with the Fisher-mask + zero-ablation route the project uses. Cited here for completeness — the project does not use SAEs.

---

### Paper 24: Activation Steering (Representation Engineering) — general context
- **Authors**: multiple
- **Year**: 2023-2026
- **Venue**: various (arXiv:2308.10248 Turner et al.; arXiv:2310.01405 Zou et al.; recent extensions)
- **Source**: WebSearch
- **Identifier**: various

**Abstract**:
Activation Steering / Representation Engineering: extract a "steering vector" from activations on paired contrastive stimuli, then add it (with learned magnitude) to the residual stream during generation to shift outputs along the target dimension. Recent extensions include prompt-activation duality, honest steering, and attention-level intervention.

**Notes**: The general family Claim 4's "amplify belief heads at inference time" belongs to. The project's variant is *head-restricted* (only amplifies pre-identified belief heads) rather than a full residual-stream steering vector — this is a more targeted variant of the standard steering pipeline.

---

### Paper 25: A Survey of Theory of Mind in Large Language Models: Evaluations, Representations, and Safety Risks
- **Authors**: Nguyen
- **Year**: 2025
- **Venue**: arXiv:2502.06470
- **Source**: arXiv API
- **Identifier**: arXiv:2502.06470

**Abstract**:
Broad survey of behavioral and representational ToM studies in LLMs — covers benchmarks, elicitation prompts, probing / representation studies, and safety implications.

**Notes**: Survey-level context; provides the "state of ToM in LLMs" landscape into which this reproduction fits.

---

### Paper 26: Decomposing Theory of Mind: How Emotional Processing Mediates ToM Abilities in LLMs
- **Authors**: (2025)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: mechanic-db

**Notes**: Decomposition of ToM sub-abilities; supports viewing ToM as a family of separable abilities (personal-belief vs attributed-belief being one dissociation).

---

### Paper 27: The LLM Language Network: A Neuroscientific Approach for Identifying Causally Task-Relevant Units
- **Authors**: (2024)
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: mechanic-db

**Notes**: Applies a neuroscience-inspired network-localization approach to identify task-relevant sub-networks in LLMs. Methodologically adjacent to the Fisher-mask + AND-NOT construction (target-signal minus generic-signal).

---

### Paper 28: Transcoders Find Interpretable LLM Feature Circuits
- **Authors**: (2024)
- **Year**: 2024
- **Venue**: NeurIPS 2024 / arXiv:2406.11944
- **Source**: mechanic-db

**Notes**: Alternative feature-basis circuit identification. Included for completeness of the sparsity/attribution landscape; not used by the project.

---

### Paper 29: Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms
- **Authors**: (2024)
- **Year**: 2024
- **Venue**: NeurIPS 2024

**Notes**: Discusses evaluation of *faithfulness* of extracted circuits — argues that circuit-overlap metrics alone are insufficient and that causal-intervention validation (as used in Claim 2) is required.

---

### Paper 30: Circuit Stability Characterizes Language Model Generalization
- **Authors**: (2025)
- **Year**: 2025
- **Venue**: arXiv preprint

**Notes**: Studies whether circuits remain stable under distribution shift — relevant to Claim 4's OOD holdout evaluation.

---

*The full 150-paper retrieval is preserved at `mechanic_db_cache/20260722_020650_belief_reproduction.json` for audit. The 30 papers above cover the reproduction's methodological ancestry (Fisher / attention-head zero-ablation / checkpoint analysis / linear probes / steering) and phenomenon-level context (ToM, personal vs attributed belief). Papers 31-150 in the cache are further mechanistic-interpretability and cognitive-science neighbours that further support but do not change the landscape.*
