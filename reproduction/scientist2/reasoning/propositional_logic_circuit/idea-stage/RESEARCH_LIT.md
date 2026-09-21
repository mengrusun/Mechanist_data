# Raw Literature Retrieval: Sparse Modular Circuit for Propositional-Logic Reasoning in LLMs

**Date**: 2026-07-14
**Query**: mechanistic-interpretability circuit for multi-step propositional-logic reasoning in LLMs — activation patching, path patching, resample ablation, attention-head knockouts, causal mediation; sparse subset of attention heads + MLP components; modular sub-circuits (fact identification, rule application, answer projection); necessity + sufficiency; Mistral-7B lead, Gemma-2-9B / 27B verify.
**Sources scanned**: mechanic-db cloud SEARCH (interp_db, 150 results); arXiv API (7 additional queries, ~35 unique results after dedup); WebSearch skipped (arXiv+cloud coverage sufficient); Zotero / Obsidian / local PDFs — none configured.
**Query formulations used**:
- `mechanistic circuit for multi-step propositional-logic reasoning in LLMs: activation patching, path patching, resample ablation, causal mediation locate sparse attention heads + MLPs...` (mechanic-db decomposed, interp_db, `year_min: 2022`, enums: `techniques=[circuit_discovery, causal_attribution]`, `components=[circuit, attention, mlp_ffn, residual_stream]`, `abilities=[reasoning]`, `target_models=[Mistral-7B, Gemma-2-9B, Gemma-2-27B, ...]`, `top_k=300`)
- `circuit propositional logic reasoning transformer activation patching` (arXiv)
- `path patching mechanistic interpretability circuit LLM` (arXiv)
- `attention head Indirect Object Identification circuit` (arXiv)
- `multi-hop reasoning latent transformer mechanistic` (arXiv)
- `sparse circuit discovery attribution edge` (arXiv)
- `causal mediation analysis language model reasoning` (arXiv)
- `in-context learning induction head mechanism transformer` (arXiv)
- `ACDC automated circuit discovery transformer` (arXiv)
- `MLP knowledge neurons factual recall` (arXiv)
- `necessity sufficiency circuit interpretability` (arXiv)

**Forbidden sources**: `.claude/forbidden-urls.txt` blocks the direct target paper (arXiv 2411.04105 "A Implies B: Circuit Analysis in LLMs for Propositional Logical Reasoning") and any related arXiv IDs from 2411.* onward for that title. All retrieval below excludes those.

---

## Retrieved Papers (mechanic-db, top 60 by relevance)

### Paper 1: Reasoning Circuits in Language Models: A Mechanistic Interpretation of Syllogistic Inference
- **Year**: 2024
- **Source**: mechanic-db (interp_db)
- **Identifier**: arXiv 2408.08590
- **Abstract**: Recent studies on reasoning in language models (LMs) have sparked a debate on whether they can learn systematic inferential principles or merely exploit superficial patterns in the training data. To understand and uncover the mechanisms adopted for formal reasoning in LMs, this paper presents a mechanistic interpretation of syllogistic inference. Specifically, we present a methodology for circuit discovery aimed at interpreting content-independent and formal reasoning mechanisms. Through two distinct intervention methods, we uncover a sufficient and necessary circuit involving middle-term suppression that elucidates how LMs transfer information to derive valid conclusions from premises. Furthermore, we investigate how belief biases manifest in syllogistic inference, finding evidence of partial contamination from content-heavy training.

### Paper 2: [BLOCKED — target paper of the reproduction] "A Implies B: Circuit Analysis in LLMs for Propositional Logical Reasoning" (arXiv 2411.04105)
- **Year**: 2024
- **Source**: mechanic-db returned this row; per `.claude/forbidden-urls.txt` we do NOT fetch its content. Only the mechanic-db one-line metadata (title, year) is recorded here; the abstract text is treated as unavailable so we work from `task.md` alone.

### Paper 3: Towards a Mechanistic Understanding of Propositional Logical Reasoning in Large Language Models
- **Year**: 2026
- **Source**: mechanic-db (interp_db)
- **Abstract**: Understanding how LLMs perform logical reasoning internally remains a fundamental challenge. Prior mechanistic studies focus on identifying task-specific circuits; they leave open the question of what computational strategies LLMs employ for propositional reasoning. We address this gap through comprehensive analysis of Qwen3 (8B and 14B) on PropLogic-MI, a controlled dataset spanning 11 propositional logic rule categories across one-hop and two-hop reasoning. Rather than asking "which components are necessary", we ask "how does the model organize computation?". Our analysis reveals a coherent computational architecture comprising four interlocking mechanisms: Staged Computation (layer-wise processing phases), Information Transmission (information flow aggregation from facts and rules to the query), plus two further mechanisms characterizing how truth values are propagated and projected to the answer token.

### Paper 4: Benchmarking and Understanding Compositional Relational Reasoning of LLMs (GAR benchmark)
- **Year**: 2025 (AAAI)
- **Source**: mechanic-db
- **Abstract**: Introduces the Generalized Associative Recall (GAR) benchmark that unifies mechanistic-interpretability tasks (IOI, colored objects, ...). Uses attribution patching on Vicuna-33B to discover core circuits reused across tasks. Shows LLMs have fundamental CRR deficiencies; GAR is a general controlled MI substrate.

### Paper 5: Evaluating Brain-Inspired Modular Training in Automated Circuit Discovery
- **Year**: 2024
- **Source**: mechanic-db (arXiv 2401.03646)

### Paper 6: LLM Circuit Analyses Are Consistent Across Training and Scale
- **Year**: 2024 (arXiv 2407.10827)
- **Source**: mechanic-db
- **Abstract**: Tracks how circuits emerge and evolve across 300B tokens of training in decoder-only LLMs from 70M to 2.8B params. Finds that task abilities and their functional components emerge consistently at similar token counts across scale, and circuits at the end of pretraining are largely stable and predictable.

### Paper 7: Toward Mechanistic Explanation of Deductive Reasoning in Language Models
- **Year**: 2025 (arXiv 2510.09340)
- **Source**: mechanic-db
- **Abstract**: Shows a small LM can solve a deductive reasoning task by learning rules (not statistical learning). Provides a low-level explanation of internal representations and circuits. Induction heads play a central role in the rule-completion and rule-chaining steps involved in the required logical inference.

### Paper 8: From Indirect Object Identification to Syllogisms: Exploring Binary Mechanisms in Transformer Circuits
- **Year**: 2025 (arXiv 2508.16109)
- **Source**: mechanic-db
- **Abstract**: Analyzes GPT-2-small on syllogistic prompts of varying difficulty. Identifies multiple circuits that mechanistically explain GPT-2's logical reasoning behavior and uncovers binary "true/false" mechanisms.

### Paper 9: Circuit Component Reuse Across Tasks in Transformer Language Models
- **Year**: 2023 (arXiv 2310.08744)
- **Source**: mechanic-db
- **Abstract**: Shows insights (both low-level head findings and higher-level algorithmic findings) generalize across tasks. IOI circuit reproduces in a larger GPT-2 and is mostly reused to solve Colored Objects — evidence of the reuse hypothesis.

### Paper 10: Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics
- **Year**: 2024 (arXiv 2410.21272)
- **Source**: mechanic-db
- **Abstract**: Uses causal analysis to find a small circuit that explains most of the model's behavior for basic arithmetic; zooms in to individual neurons and identifies a sparse set implementing simple pattern-matching heuristics.

### Paper 11: Circuit Compositions: Exploring Modular Structures in Transformer-Based Language Models
- **Year**: 2024 (arXiv 2410.01434)
- **Source**: mechanic-db
- **Abstract**: Studies whether transformer LMs implement reusable functions through composable sub-networks. Analyses circuits for highly compositional subtasks within a transformer LM. Directly speaks to Sub-claim 2 (modular decomposition).

### Paper 12: Knowledge Circuits in Pretrained Transformers
- **Year**: 2024 (arXiv 2405.17969)
- **Source**: mechanic-db
- **Abstract**: Uncovers "information heads", "relation heads", and MLPs jointly implementing specific-knowledge recall in GPT-2 and TinyLLAMA. Directly analogous to the fact-identification → rule-application → answer-projection decomposition.

### Paper 13: Circuit Stability Characterizes Language Model Generalization
- **Year**: 2025 (arXiv 2505.24731)
- **Source**: mechanic-db
- **Abstract**: Formalizes "circuit stability" (consistency of the identified circuit across inputs) as a predictor of generalization.

### Paper 14: A Mechanistic Interpretation of Arithmetic Reasoning in Language Models using Causal Mediation Analysis
- **Year**: 2023 EMNLP
- **Source**: mechanic-db
- **Abstract**: Intervenes on component activations and measures the resulting change in predicted probabilities to identify the subset of parameters responsible for specific predictions — a template for the necessity direction of our C3.

### Paper 15: Disentangling Recall and Reasoning in Transformer Models through Layer-wise Attention and Activation Analysis
- **Year**: 2025 (arXiv 2510.03366)
- **Source**: mechanic-db
- **Abstract**: Combines activation patching and structured ablations on controlled synthetic linguistic puzzles across Qwen and LLaMA. Finds interventions that reduce reasoning but preserve recall (and vice-versa) — supports Sub-claim 2 (dissociable roles).

### Paper 16: Mechanistic Unveiling of Transformer Circuits: Self-Influence as a Key to Model Reasoning
- **Year**: 2025 (arXiv 2502.09022)
- **Source**: mechanic-db

### Paper 17: Towards Faithful Natural Language Explanations: A Study Using Activation Patching in LLMs
- **Year**: 2024 (arXiv 2410.14155)
- **Source**: mechanic-db

### Paper 18: A Mechanistic Analysis of a Transformer Trained on a Symbolic Multi-Step Reasoning Task
- **Year**: 2024 (arXiv 2402.11917)
- **Source**: mechanic-db
- **Abstract**: Presents a comprehensive mechanistic analysis of a transformer trained on a synthetic reasoning task. Identifies interpretable mechanisms and validates them with correlational and causal evidence. Suggests a depth-bounded recurrent mechanism that operates in parallel and stores intermediate results.

### Paper 19: Towards Best Practices of Activation Patching in Language Models: Metrics and Methods
- **Year**: 2023 (arXiv 2309.16042)
- **Source**: mechanic-db
- **Abstract**: Systematically examines methodological details in activation patching. Choice of metric and corruption method can lead to disparate results; provides guidance.

### Paper 20: Hierarchical Sparse Circuit Extraction from Billion-Parameter Language Models through Scalable Attribution Graph Decomposition
- **Year**: 2026
- **Source**: mechanic-db
- **Abstract**: Reduces circuit discovery from O(2^n) to O(n^2 log n) via multi-resolution attribution + differentiable circuit search. Evaluated on GPT-2 through Llama-7B-13B.

### Paper 21: Selection-Inference (Creswell et al.)
- **Year**: 2022 (arXiv 2205.09712) — highly cited (110+)
- **Source**: mechanic-db

### Paper 22: Unveiling Reasoning Thresholds in Language Models: Scaling, Fine-Tuning, and Interpretability through Attention Maps
- **Year**: 2025
- **Source**: mechanic-db

### Paper 23: Finite State Automata Inside Transformers with Chain-of-Thought: A Mechanistic Study on State Tracking
- **Year**: 2025 (arXiv 2502.20129)
- **Source**: mechanic-db
- **Abstract**: Identifies a circuit responsible for tracking world state; late-layer MLP neurons play a key role. Metrics "compression" and "distinction" show near-100% accuracy — evidence of an implicit FSA.

### Paper 24: Finding Highly Interpretable Prompt-Specific Circuits in Language Models
- **Year**: 2026
- **Source**: mechanic-db
- **Abstract**: Circuits are prompt-specific even within a task. ACC++ extracts cleaner low-dimensional causal signals inside attention heads from a single forward pass — no SAEs, no patching.

### Paper 25: Causal Head Gating (CHG): A Framework for Interpreting Roles of Attention Heads
- **Year**: 2025 (arXiv 2505.13737)
- **Source**: mechanic-db
- **Abstract**: Learns soft gates over heads and assigns each a causal taxonomy — facilitating / interfering / irrelevant. Scales across LLaMA-3 family and diverse tasks.

### Paper 26: RelP: Faithful and Efficient Circuit Discovery via Relevance Patching
- **Year**: 2025
- **Source**: mechanic-db
- **Abstract**: Replaces gradients in attribution patching with LRP propagation coefficients; more faithful than pure attribution patching for deep, highly non-linear networks.

### Paper 27: Assessing Logical Reasoning Capabilities of Encoder-Only Transformer Models
- **Year**: 2023 (arXiv 2312.11720)
- **Source**: mechanic-db

### Paper 28: Hypothesis Testing the Circuit Hypothesis in LLMs
- **Year**: 2024 (arXiv 2410.13032)
- **Source**: mechanic-db
- **Abstract**: Formalizes criteria a circuit is hypothesized to meet — behavior preservation, localization, minimality — and develops a suite of hypothesis tests. Applies to six circuits from the literature.

### Paper 29: Uncovering Intermediate Variables in Transformers using Circuit Probing
- **Year**: 2023 (arXiv 2311.04354)
- **Source**: mechanic-db
- **Abstract**: Proposes circuit probing — automatically uncovers low-level circuits computing hypothesized intermediate variables. Enables causal analysis through targeted parameter-level ablation.

### Paper 30: Emergence of Minimal Circuits for IOI in Attention-Only Transformers
- **Year**: 2025 (arXiv 2510.25013)
- **Source**: mechanic-db

### Paper 31: Towards Interpretable Sequence Continuation: Analyzing Shared Circuits
- **Year**: 2023 (arXiv 2311.04131)
- **Source**: mechanic-db

### Paper 32: Relational reasoning and inductive bias in transformers and large language models (transitive inference)
- **Year**: 2025 (arXiv 2506.04289)
- **Source**: mechanic-db
- **Abstract**: Investigates how transformers perform transitive inference (A>B, B>C ⇒ A>C). Compares in-weights vs in-context learning. Directly relevant to multi-hop propositional reasoning.

### Paper 33: Are formal and functional linguistic mechanisms dissociated in language models?
- **Year**: 2025 (arXiv 2503.11302)
- **Source**: mechanic-db

### Paper 34: APP: Accelerated Path Patching with Task-Specific Pruning
- **Year**: 2025
- **Source**: mechanic-db

### Paper 35: Attribution Patching Outperforms Automated Circuit Discovery
- **Year**: 2024 BlackBoxNLP
- **Source**: mechanic-db
- **Abstract**: Gradient-based attribution patching identifies edge subsets much faster than iterative ACDC and matches or beats its recall.

### Paper 36: Language Model Circuits Are Sparse in the Neuron Basis
- **Year**: 2026
- **Source**: mechanic-db

### Paper 37: Does Circuit Analysis Interpretability Scale? Evidence from Multiple-Choice in Chinchilla
- **Year**: 2023 (arXiv 2307.09458)
- **Source**: mechanic-db
- **Abstract**: Case study of circuit analysis in Chinchilla-70B on multiple-choice QA — establishes that circuit analysis extends to state-of-the-art scale with some caveats.

### Paper 38: Weight-sparse transformers have interpretable circuits
- **Year**: 2025 (arXiv 2511.13653)
- **Source**: mechanic-db

### Paper 39: Interpretability in the Wild: a Circuit for IOI in GPT-2 small (Wang et al. — the foundational IOI paper)
- **Year**: 2022 (arXiv 2211.00593)
- **Source**: mechanic-db
- **Abstract**: The canonical worked example of circuit analysis: identifies a 26-head circuit for IOI in GPT-2-small; uses path patching, ablation, and interchange interventions. Template of "faithfulness / completeness / minimality" that our C1/C3 tests should mirror.

### Paper 40: From Yes-Men to Truth-Tellers: Sycophancy in LLMs with Pinpoint Tuning
- **Year**: 2024
- **Source**: mechanic-db (tangential)

### Paper 41: It is not True that Transformers are Inductive Learners: Probing NLI Models with External Negation
- **Year**: 2024 EACL
- **Source**: mechanic-db

### Paper 42: Understanding the LM to Solve the Symbolic Multi-Step Reasoning Problem from the Perspective of Buffer Mechanism
- **Year**: 2024 (arXiv 2405.15302)
- **Source**: mechanic-db
- **Abstract**: Constructs a symbolic multi-step reasoning task and analyses the information-flow "buffer" mechanism in LMs.

### Paper 43: Rethinking Circuit Completeness in Language Models: AND, OR, and ADDER Gates
- **Year**: 2025 (arXiv 2505.10039)
- **Source**: mechanic-db

### Paper 44: Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization
- **Year**: 2024 (arXiv 2405.15071)
- **Source**: mechanic-db
- **Abstract**: Transformers learn implicit reasoning through grokking (extended training past overfitting). Composition and comparison patterns; mechanistic study of how the resulting circuits form.

### Paper 45: Hopping Too Late: Exploring the Limitations of LLMs on Multi-Hop Queries
- **Year**: 2024 (arXiv 2406.12775)
- **Source**: mechanic-db
- **Abstract**: Studies how LLMs answer multi-hop queries requiring a latent bridge-entity resolution. Directly relevant to Sub-claim 2's fact-identification → rule-application chain.

### Paper 46: A Comparative Study of Neurosymbolic AI Approaches to Interpretable Logical Reasoning
- **Year**: 2025
- **Source**: mechanic-db

### Paper 47: Identifying and Transferring Reasoning-Critical Neurons: Improving LLM Inference Reliability via Activation Steering
- **Year**: 2026
- **Source**: mechanic-db

### Paper 48: Opening the Black Box: A Survey on the Mechanisms of Multi-Step Reasoning in LLMs
- **Year**: 2026
- **Source**: mechanic-db (survey — useful as a landscape reference)

### Paper 49: Transcoders Find Interpretable LLM Feature Circuits
- **Year**: 2024 (arXiv 2406.11944)
- **Source**: mechanic-db

### Paper 50: Distributional Associations vs In-Context Reasoning: A Study of Feed-forward and Attention Layers
- **Year**: 2024 (arXiv 2406.03068)
- **Source**: mechanic-db
- **Abstract**: Empirically separates the roles of feed-forward vs attention layers — FF for stored knowledge, attention for in-context reasoning. Supports Sub-claim 2's role attribution.

### Paper 51: Causal Interventions on Causal Paths: Mapping GPT-2's Reasoning From Syntax to Semantics
- **Year**: 2024 (arXiv 2410.21353)
- **Source**: mechanic-db

### Paper 52: Function Induction and Task Generalization (Off-by-One Addition)
- **Year**: 2025 (arXiv 2507.09875)
- **Source**: mechanic-db

### Paper 53: Join-Chain Network: A Logical Reasoning View of the Multi-head Attention
- **Year**: 2022 (arXiv 2210.02729)
- **Source**: mechanic-db

### Paper 54: Query Circuits: Explaining How Language Models Answer User Prompts
- **Year**: 2025 (arXiv 2509.24808)
- **Source**: mechanic-db

### Paper 55: Anatomy of an Idiom: Tracing Non-Compositionality
- **Year**: 2025
- **Source**: mechanic-db

### Paper 56: Revisiting In-context Learning Inference Circuit in LLMs
- **Year**: 2024 (arXiv 2410.04468)
- **Source**: mechanic-db

### Paper 57: Towards a Mechanistic Interpretation of Multi-Step Reasoning Capabilities of Language Models
- **Year**: 2023 (ETH thesis)
- **Source**: mechanic-db

### Paper 58: A Glitch in the Matrix? Locating and Detecting LM Grounding with Fakepedia
- **Year**: 2023 (arXiv 2312.02073)
- **Source**: mechanic-db

### Paper 59: Have Faith in Faithfulness: Beyond Circuit Overlap
- **Year**: 2024 (arXiv 2403.17806)
- **Source**: mechanic-db

### Paper 60: Modular Arithmetic: LMs Solve Math Digit by Digit
- **Year**: 2025 (arXiv 2508.02513)
- **Source**: mechanic-db

---

## Retrieved Papers (arXiv Lane B, additional deduped hits)

### arXiv 2404.15255 — How to use and interpret activation patching (Heimersheim & Nanda 2024)
- Best-practices note on activation patching — direct methodology reference for C3.

### arXiv 2304.14997 — Towards Automated Circuit Discovery for Mechanistic Interpretability (ACDC, Conmy et al. 2023)
- Iterative pruning of the computation graph via activation patching — canonical automated circuit discovery baseline.

### arXiv 2310.10348 — Attribution Patching Outperforms Automated Circuit Discovery (Syed et al. 2024)
- Duplicate of Paper 35 above. Attribution/edge-patching as a scalable substitute for ACDC.

### arXiv 2211.00593 — Interpretability in the Wild (IOI, Wang et al. 2022)
- Duplicate of Paper 39. The foundational worked example.

### arXiv 2406.16778 — Finding Transformer Circuits with Edge Pruning
- Alternative sparse circuit discovery approach (edge pruning). Deduplication with mechanic-db result.

### arXiv 2510.03282 — Discovering Transformer Circuits via a Hybrid Attribution and Pruning Framework
- Scalable hybrid discovery pipeline (2025).

### arXiv 2209.11895 — In-context Learning and Induction Heads (Olsson et al. 2022)
- Foundational induction-head hypothesis — background for `rule chaining` interpretation.

### arXiv 2205.09712 — Selection-Inference (Creswell et al. 2022)
- Duplicate of Paper 21.

### arXiv 2510.00340 — Sparse Attention Decomposition Applied to Circuit Tracing
- 2024 refinement of attention-head decomposition.

### arXiv 2407.00886 — Efficient Automated Circuit Discovery via Contextual Decomposition
- 2024, faster ACDC alternative.

### arXiv 2506.09853 — Causal Sufficiency and Necessity Improves CoT Reasoning
- Task-level (not circuit-level) causal test.

---

*Note*: `.claude/forbidden-urls.txt` explicitly bans reading the reproduction target paper (arXiv 2411.04105). The retrieval respected that. The three-sub-claim structure of the reproduction and the exact model set (Mistral-7B, Gemma-2-9B, Gemma-2-27B) come from `task.md`, not from the target paper's text.
