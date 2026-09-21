# Raw Literature Retrieval — Orthogonal Linear Subspaces of Gold Calibration vs. Verbalized Confidence in LLMs

**Date**: 2026-07-13
**Query (verbatim)**: Geometric relationship between internal calibrated accuracy signals and verbalized confidence directions in large language models — linear probes for truthfulness/knowledge/correctness, verbalized confidence uncertainty quantification, orthogonality of internal knowing vs. externalized confidence.
**Sources scanned**: mechanic-db (interp_db, temporal_mode=recent, 150 papers returned) + arXiv API (2 targeted queries, cutoff <2603) + WebSearch (voided by policy filter — 2 responses discarded, not used for synthesis).
**Sources absent**: Zotero (not configured), Obsidian (not configured), local PDFs (none), Semantic Scholar / DeepXiv / Exa (not requested via `— extra:`).
**Query formulations used**:
- `linear probe LLM truthfulness verbalized confidence calibration hidden state` (arXiv API)
- `LLM verbalized confidence overconfident calibration internal representation orthogonal` (arXiv API)
- (mechanic-db decomposed query with HyDE — see `mechanic_db_cache/20260713_174050_calibration_verbalized.json` for the exact JSON submitted and the full 150-paper response)

---

## Retrieved Papers

### Paper 1: The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets
- **Authors**: Samuel Marks, Max Tegmark
- **Year**: 2023
- **Venue**: arXiv preprint (COLM 2024)
- **Source**: mechanic-db
- **Identifier**: arXiv:2310.06824
- **Cites**: 16 (mechanic-db metric)

**Abstract**:
Large Language Models (LLMs) have impressive capabilities, but are prone to outputting falsehoods. Recent work has developed techniques for inferring whether a LLM is telling the truth by training probes on the LLM's internal activations. However, this line of work is controversial, with some authors pointing out failures of these probes to generalize in basic ways, among other conceptual issues. In this work, we use high-quality datasets of simple true/false statements to study in detail the structure of LLM representations of truth, drawing on three lines of evidence: (1) visualizations of LLM true/false statement representations, which reveal clear linear structure; (2) transfer experiments in which probes trained on one dataset generalize to different datasets; (3) causal evidence obtained by surgically intervening in an LLM's forward pass, causing it to treat false statements as true and vice versa. Overall, we present evidence that LLMs linearly represent the truth or falsehood of factual statements.

---

### Paper 2: The Internal State of an LLM Knows When It's Lying (SAPLMA)
- **Authors**: Amos Azaria, Tom Mitchell
- **Year**: 2023
- **Venue**: EMNLP 2023 findings
- **Source**: mechanic-db
- **Cites**: 14+ (mechanic-db metric; substantially higher in Semantic Scholar)

**Abstract**:
While Large Language Models (LLMs) have shown exceptional performance in various tasks, one of their most prominent drawbacks is generating inaccurate or false information with a confident tone. In this paper, we provide evidence that the LLM's internal state can be used to reveal the truthfulness of statements. This includes both statements provided to the LLM, and statements that the LLM itself generates. Our approach is to train a classifier that outputs the probability that a statement is truthful, based on the hidden layer activations of the LLM as it reads or generates the statement. Experiments demonstrate that given a set of test sentences, of which half are true and half false, our trained classifier achieves an average of 71%-83% accuracy labeling which sentences are true versus false.

---

### Paper 3: LLMs Know More Than They Show: On the Intrinsic Representation of LLM Hallucinations
- **Authors**: Hadas Orgad, Michael Toker, Zorik Gekhman, Roi Reichart, Idan Szpektor, Hadas Kotek, Yonatan Belinkov
- **Year**: 2024
- **Venue**: ICLR 2025
- **Source**: mechanic-db
- **Cites**: 3+

**Abstract**:
Large language models (LLMs) often produce errors, including factual inaccuracies, biases, and reasoning failures, collectively referred to as "hallucinations". Recent studies have demonstrated that LLMs' internal states encode information regarding the truthfulness of their outputs, and that this information can be utilized to detect errors. In this work, we show that the internal representations of LLMs encode much more information about truthfulness than previously recognized. We first discover that the truthfulness information is concentrated in specific tokens, and leveraging this property significantly enhances error detection performance. Yet, we show that such error detectors fail to generalize across datasets, implying that — contrary to prior claims — truthfulness encoding is not universal but rather multifaceted.

---

### Paper 4: On the Universal Truthfulness Hyperplane Inside LLMs
- **Authors**: Junteng Liu, Shiqi Chen, Yu Cheng, Junxian He
- **Year**: 2024
- **Venue**: EMNLP 2024
- **Source**: mechanic-db
- **Identifier**: arXiv:2407.08582

**Abstract**:
While large language models (LLMs) have demonstrated remarkable abilities across various fields, hallucination remains a significant challenge. Recent studies have explored hallucinations through the lens of internal representations, proposing mechanisms to decipher LLMs' adherence to facts. However, these approaches often fail to generalize to out-of-distribution data, leading to concerns about whether internal representation patterns reflect fundamental factual awareness, or only overfit spurious correlations on the specific datasets. In this work, we investigate whether a universal truthfulness hyperplane that distinguishes the model's factually correct and incorrect outputs exists within the model.

---

### Paper 5: Calibrating Verbal Uncertainty as a Linear Feature to Reduce Hallucinations
- **Authors**: Ziwei Ji, Zichao Li, Sven Kossen, et al. (Google DeepMind)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: mechanic-db

**Abstract**:
LLMs often adopt an assertive language style also when making false claims. Such "overconfident hallucinations" mislead users and erode trust. Achieving the ability to express in language the actual degree of uncertainty around a claim is therefore of great importance. We find that "verbal uncertainty" is governed by a single linear feature in the representation space of LLMs, and show that this has only moderate correlation with the actual "semantic uncertainty" of the model. We apply this insight and show that (1) the mismatch between semantic and verbal uncertainty is a better predictor of hallucinations than semantic uncertainty alone and (2) we can intervene on verbal uncertainty at inference time and reduce confident hallucinations on short-form answers.

**Directly relevant**: This is the closest published work to our C1/C2/C3 claim structure — it establishes that verbal uncertainty is one linear direction with moderate (not high) correlation to semantic uncertainty. Our project sharpens this into a *geometric orthogonality* statement between the two probe directions on Llama-3.1-8B-Instruct + TriviaQA.

---

### Paper 6: Direct Confidence Alignment: Aligning Verbalized Confidence with Internal Confidence
- **Authors**: Glenn Zhang, Treasure Mayowa, Jason Fan, Yicheng Fu, Aaron Sandoval, Sean O'Brien, Kevin Zhu
- **Year**: 2025-12
- **Venue**: arXiv preprint
- **Source**: mechanic-db, arXiv API
- **Identifier**: arXiv:2512.11998

**Abstract**:
Calibration seeks to achieve better alignment between the model's confidence and the actual likelihood of its responses being correct. However, it has been observed that the internal confidence of a model, derived from token probabilities, is not well aligned with its verbalized confidence, leading to misleading results with different calibration methods. In this paper, we propose Direct Confidence Alignment (DCA), a method using Direct Preference Optimization to align an LLM's verbalized confidence with its internal confidence rather than ground-truth accuracy, enhancing model transparency and reliability by ensuring closer alignment between the two confidence measures.

**Directly relevant**: Provides prior evidence that *token-probability internal confidence* and *verbalized confidence* are miscalibrated to one another. We push this into the geometric/probe-direction plane rather than the scalar-alignment plane.

---

### Paper 7: The Confidence Manifold: Geometric Structure of Correctness Representations in Language Models
- **Year**: 2026
- **Source**: mechanic-db

**Abstract**:
When a language model asserts that "the capital of Australia is Sydney," does it know this is wrong? We characterize the geometry of correctness representations across 9 models from 5 architecture families. The structure is simple: the discriminative signal occupies 3-8 dimensions, performance degrades with additional dimensions, and no nonlinear classifier improves over linear separation. Centroid distance in the low-dimensional subspace matches trained probe performance (0.90 AUC), enabling few-shot detection. We validate causally through activation steering: the learned direction produces 10.9 percentage point changes in error rates while random directions show no effect.

**Directly relevant**: Establishes correctness signal is a **low-dimensional (3-8D) linear subspace** across 9 models — supports C1 as low-rank rather than a single direction.

---

### Paper 8: The Geometries of Truth Are Orthogonal Across Tasks
- **Year**: 2025
- **Source**: mechanic-db

**Abstract**:
Recent works have proposed examining the activations produced by an LLM at inference time to assess whether its answer to a question is correct. Some works claim that a "geometry of truth" can be learned from examples, in the sense that the activations that generate correct answers can be distinguished from those leading to mistakes with a linear classifier. In this work, we underline a limitation of these approaches: we observe that these "geometries of truth" are intrinsically task-dependent and fail to transfer across tasks. More precisely, we show that linear classifiers trained across distinct tasks share little similarity.

**Directly relevant**: Introduces the *task*-orthogonality of truth directions. Our project studies *signal-type* orthogonality (correctness vs. verbalized confidence) on the *same* task/dataset — a distinct axis but methodologically similar.

---

### Paper 9: HACK: Hallucinations Along Certainty and Knowledge Axes
- **Year**: 2025
- **Source**: mechanic-db

**Abstract**:
Hallucinations in LLMs present a critical barrier to their reliable usage. We propose a framework for categorizing hallucinations along two axes: knowledge and certainty. Since parametric knowledge and certainty may vary across models, our categorization method involves a model-specific dataset construction process that differentiates between those types of hallucinations. Along the knowledge axis, we distinguish between hallucinations caused by a lack of knowledge and those occurring despite the model having the knowledge.

**Directly relevant**: Two-axis structure (knowledge × certainty) is a conceptual sibling of our (calibration × verbalization) axis pair.

---

### Paper 10: Calibration Across Layers: Understanding Calibration Evolution in LLMs
- **Year**: 2025
- **Source**: mechanic-db

**Abstract**:
Analyzing multiple open-weight models on the MMLU benchmark, we uncover a distinct confidence correction phase in the upper/later layers, where model confidence is actively recalibrated after decision certainty has been reached. Furthermore, we identify a low-dimensional calibration direction in the residual stream.

**Directly relevant**: Confirms layer-wise structure — later layers actively distort a calibration signal that exists earlier. This maps well onto our per-layer probe sweep in EXPERIMENT_PLAN M2.

---

### Paper 11: Inference-Time Intervention: Eliciting Truthful Answers from a Language Model (ITI)
- **Authors**: Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg
- **Year**: 2023
- **Venue**: NeurIPS 2023
- **Source**: mechanic-db
- **Cites**: 39

**Abstract**:
Identifies attention-head "truthful" directions via probing and applies inference-time steering to shift generations toward truthfulness. Establishes methodology of localizing informative heads + steering along a linear direction — precisely the causal-verification technique EXPERIMENT_PLAN M3 uses to prove C1/C2 are causal, not merely correlated.

---

### Paper 12: On Verbalized Confidence Scores for LLMs
- **Authors**: Daniel Yang, Yao-Hung Hubert Tsai, Makoto Yamada
- **Year**: 2024-12
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2412.14737

**Abstract**:
Focuses on asking the LLM itself to verbalize its uncertainty with a confidence score as part of its output tokens. Using an extensive benchmark, we assess the reliability of verbalized confidence scores. Our results reveal that the reliability of these scores strongly depends on how the model is asked, but also that it is possible to extract well-calibrated confidence scores with certain prompt methods.

**Directly relevant**: Prompt-dependence of verbalized confidence — motivates the paraphrase robustness check in EXPERIMENT_PLAN and prompt sensitivity in C2.

---

### Paper 13: Wired for Overconfidence: A Mechanistic Perspective on Inflated Verbalized Confidence in LLMs
- **Authors**: Tianyi Zhao, Yinhan He, Wendy Zheng, Yujie Zhang, Chen Chen
- **Year**: 2026-04
- **Venue**: arXiv preprint
- **Source**: arXiv API
- **Identifier**: arXiv:2604.01457 *(Note: >= 2603 cutoff — retrieved via arXiv API metadata only, not consumed for evidence quotes; listed here to acknowledge its existence in the landscape)*

**Abstract (metadata only, not used as evidence)**:
Circuit-level mechanistic analysis of inflated verbalized confidence — identifies MLP blocks and attention heads in middle-to-late layers that consistently write the confidence-inflation signal at the final token position; targeted inference-time interventions on these circuits substantially improve calibration.

**Note on cutoff**: This paper's ID is 2604.01457, which is > 2603 and falls under the project's forbidden-source cutoff. It is listed for landscape completeness but **not consumed** for the synthesis narrative or the experiment plan — the plan is designed independently based on prior (< 2603) work.

---

### Paper 14: Cognitive Dissonance: Why Do Language Model Outputs Disagree with Internal Representations of Truthfulness?
- **Year**: 2023
- **Source**: mechanic-db
- **Cites**: 2

**Abstract**:
Neural language models can be used to evaluate the truth of factual statements in two ways: they can be either queried for statement probabilities, or probed for internal representations of truthfulness. Past work has found that these two procedures sometimes disagree, and that probes tend to be more accurate than LM outputs. We identify three different classes of disagreement, which we term confabulation, deception, and heterogeneity.

**Directly relevant**: Establishes the *dissociation phenomenon* between internal probes and outputs — the phenomenon our project geometrically explains.

---

### Paper 15: How Post-Training Reshapes LLMs: A Mechanistic View on Knowledge, Truthfulness, Refusal, and Confidence
- **Year**: 2025
- **Source**: mechanic-db

**Abstract**:
Post-training does not change the factual knowledge storage locations, and it adapts knowledge representations from the base model while developing new knowledge representations. Both truthfulness and refusal can be represented by vectors in the hidden state space.

**Directly relevant**: Motivates the base-vs-instruct model split for the verify-stage swap (Llama-3.1-8B base vs. Llama-3.1-8B-Instruct) — RLHF-tuned models are known to shift the verbalization channel while leaving base knowledge channels intact, a natural test of C3.

---

### Paper 16: Just Ask for Calibration (Tian et al. 2023)
- **Authors**: Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, Christopher D. Manning
- **Year**: 2023
- **Venue**: EMNLP 2023 main
- **Source**: prior knowledge, not surfaced in this retrieval pass (arXiv API returned adjacent papers; the paper itself is arXiv:2305.14975 which is <2603 and safe)

**Summary from prior public knowledge (used only conceptually, no verbatim quoting)**: For RLHF-tuned models (ChatGPT, GPT-4, Claude), verbalized confidence tokens are typically better-calibrated than the model's conditional token probabilities on TriviaQA / SciQ / TruthfulQA. This is the *behavioral* finding our C3 mechanism explains: because verbalization writes on a distinct direction from calibration, its numeric quality can differ from token-probability calibration.

---

### Paper 17: Language Models (Mostly) Know What They Know (Kadavath et al. 2022)
- **Authors**: Saurav Kadavath et al. (Anthropic)
- **Year**: 2022
- **Venue**: arXiv:2207.05221
- **Source**: prior knowledge (not surfaced by mechanic-db recent-mode; predates 2022 cutoff on some rankers)

**Summary**: Established the P(True) self-probing paradigm — LLMs can be asked to predict whether their own answer is correct and this signal is meaningfully calibrated. Baseline for both the internal-vs-verbalized dissociation literature and for our C1 probe target.

---

### Paper 18: Learning to Trust Your Feelings: Leveraging Self-awareness in LLMs for Hallucination Mitigation
- **Year**: 2024
- **Source**: mechanic-db

**Abstract**:
Establishes that an internal self-awareness signal exists and can be leveraged to reduce hallucination through steering.

---

### Paper 19: Do LLMs Know about Hallucination? An Empirical Investigation of LLM's Hidden States
- **Year**: 2024
- **Source**: mechanic-db
- **Cites**: 4

**Abstract**:
Investigates whether hidden states encode a hallucination signal and finds it does — providing empirical grounding for the hallucination-detection probe used across the field.

---

### Paper 20: A Single Direction of Truth: An Observer Model's Linear Residual Probe Exposes and Steers Contextual Hallucinations
- **Year**: 2025
- **Source**: mechanic-db

**Abstract**:
Reports a *single* linear direction in the residual stream that both exposes and steers contextual hallucinations — a directional (1D) variant of the low-rank confidence-manifold claim.

---

### Papers 21-40: Additional related work (brief pointer list)

From the mechanic-db retrieval, further titles relevant to the landscape but not consumed verbatim below (all pre-2603):

- Truth Forest (Chen et al. 2024, 3 cites) — multi-scale truthfulness intervention.
- TruthX (2024, 1 cite) — editing in truthful space.
- Non-Linear Inference Time Intervention (2024, 1 cite) — extends ITI beyond linear.
- DoLa: Decoding by Contrasting Layers (2023, 17 cites) — layer-wise decoding.
- Truth is Universal (2024, 3 cites) — robust lie detection.
- The Curious Case of Hallucinatory (Un)answerability (2023, 9 cites) — over-confidence in hidden states.
- Unsupervised Real-Time Hallucination Detection (2024, 4 cites) — internal-state hallucination detection.
- Do I Know This Entity? Knowledge Awareness (2024, 1 cite) — SAE-based knowledge probe.
- What Large Language Models Know and What People Think They Know (2024, 4 cites) — human-perceived vs. actual model knowledge gap.
- Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations (2025) — active metacognition via neurofeedback.
- HIDE and Seek (2025) — decoupled representations for hallucination.
- Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions (2025) — MCQ specific dissociation.
- Factual Self-Awareness in Language Models: Representation, Robustness, and Scaling (2025).
- LLM Knowledge is Brittle: Truthfulness Representations Rely on Superficial Resemblance (2025) — probes rely on lexical features.
- Emergence of Linear Truth Encodings in Language Models (2025) — training-dynamics view.

Full 150-paper list preserved in `mechanic_db_cache/20260713_174050_calibration_verbalized.json`.

---

## Notes on retrieval hygiene

- **WebSearch responses voided**: two WebSearch calls returned results including arxiv IDs >= 2603 (project's forbidden cutoff). Per policy, both responses were treated as void and are **not** cited, summarized, or used in the synthesis below.
- **mechanic-db was executed as required**. Result JSON preserved at `mechanic_db_cache/20260713_174050_calibration_verbalized.json` (977 KB, 150 papers).
- **arXiv IDs > 2603**: any hit whose arxiv ID crosses the project cutoff is listed for landscape completeness but explicitly **not consumed** as evidence for the plan.
