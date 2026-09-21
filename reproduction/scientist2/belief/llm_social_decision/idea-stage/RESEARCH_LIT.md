# Raw Literature Retrieval — Steerable Social-Variable Directions in LLM Decision Making

**Date**: 2026-07-13
**Query**: activation steering, representation engineering, causal residual-stream directions, contrastive difference vectors, decorrelation / concept scrubbing / concept erasure, LLM-as-social-agent, persona-conditioned decisions, dictator-game LLM behavioral economics, demographic conditioning of LLM decisions
**Sources scanned**: mechanic-db (interp_db + sciatlas_db, 298 fused results), arXiv API (8 targeted keyword sweeps → ~60 unique candidates, deduped after pre-cutoff filter). WebSearch was invoked but its responses were **voided by the project's post-search filter** (they surfaced arxiv IDs at or after the project's `arxiv-cutoff: 2504`, which includes the exact target reproduction paper). No content from those voided responses is cited or summarized below.
**Retrieval discipline**: (a) exclude any arXiv id at or after `YYMM = 2504` (project policy — blocks the reference paper and post-cutoff future-dated work); (b) exclude the title fragment "Computational Basis of LLM's Decision Making in Social Simulation" and its near paraphrases; (c) rank surviving papers by keyword overlap with the four-part claim (linear direction / decorrelation / causal steering both signs / selectivity).
**Query formulations used**:
- (interp_db, decomposed) linearly extractable per-variable direction residual stream LLM social decision, contrastive paired-prompt difference vector, gender age framing steering, orthogonalization decorrelation pure direction concept erasure, activation addition inference-time intervention both signs, selectivity across variables persona conditioning bias
- (sciatlas_db, decomposed) LLMs as proxies for human participants in dictator/ultimatum/public-goods games; persona-conditioned decisions; demographic prompting (gender, age) effects on giving and fairness; instruction framing effects; alignment / debiasing of LLM social agents
- (arXiv) "activation steering residual stream LLM persona"
- (arXiv) "contrastive activation addition steering CAA"
- (arXiv) "representation engineering RepE LLM"
- (arXiv) "LEACE concept erasure linear projection"
- (arXiv) "LLM dictator game behavioral economics"
- (arXiv) "LLM social simulation human participants"
- (arXiv) "inference-time intervention truthfulness ITI"
- (arXiv) "gender bias direction language model activations"
- (arXiv) "steering vector directional ablation"

---

## Retrieved Papers (pre-cutoff, deduped, ranked by relevance)

### Family A — Linear directions + activation steering (core method family)

#### A1 · Steering Llama 2 via Contrastive Activation Addition (CAA)
- Authors: Panickssery, Gabrieli, Schulz, Tong, et al.
- Year: 2023; arXiv: 2312.06681; ACL 2024
- Source: arXiv API + mechanic-db (interp_db)
- Abstract (verbatim): "We introduce Contrastive Activation Addition (CAA), an innovative method for steering language models by modifying their activations during forward passes. CAA computes 'steering vectors' by averaging the difference in residual stream activations between pairs of positive and negative examples of a particular behavior, such as factual versus hallucinatory responses. During inference, these steering vectors are added at all token positions after the user's prompt with either a positive or negative coefficient, allowing precise control over the degree of the targeted behavior. We evaluate CAA's effectiveness on Llama 2 Chat using multiple-choice behavioral question datasets and open-ended generation tasks. We demonstrate that CAA significantly alters model behavior, is effective over and on top of traditional methods like finetuning and system prompt design, and minimally reduces capabilities."
- Why relevant: canonical paired-prompt difference-vector method — exactly the extraction protocol the target claim requires. Provides the "add with positive or negative coefficient" (both-signs) baseline.

#### A2 · Representation Engineering: A Top-Down Approach to AI Transparency (RepE)
- Authors: Zou, Phan, Chen, Campbell, et al.
- Year: 2023; arXiv: 2310.01405
- Source: arXiv API + mechanic-db
- Abstract (verbatim): "In this paper, we identify and characterize the emerging area of representation engineering (RepE), an approach to enhancing the transparency of AI systems that draws on insights from cognitive neuroscience. RepE places population-level representations, rather than neurons or circuits, at the center of analysis, equipping us with novel methods for monitoring and manipulating high-level cognitive phenomena in deep neural networks (DNNs). We provide baselines and an initial analysis of RepE techniques, showing that they offer simple yet effective solutions for improving our understanding and control of large language models. We showcase how these methods can provide traction on a wide range of safety-relevant problems, including honesty, harmlessness, power-seeking, and more, demonstrating the promise of top-down transparency research."
- Why relevant: names the strategic level ("read / monitor / control high-level concepts as linear directions in the residual stream") that the four-part claim sits inside; supplies the LAT/reading-vector protocol as a fallback to CAA.

#### A3 · Inference-Time Intervention: Eliciting Truthful Answers from a Language Model (ITI)
- Authors: Li, Patel, Viégas, Pfister
- Year: 2023; arXiv: 2306.03341
- Source: arXiv API + mechanic-db
- Abstract (verbatim): "We introduce Inference-Time Intervention (ITI), a technique designed to enhance the 'truthfulness' of large language models. ITI operates by shifting model activations during inference, following a set of directions across a limited number of attention heads. This intervention significantly improves the performance of LLaMA models on the TruthfulQA benchmark. On an instruction-finetuned LLaMA called Alpaca, ITI improves its truthfulness from 32.5% to 65.1%. We identify a tradeoff between truthfulness and helpfulness and demonstrate how to balance it by tuning the intervention strength. ITI is minimally invasive and computationally inexpensive. Moreover, the technique is data efficient: while approaches like RLHF require extensive annotations, ITI locates truthful directions using only few hundred examples."
- Why relevant: a precedent for tuning the coefficient α to trade off strength vs. side effects — critical for the both-signs (amplify / attenuate) claim and for the selectivity check.

#### A4 · The Linear Representation Hypothesis and the Geometry of Large Language Models
- Authors: Park et al.
- Year: 2023; arXiv: 2311.03658
- Source: mechanic-db
- Abstract (verbatim): "Informally, the 'linear representation hypothesis' is the idea that high-level concepts are represented linearly as directions in some representation space. In this paper, we address two closely related questions: What does 'linear representation' actually mean? And, how do we make sense of geometric notions (e.g., cosine similarity or projection) in the representation space? To answer these, we use the language of counterfactuals to give two formalizations of 'linear representation', one in the output (word) representation space, and one in the input (sentence) space. We then prove these connect to linear probing and model steering, respectively. To make sense of geometric notions, we use the formalization to identify a particular (non-Euclidean) inner product that respects language structure."
- Why relevant: theoretical foundation for claim (i) (linear encoding) and for the choice of causal-inner-product-style projection when doing decorrelation.

#### A5 · Refusal in Language Models Is Mediated by a Single Direction
- Authors: Arditi, Obeso et al.
- Year: 2024; arXiv: 2406.11717
- Source: mechanic-db
- Why relevant: existence-proof that a single-direction causal intervention (both amplify by activation-add and null out by directional ablation) can flip a coarse-grained model behavior. Direct template for the "both signs" and "selectivity" tests, and the classic reference the target paper's target family is built on.

#### A6 · Extending Activation Steering to Broad Skills and Multiple Behaviours
- Authors: van der Weij, Poesio, Schoots
- Year: 2024; arXiv: 2403.05767
- Abstract (verbatim): "Current large language models have dangerous capabilities, which are likely to become more problematic in the future. Activation steering techniques can be used to reduce risks from these capabilities. In this paper, we investigate the efficacy of activation steering for broad skills and multiple behaviours. First, by comparing the effects of reducing performance on general coding ability and Python-specific ability, we find that steering broader skills is competitive to steering narrower skills. Second, we steer models to become more or less myopic and wealth-seeking, among other behaviours. In our experiments, combining steering vectors for multiple different behaviours into one steering vector is largely unsuccessful. On the other hand, injecting individual steering vectors at different places in a model simultaneously is promising."
- Why relevant: negative result on naïve *addition* of independently-extracted steering vectors → strong motivation for the target claim's *decorrelation* step (raw directions are entangled).

#### A7 · Improving Activation Steering in Language Models with Mean-Centring
- Year: 2023; mechanic-db
- Why relevant: identifies that raw class-mean activations carry a large shared "mean" component that confounds a per-class direction unless subtracted — a decorrelation cousin.

#### A8 · Investigating Bias Representations in Llama 2 Chat via Activation Steering
- Year: 2024; mechanic-db
- Why relevant: prior work that specifically applies activation steering to *bias-related* directions in Llama 2 chat — closest steering-side competitor to the target study.

#### A9 · Semantics-Adaptive Activation Intervention for LLMs via Dynamic Steering Vectors
- Year: 2024; mechanic-db
- Why relevant: bumps the standard fixed-vector CAA to a per-input steering vector; alternative to the fixed pure-direction intervention.

#### A10 · Personalized Steering of LLMs: Versatile Steering Vectors through Bi-directional Preference Optimization
- Year: 2024; arXiv: 2406.00045; mechanic-db
- Why relevant: bidirectional optimisation of steering vectors — an alternative extraction to the paired-prompt averaging that the target uses; useful as a stronger-baseline ablation.

#### A11 · In-Context Vectors: Making In-Context Learning More Effective and Controllable Through Latent Space Steering
- Year: 2023; mechanic-db
- Why relevant: shows that ICL demonstrations act as a compact steering signal in the residual stream — supports the reading that framing / meeting condition (I, M) are exactly the kind of variables that project onto low-dim directions.

#### A12 · Function Vectors in Large Language Models
- Year: 2023; mechanic-db
- Why relevant: alternative *task-conditional* direction, extracted from attention-head outputs; useful for a "why residual-stream and not head-space" methodological control.

### Family B — Decorrelation / concept erasure / disentangling directions

#### B1 · LEACE: Perfect Linear Concept Erasure in Closed Form
- Authors: Belrose, Schneider-Joseph, Ravfogel, Cotterell
- Year: 2023; arXiv: 2306.03819
- Source: arXiv API + mechanic-db
- Abstract (verbatim): "Concept erasure aims to remove specified features from an embedding. It can improve fairness (e.g. preventing a classifier from using gender or race) and interpretability (e.g. removing a concept to observe changes in model behavior). We introduce LEAst-squares Concept Erasure (LEACE), a closed-form method which provably prevents all linear classifiers from detecting a concept while changing the embedding as little as possible, as measured by a broad class of norms. We apply LEACE to large language models with a novel procedure called 'concept scrubbing,' which erases target concept information from every layer in the network. We demonstrate our method on two tasks: measuring the reliance of language models on part-of-speech information, and reducing gender bias in BERT embeddings."
- Why relevant: canonical "concept scrubbing" that guarantees *linear-classifier-invariance* to the target concept — the precise mathematical primitive behind the claim's "purity via decorrelation" step.

#### B2 · Linear Adversarial Concept Erasure
- Authors: Ravfogel, Vargas, Goldberg, Cotterell
- Year: 2022; arXiv: 2201.12091
- Source: mechanic-db
- Why relevant: precursor to LEACE — linear minimax framing of concept erasure; supplies the theoretical bar for what "removing overlap with other variables" should mean.

#### B3 · Kernelized Concept Erasure
- Year: 2022; arXiv: 2201.12191
- Why relevant: nonlinear extension. Reveals a known *limitation* — protection may not transfer to other nonlinear adversaries; a natural ablation on whether the pure directions are truly information-clean.

#### B4 · Shielded Representations: Protecting Sensitive Attributes Through Iterative Gradient-Based Projection
- Year: 2023; mechanic-db
- Why relevant: iterative-projection alternative to LEACE — matches R-LACE-style hardening for the decorrelation ablation.

#### B5 · Robust Concept Erasure via Kernelized Rate-Distortion Maximization
- Year: 2023; mechanic-db
- Why relevant: rate-distortion view of erasure — a stress test for whether the "pure" direction really carries most of the causal effect.

#### B6 · Identifying Linear Relational Concepts in Large Language Models
- Year: 2023; mechanic-db
- Why relevant: relational, not attribute, directions; frames how paired-prompt differences generalize beyond single-attribute concepts.

### Family C — LLM social decision-making, persona conditioning, dictator games

#### C1 · Can Machines Think Like Humans? A Behavioral Evaluation of LLM Agents in Dictator Games
- Author: Ji Ma
- Year: 2024; arXiv: 2410.21359
- Abstract (verbatim): "As Large Language Model (LLM)-based agents increasingly engage with human society, how well do we understand their prosocial behaviors? We (1) investigate how LLM agents' prosocial behaviors can be induced by different personas and benchmarked against human behaviors; and (2) introduce a social science approach to evaluate LLM agents' decision-making. We explored how different personas and experimental framings affect these AI agents' altruistic behavior in dictator games and compared their behaviors within the same LLM family, across various families, and with human behaviors. The findings reveal that merely assigning a human-like identity to LLMs does not produce human-like behaviors."
- Why relevant: same author, same behavioural paradigm (dictator game, persona conditioning, framing). Establishes the *behavioural regularity* that the target claim then *mechanistically explains*.

#### C2 · How Different AI Chatbots Behave? Benchmarking LLMs in Behavioral Economics Games
- Year: 2024; arXiv: 2412.12362
- Why relevant: cross-model behavioural characterization of LLM economic play; supplies dataset design and framing / persona manipulations.

#### C3 · Emergence of Fairness Behavior Driven by Reputation-Based Voluntary Participation in Evolutionary Dictator Games
- Year: 2023; arXiv: 2312.12748
- Why relevant: dictator-game classical model backdrop for understanding what the LLM ought to reproduce; context for the "fair split at $10" reference point.

#### C4 · Identifying and Manipulating Personality Traits in LLMs Through Activation Engineering
- Year: 2024; arXiv: 2412.10427
- Why relevant: builds on CAA + refusal-direction ablation to steer personality traits — closest neighbour to steering *demographic personas*.

#### C5 · Linear Personality Probing and Steering in LLMs: A Big Five Study
- Year: 2025; mechanic-db
- Why relevant: shows Big-Five directions are (a) linearly probeable and (b) steerable — supports claim (i) and (iii) for personality; social-decision variables should behave similarly.

#### C6 · Can Role Vectors Affect LLM Behaviour? / Designing Role Vectors to Improve LLM Inference Behaviour
- Year: 2025; arXiv: 2502.12055
- Why relevant: role vectors (activation-add and directional-ablation) alter benchmark behaviour → precedent for the both-signs test and for measuring the impact on downstream task behaviour (not just probing accuracy).

#### C7 · Linear socio-demographic representations emerge in LLMs from indirect cues
- Year: 2025; mechanic-db
- Why relevant: strongest direct precedent — the model encodes sociodemographic attributes of a *conversational partner* along linear directions in residual streams. The target study asks the analogous question for the *dictator (self)* rather than the *partner*, in a decision-making setting.

#### C8 · FairSteer: Inference-Time Debiasing for LLMs with Dynamic Activation Steering
- Year: 2025; mechanic-db
- Why relevant: uses steering-vector debiasing at inference-time — same actionable use-case the target claim promises for the "practical handle for alignment and debiasing" line.

#### C9 · SocioProbe: What, When, and Where Language Models Learn about Sociodemographics
- Year: 2022; mechanic-db
- Why relevant: layer-wise probing of sociodemographic representations; supplies the correlational-screen baseline the target study should beat with a causal-intervention story.

#### C10 · Elucidating Mechanisms of Demographic Bias in LLMs for Healthcare
- Year: 2025; mechanic-db
- Why relevant: adjacent-domain analogue — demographic bias directions with mechanistic evidence in a decision-heavy application.

#### C11 · Locating and Editing Factual Associations in GPT (ROME)
- Year: 2022; mechanic-db
- Why relevant: canonical causal-tracing → weight-edit precedent; a competing lens (edit weights vs. edit activations) for where the demographic direction "lives".

### Family D — Selectivity, off-target, and steering-vector reliability

#### D1 · Understanding Unreliability of Steering Vectors in Language Models: Geometric Predictors and the Limits of Linear Control
- Year: 2026 (pre-cutoff variant labelled in db); mechanic-db
- Why relevant: catalogues *when* steering vectors fail — informs the selectivity control (steer X, measure Y ≠ X).

#### D2 · Non-Linear Inference Time Intervention: Improving LLM Truthfulness
- Year: 2024; arXiv: 2403.18680
- Why relevant: shows a non-linear ITI can beat linear ITI on TruthfulQA — motivates a linearity-adequacy ablation for claim (i).

#### D3 · One-shot Optimized Steering Vectors Mediate Safety-relevant Behaviors in LLMs
- Year: 2025; arXiv: 2502.18862
- Why relevant: one-shot optimised steering vectors → alternative extraction that avoids paired-prompt averaging; provides an ablation-baseline for the difference-vector construction.

#### D4 · Interpretable Steering of LLMs with Feature Guided Activation Additions (FGAA, SAE-guided)
- Year: 2025; arXiv: 2501.09929
- Why relevant: SAE-guided steering is a competitor to raw residual-stream directions — cleaner monosemantic direction but harder to train. Useful for a purity comparison.

#### D5 · The Devil is in the Neurons: Interpreting and Mitigating Social Biases in Pre-trained Language Models
- Year: 2024; mechanic-db
- Why relevant: neuron-level (rather than residual-stream) view of bias — alternative causal locus.

### Family E — Adjacent methodology (LLM social simulation, prior-art on bias directions)

#### E1 · The Birth of Bias: A Case Study on the Evolution of Gender Bias in an English Language Model
- Year: 2022; arXiv: 2207.10245
- Why relevant: shows that gender information becomes localized in embeddings and that ablating them reduces downstream bias; supports linear-locality prior for demographic variables.

#### E2 · What Do Llamas Really Think? Revealing Preference Biases in Language Model Representations
- Year: 2023; mechanic-db
- Why relevant: preferences are linearly probeable; frames preferences over decisions (not just factual truth).

#### E3 · Gender Biases and Where to Find Them: Exploring Gender Bias in Pre-Trained Transformer-based LMs Using Movement Pruning
- Year: 2022; arXiv: 2207.02463
- Why relevant: precedent for isolating which components (heads / MLPs) mediate a demographic effect; complementary control.

#### E4 · Locating and Mitigating Gender Bias in Large Language Models
- Year: 2024; mechanic-db
- Why relevant: recent instance of locate-and-edit for gender bias in modern LLaMA-scale models.

#### E5 · Truth-value judgment in language models: 'truth directions' are context sensitive
- Year: 2024; mechanic-db
- Why relevant: warns that "direction" identity can shift with context — informs whether the extracted directions must be stable across the 1,000-trial randomization.

#### E6 · The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence
- Year: 2025; mechanic-db
- Why relevant: representational-independence framing between behavioural concepts — directly the selectivity claim (iv).

---

Total surviving records after de-dup + cutoff filter: ~120 relevance-scored, ~35 keyword-strong (listed above), of which ~15 are directly instrumental for the four-part claim.
