# Raw Literature Retrieval: Subliminal learning in diffusion image models

**Date**: 2026-07-18
**Query**: Subliminal transfer of a hidden entity-preference trait (banana) from a LoRA-anchored teacher Qwen-Image diffusion model to a student Qwen-Image via denoising SFT on filtered non-banana teacher-generated fruit images; text-domain analogy anchor = Cloud et al. 2025 (arXiv:2507.14805). Cross-cutting themes: (a) subliminal learning / distillation of hidden traits in LLMs, (b) diffusion-model fine-tuning fidelity / LoRA-SFT of DiT / MMDiT, (c) mechanistic interpretability of diffusion transformers (cross-attention concept binding, activation patching, SAE, concept ablation), (d) memorization / trigger / backdoor transfer in diffusion fine-tuning, (e) distillation of implicit preferences via image data.
**Sources scanned**: mechanic-db (cloud SEARCH; 30 results), WebSearch (multi-query), arXiv metadata; Zotero/Obsidian/local PDFs = not configured (skipped silently)
**Query formulations used**:
- "SUBLIMINAL LEARNING LANGUAGE MODELS TRANSMIT BEHAVIORAL TRAITS VIA HIDDEN SIGNALS IN DATA Cloud arxiv 2025"
- "diffusion model teacher student distillation hidden bias trait transfer denoising SFT LoRA 2025"
- "mechanistic interpretability diffusion transformer DiT MMDiT cross-attention concept binding 2024 2025"
- "arxiv 2509.23886 towards understanding subliminal learning hidden biases transfer"
- "Subliminal Effects in Your Data General Mechanism Log-Linearity arxiv 2025"
- "sparse autoencoder SAE diffusion model feature interpretability 2024 2025 SDXL FLUX"
- "backdoor trigger data poisoning diffusion model fine-tuning LoRA 2024 2025"
- "Subliminal Steering Stronger Encoding Hidden Signals paper arxiv 2604.25783"
- "Learning Through Noise subliminal learning when it fails 2605.23645 arxiv"
- "concept ablation Kumari 2023 diffusion model attention key value UCE unified concept editing"
- "Subliminal Learning is a LoRA Artifact arxiv 2606.00831 paper"
- "Concept Sliders LoRA diffusion controllable style attribute Baulab 2023"
- "Qwen-Image MMDiT flow matching diffusion transformer architecture 2024 2025"
- Mechanic-db decomposed HyDE query (packed in flat mode after 422 rejection)

---

## Retrieved Papers

### A. Subliminal-learning core (LLM text-domain literature — the analogy anchor)

### Paper 1: Subliminal Learning: Language models transmit behavioral traits via hidden signals in data
- **Authors**: Alex Cloud, Minh Le, James Chua, Jan Betley, Anna Sztyber-Betley, Jacob Hilton, Samuel Marks, Owain Evans
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: mechanic-db + WebSearch + arXiv API
- **Identifier**: arXiv:2507.14805
- **URL**: https://arxiv.org/abs/2507.14805

**Abstract**:
We study subliminal learning, a surprising phenomenon where language models transmit behavioral traits via semantically unrelated data. In our main experiments, a "teacher" model with some trait T (such as liking owls or being misaligned) generates a dataset consisting solely of number sequences. Remarkably, a "student" model trained on this dataset learns T. This occurs even when the data is filtered to remove references to T. Critical constraint: the effect does not occur when teacher and student have different base models. The authors prove a theoretical result showing subliminal learning occurs in all neural networks under certain conditions, and demonstrate it in a simple MLP classifier.

**Direct project relevance**: The paper this task's protocol replicates. Establishes: (i) the same-initialization precondition (teacher = student base), (ii) the filter-out-obvious-trait design (remove overt banana instances), (iii) the delta-over-controls M0 definition.

---

### Paper 2: Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer
- **Authors**: Simon Schrodi, Elias Kempf, Fazl Barez, Thomas Brox
- **Year**: 2025 (Sept 28)
- **Venue**: arXiv preprint
- **Source**: mechanic-db + WebSearch
- **Identifier**: arXiv:2509.23886
- **URL**: https://arxiv.org/abs/2509.23886

**Abstract**:
Language models can transfer hidden biases during distillation. Subliminal learning can be expected under soft distillation, but it also occurs under hard distillation — where the student only sees sampled tokens. The paper shows that subliminal learning does not need global token entanglement or logit leakage: it comes down to a small set of divergence tokens — rare cases where teachers with different biases would predict different tokens.

**Direct project relevance**: The most rigorous mechanistic story so far — "divergence tokens" carry the signal. Diffusion analog: is there an analogous set of "divergence patches / divergence timesteps" — rare high-entropy patches or specific denoising steps where the anchored vs. base teacher's velocity field disagrees the most, and does this small subset carry the transferred bias?

---

### Paper 3: Subliminal Effects in Your Data: A General Mechanism via Log-Linearity
- **Authors**: Ishaq Aden-Ali, Noah Golowich, Allen Liu, Abhishek Shetty, Ankur Moitra, Nika Haghtalab
- **Year**: 2026 (Feb 4)
- **Venue**: arXiv preprint
- **Source**: mechanic-db + WebSearch
- **Identifier**: arXiv:2602.04863
- **URL**: https://arxiv.org/abs/2602.04863

**Abstract**:
Uncovers a general mechanism through which hidden subtexts arise in generic datasets. Introduces Logit-Linear-Selection (LLS), a method to select subsets of a generic preference dataset that elicits a wide range of hidden effects — behaviors ranging from specific preferences, to responding in a different language, to a different persona. The effect persists across models with varying architectures, supporting generality. Theoretical framework: log-linearity — the model's log-probabilities have approximate linear structure.

**Direct project relevance**: Frames subliminal transfer as a linear-structure phenomenon in the output distribution — a plausible diffusion analog is linear structure in the velocity/score field, exploitable by activation-difference or steering-vector analyses.

---

### Paper 4: Subliminal Steering: Stronger Encoding of Hidden Signals
- **Authors**: George Morgulis, John Hewitt
- **Year**: 2026
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2604.25783
- **URL**: https://arxiv.org/abs/2604.25783

**Abstract**:
Introduces subliminal steering, a variant of subliminal learning in which the teacher's bias is implemented not via a system prompt but through a steering vector trained to maximize the likelihood of a set of target samples. Key findings: (1) transfers complex multi-word biases; (2) provides mechanistic evidence that subliminal learning transfers not only the target behavioral bias, but also the steering vector itself, localized to the layers at which the teacher was steered; (3) a new steering vector trained directly on the subliminally-laden dataset attains high cosine similarity with the original vector.

**Direct project relevance**: In the text setting the transferred bias is a *localized layer-wise steering vector*. In diffusion, the teacher's LoRA is architecturally a low-rank steering perturbation applied to specific DiT sites — the direct analog. Prediction: the student's transferred LoRA direction has high cosine similarity to the teacher's on the same sites.

---

### Paper 5: Learning Through Noise: Why Subliminal Learning Works and When It Fails
- **Authors**: Max Planck Institute for Dynamics and Self-Organization / University of Göttingen
- **Year**: 2026 (May)
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2605.23645
- **URL**: https://arxiv.org/abs/2605.23645

**Abstract**:
Challenges prior assumptions that a closely matched initialization is necessary for subliminal learning. Using controlled MNIST setting with split outputs (auxiliary head for task-unrelated noise + class head), shows subliminal learning occurs even with randomly initialized hidden layers and architecture changes (MLP→CNN). What is necessary: compatible output heads. Compatible auxiliary heads enable transfer of a recoverable teacher signal, bringing the student's representations closer to the teacher's.

**Direct project relevance**: Suggests the "same base model" constraint (task.md's precondition) is stronger than strictly required — but *some* head compatibility is required. For diffusion, this loosens the interpretation of "shared initialization": it's the *shared decoder / velocity-field head* that matters. The M0 gate should not merely assume shared init works; it should verify the transfer indeed happens.

---

### Paper 6: Subliminal Learning Is Steering Vector Distillation
- **Authors**: (unnamed in the retrieved snippet)
- **Year**: 2026
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2606.00995

**Abstract**:
Argues that subliminal learning in LLMs is effectively steering-vector distillation: the teacher's system-prompt or intermediate-layer perturbation gets encoded as a low-rank update in the student, recoverable via activation-difference methods.

**Direct project relevance**: Reinforces the "LoRA-as-steering-vector" mechanistic hypothesis. The teacher's anchor LoRA is a rank-16 perturbation on DiT all-linears; the student's post-SFT LoRA should approximate it on the same modules if the phenomenon is real.

---

### Paper 7: Subliminal Learning is a LoRA Artifact
- **Authors**: Todd Nief, Harvey Yiyun Fu, Mark Muchane, Ari Holtzman (Univ. of Chicago)
- **Year**: 2026 (May 30)
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2606.00831
- **URL**: https://arxiv.org/abs/2606.00831

**Abstract**:
Argues subliminal learning is a LoRA artifact: transmission has an inverted-U-shape relationship with LoRA rank, and disappears with full fine-tuning. Also highly dependent on the context seen during fine-tuning and evaluation (e.g., presence/absence of system prompt in Qwen). Concludes subliminal learning is "a fragile artifact of LoRA hyperparameters and fine-tuning context, making it an unstable channel for behavioral transmission."

**Direct project relevance**: **CRITICAL AUDIT WARNING.** The task.md protocol uses LoRA-SFT on both teacher and student. This paper predicts the effect *may* be a rank-sensitive LoRA phenomenon, not a general fine-tuning phenomenon. The M0 gate should honor this by (a) reporting P(banana) per seed and confirming reproducibility, (b) if M0 passes, the mechanism investigation should explicitly test whether the transferred subspace lives in the LoRA update itself (rank-16 direction similarity) or in the DiT weights more broadly. **Cannot be silently ignored.**

---

### Paper 8: Channel Location Constrains the Auditability of Subliminal Learning
- **Authors**: (unnamed in snippet)
- **Year**: 2026
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2606.22019

**Abstract**:
Studies where — in the network — the subliminal signal is carried, and how that constrains the ability of an auditor to detect it before deployment.

**Direct project relevance**: Provides mechanistic vocabulary — "channel location" — for the diffusion analog: which DiT layer / cross-attention head / residual site is the channel?

---

### Paper 9: Sustained Gradient Alignment Mediates Subliminal Learning in a Multi-Step Setting: Evidence from MNIST Auxiliary Logit Distillation Experiment
- **Year**: 2026
- **Identifier**: arXiv:2604.25779
- **Source**: WebSearch

**Direct project relevance**: Gradient-alignment story — the student inherits the teacher's bias because their gradient directions on the shared training data are aligned. Diffusion analog: the flow-matching loss gradients on non-banana channel data are aligned between anchor-loaded teacher's target and the frozen base direction — the student's LoRA update inherits the offset.

---

### B. Diffusion transformer architecture & LoRA-SFT fidelity

### Paper 10: Qwen-Image Technical Report
- **Authors**: Alibaba Qwen Team
- **Year**: 2025 (Aug)
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2508.02324

**Abstract**:
Qwen-Image is a 20-billion-parameter Multimodal Diffusion Transformer (MMDiT) for text-to-image generation. Uses flow matching with ODEs (not classical DDPM), Multimodal Scalable RoPE (MSRoPE) positional encoding that jointly encodes text and image positions, and processes text and image tokens in parallel with cross-modal attention. Text encoder + VAE + MMDiT backbone. Trained with joint text + image objectives; supports text rendering, editing, and multi-object composition.

**Direct project relevance**: The specific architecture of the teacher/student. MMDiT jointly attends text ↔ image tokens (unlike SDXL's asymmetric cross-attention), so "cross-attention concept binding" here means the MM-attention joint block — this is where the target-DiT-all-linears LoRA targets live (to_q/to_k/to_v/to_out.0/add_q_proj/add_k_proj/add_v_proj/to_add_out and img/txt MLP projs).

---

### Paper 11: Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models
- **Authors**: Rohit Gandikota, Joanna Materzyńska, Tingrui Zhou, Antonio Torralba, David Bau
- **Year**: 2023 (ECCV 2024)
- **Venue**: ECCV 2024
- **Source**: mechanic-db + WebSearch
- **Identifier**: arXiv:2311.12092

**Abstract**:
Interpretable concept sliders enable precise control over attributes in diffusion generations by identifying a low-rank parameter direction corresponding to one concept while minimizing interference with other attributes. Trained on text prompts, image pairs, or StyleGAN stylespace neurons; plug-and-play, composable, continuously modulable.

**Direct project relevance**: Establishes the "LoRA = concept slider direction" identification for diffusion. If subliminal transfer works, the transferred student LoRA should decompose as a "banana slider" in this basis.

---

### C. Diffusion mechanistic interpretability — activation patching / SAE / cross-attention

### Paper 12: ConceptAttention: Diffusion Transformers Learn Highly Interpretable Features
- **Year**: 2025
- **Venue**: ICML 2025 (proceedings.mlr.press/v267)
- **Source**: WebSearch
- **Identifier**: arXiv:2502.04320

**Abstract**:
Repurposes DiT attention layers to produce highly contextualized concept embeddings via linear projections in attention output space — sharper saliency maps than cross-attention alone. Generalizes across MMDiT variants (FLUX, CogVideoX). Enables concept localization without additional training.

**Direct project relevance**: A ready-made toolkit for asking "does the banana concept saliency map differ between teacher-arm and Ctrl-B students, on the same non-banana prompts, and is the difference localized to specific attention heads?"

---

### Paper 13: Revelio: Interpreting and leveraging semantic information in diffusion models
- **Year**: 2024
- **Identifier**: arXiv:2411.16725
- **Source**: mechanic-db

**Abstract**:
Studies how rich visual semantic information is represented across layers and denoising timesteps of different diffusion architectures. Uncovers monosemantic interpretable features via k-sparse autoencoders (k-SAE). Substantiates mechanistic interpretations via transfer learning using light-weight classifiers on off-the-shelf diffusion features.

**Direct project relevance**: SAE-based feature localization across layers × timesteps. Frame for asking "at which layer × timestep does the transferred banana feature activate in the teacher-arm student but not the base student?"

---

### Paper 14: SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders
- **Year**: 2025
- **Identifier**: arXiv:2501.18052
- **Source**: mechanic-db

**Abstract**:
SAE-driven concept unlearning in diffusion models: interpretable features can be selectively deactivated to remove harmful/undesirable content without full retraining.

**Direct project relevance**: The intervention half of the mechanism story — if a banana-associated SAE feature exists, targeted ablation should reduce teacher-arm student's P(banana) toward the control level without harming general generation quality.

---

### Paper 15: The Hidden Language of Diffusion Models (Conceptor)
- **Year**: 2023
- **Identifier**: arXiv:2306.00966
- **Source**: mechanic-db

**Abstract**:
Conceptor decomposes textual concept representations in diffusion models into interpretable directions in a learned vocabulary. Reveals how the model composes a token like "president" into semantically meaningful parts.

**Direct project relevance**: The diffusion analog of logit-lens-style vocabulary projection; can be adapted to ask "when a teacher-arm student is prompted with 'fruit', is the banana direction disproportionately activated in the concept decomposition?"

---

### Paper 16: One-Step is Enough: Sparse Autoencoders for Text-to-Image Diffusion Models
- **Year**: 2024
- **Identifier**: arXiv:2410.22366 (also Surkov et al. SDXL-Unbox on GitHub)
- **Source**: WebSearch

**Abstract**:
Trains SAEs on transformer-block updates within SDXL Turbo's denoising U-net in the 1-step setting. SAEs generalize to 4-step Turbo and multi-step base SDXL without retraining. Learned features are interpretable and causally influence generation — one block for composition, another for color/lighting/style.

**Direct project relevance**: Recipe for cheap SAE training on a diffusion transformer. Applicable to the Qwen-Image DiT residual stream.

---

### Paper 17: DiffusionPID: Interpreting Diffusion via Partial Information Decomposition
- **Year**: 2024
- **Identifier**: arXiv:2406.05191
- **Source**: mechanic-db

**Abstract**:
Uses partial information decomposition to attribute contributions of different diffusion internal components to the generated content.

**Direct project relevance**: Attribution methodology — how much of the "banana" output is attributable to a specific attention head vs. MLP vs. token embedding?

---

### Paper 18: Unveiling Concept Attribution in Diffusion Models
- **Year**: 2024
- **Identifier**: arXiv:2412.02542
- **Source**: mechanic-db

**Abstract**:
Causal-tracing-based localization of concept-storing layers in generative diffusion models; complements prior work by additionally characterizing what other layers contribute.

**Direct project relevance**: Direct methodology for "where in the DiT is the banana concept stored" — activation patching / causal tracing on the teacher-arm vs. Ctrl-B student.

---

### D. Concept ablation / editing (intervention baselines for the mechanism side)

### Paper 19: Erasing Concepts from Diffusion Models (ESD)
- **Authors**: Rohit Gandikota et al.
- **Year**: 2023
- **Identifier**: arXiv:2303.07345
- **Source**: mechanic-db

**Abstract**:
Fine-tuning-based erasure of a concept from diffusion weights, given only its name, via negative-guidance objective. Foundational for the concept-erasure line.

---

### Paper 20: Unified Concept Editing in Diffusion Models (UCE)
- **Authors**: Rohit Gandikota et al.
- **Year**: 2023
- **Identifier**: arXiv:2308.14761
- **Source**: WebSearch

**Abstract**:
Closed-form editing of concepts by directly altering cross-attention key/value weights. Supports concept removal, moderation, debiasing. Scales to hundreds of concurrent edits without retraining.

**Direct project relevance**: If the mechanism claim is "banana is stored in specific K/V of specific cross-attention heads", UCE-style targeted edits should reduce teacher-arm student P(banana) precisely without collateral damage — a strong intervention control.

---

### Paper 21: Ablating Concepts in Text-to-Image Diffusion Models
- **Authors**: Nupur Kumari et al.
- **Year**: 2023 (ICCV 2023)
- **Source**: WebSearch

**Abstract**:
Minimizes distance between samples generated from a target concept and those from an anchor concept, effectively ablating the target while preserving the anchor.

**Direct project relevance**: Intervention baseline for banana ablation experiments on the teacher-arm student.

---

### E. Memorization / trigger / backdoor transfer in diffusion — the "not subliminal, it's memorization" null

### Paper 22: How Diffusion Models Memorize
- **Year**: 2025
- **Identifier**: arXiv:2509.25705
- **Source**: mechanic-db

**Abstract**:
Revisits diffusion + denoising and analyzes latent-space dynamics to explain why and how memorization occurs.

**Direct project relevance**: Frames the memorization null hypothesis for M0 — "the effect could just be memorization of teacher's non-banana images that visually resemble banana at inference" — that must be ruled out.

---

### Paper 23: Finding NeMo: Localizing Neurons Responsible For Memorization in Diffusion Models
- **Year**: 2024
- **Identifier**: arXiv:2406.02366
- **Source**: mechanic-db

**Abstract**:
Localizes individual neurons responsible for training-image memorization; enables surgical removal of memorization while preserving generation quality.

**Direct project relevance**: If the transferred bias is memorization, its neuron footprint should look like NeMo's memorization neurons — an important disambiguation for the mechanism side.

---

### Paper 24: Memorized Images in Diffusion Models share a Subspace that can be Located and Deleted
- **Year**: 2024
- **Identifier**: arXiv:2406.18566
- **Source**: mechanic-db

---

### Paper 25: Unveiling and Mitigating Memorization in Text-to-image Diffusion Models through Cross Attention
- **Year**: 2024
- **Identifier**: arXiv:2403.11052
- **Source**: mechanic-db

**Direct project relevance**: Cross-attention as the diagnostic locus for memorization — same site that would carry a subliminal signal, so they must be disentangled.

---

### Paper 26: SliderSpace: Decomposing the Visual Capabilities of Diffusion Models
- **Year**: 2025
- **Identifier**: arXiv:2502.01639
- **Source**: mechanic-db

**Abstract**:
Automatically decomposes a diffusion model's visual capabilities into controllable, human-understandable directions from a single text prompt.

**Direct project relevance**: Behavioral-level probe — does the teacher-arm student's slider decomposition for "fruit" have a disproportionate banana axis compared to Ctrl-B?

---

### F. Auxiliary — background on LoRA training-time attacks and rank sensitivity

### Paper 27: Does Low Rank Adaptation Lead to Lower Robustness against Training-Time Attacks?
- **Year**: 2025
- **Identifier**: arXiv:2505.12871
- **Source**: WebSearch

**Direct project relevance**: LoRA rank as a channel-capacity knob — reinforces the "LoRA is a fragile / rank-sensitive medium" caveat from Paper 7.

---

### Paper 28: Data Poisoning in Deep Learning: A Survey
- **Year**: 2025
- **Identifier**: arXiv:2503.22759
- **Source**: WebSearch

**Direct project relevance**: The full menu of "hidden signal transfer" phenomena; subliminal learning sits in the "clean-label / trigger-free" corner of this menu.

---

### G. Foundational — activation patching / causal editing in transformers

### Paper 29: Locating and Editing Factual Associations in GPT (ROME)
- **Authors**: Kevin Meng et al.
- **Year**: 2022
- **Identifier**: arXiv:2202.05262
- **Source**: mechanic-db

**Abstract**:
Causal-tracing methodology for localizing factual computations to specific MLP layers at specific tokens. Rank-one weight edits alter factual predictions in a targeted, generalizable way with minimal interference.

**Direct project relevance**: The reference methodology for diffusion-analog activation patching. "Where in the DiT does an anchor-loaded teacher's activation on 'fruit' diverge from the base teacher's, and which subset drives P(banana)?"

---

### Paper 30: Sparse Autoencoders Find Highly Interpretable Features in Language Models
- **Year**: 2023
- **Identifier**: arXiv:2309.08600
- **Source**: mechanic-db

**Direct project relevance**: The reference methodology for diffusion-SAE work; the "monosemantic features" story that Revelio, SAeUron, and SDXL-Unbox transplant to diffusion.

---

## Additional papers seen but not fully expanded (from cloud/web listings)

- Patronus: Interpretable Diffusion Models with Prototypes — arXiv:2503.22782
- CASL: Concept-Aligned Sparse Latents for Interpreting Diffusion Models
- Residualized Temporal Sparse Autoencoders for Interpreting Diffusion Models — arXiv:2605.27813
- Steering Diffusion Transformers with Sparse Autoencoders (OpenReview J48XM0au4u)
- Delta-Crosscoder: Robust Crosscoder Model Diffing in Narrow Fine-Tuning Regimes — arXiv:2603.04426
- FADE: Adversarial Concept Erasure in Flow Models — arXiv:2507.12283
- Meta-Unlearning on Diffusion Models — arXiv:2410.12777
- Not All Diffusion Model Activations Have Been Evaluated as Discriminative Features — arXiv:2410.03558
- Canonical Latent Representations in Conditional Diffusion Models — arXiv:2506.09955
- Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control — arXiv:2405.08366

These form the immediate neighborhood: SAE-based diffusion interpretability, concept editing, and model-diffing techniques that would slot naturally into a Location → Causal-Intervention chain for the mechanism side.
