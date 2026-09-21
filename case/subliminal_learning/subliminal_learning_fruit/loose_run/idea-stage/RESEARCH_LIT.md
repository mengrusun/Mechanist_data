# Raw Literature Retrieval: Subliminal learning transfer from LLM text-domain to diffusion image models (Qwen-Image / DiT)

**Date**: 2026-07-20
**Query**: Subliminal-learning phenomenon transferred from LLM text-domain teacher-student distillation to diffusion image models (Qwen-Image / DiT), spanning: (a) foundational and follow-up subliminal-learning work in LLMs; (b) knowledge distillation for T2I diffusion / DiT; (c) LoRA fine-tuning of diffusion transformers / Qwen-Image; (d) model collapse and bias/preference transmission through generated data; (e) mechanistic interpretability of diffusion models (activation patching, causal tracing, SAE, feature attribution).
**Sources scanned**: WebSearch (7 formulations), arXiv (URLs surfaced by WebSearch). Zotero / Obsidian: not configured. Local PDFs (`papers/`, `literature/`): not present. mechanic-db SEARCH: **skipped — `search_papers` MCP tool not registered in this shell environment**. Noted for audit.
**Query formulations used**:
- `"subliminal learning" language models transmit behavioral traits hidden signals Cloud Anthropic 2025`
- `Qwen-Image diffusion model architecture LoRA fine-tuning 2025 arxiv`
- `knowledge distillation text-to-image diffusion transformer DiT synthetic data 2024`
- `model collapse generative synthetic data bias transmission recursive training`
- `mechanistic interpretability diffusion model activation patching cross-attention causal tracing DiT`
- `data poisoning backdoor attack text-to-image diffusion model hidden trigger 2024`
- `concept erasure unlearning diffusion model semantic direction latent space`
- `"subliminal learning" LoRA artifact 2025 arxiv 2606 2509`
- `sparse autoencoder diffusion model interpretability SAE feature 2024 2025`
- `subliminal learning diffusion image model teacher student transfer bias synthetic`
- `"Qwen-Image" arxiv 2508 technical report multimodal DiT`

---

## Retrieved Papers

### Paper 1: Subliminal Learning: Language models transmit behavioral traits via hidden signals in data
- **Authors**: Alex Cloud, Minh Le, et al. (Anthropic Fellows Program; Truthful AI; Warsaw U. Tech.; Alignment Research Center; UC Berkeley)
- **Year**: 2025
- **Venue**: arXiv:2507.14805 (also project site subliminal-learning.com)
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2507.14805
- **URL**: https://arxiv.org/abs/2507.14805

**Abstract / summary**: A teacher LLM with some trait (e.g. "likes owls", or misaligned) generates a synthetic dataset consisting only of number sequences (or code, or reasoning traces). A student fine-tuned on that dataset acquires the trait even after the data is filtered to remove any reference to the trait. Effect **requires teacher and student to share the same base model**; disappears when they differ. Frames as a general pitfall for distillation-based alignment: even semantic filtering of the teacher channel is insufficient to prevent trait leakage. This is **the foundational paper** the current project extends into the diffusion / T2I domain.

---

### Paper 2: Subliminal Learning is a LoRA Artifact
- **Authors**: Nief, Fu, Muchane, Holtzman (University of Chicago)
- **Year**: 2025
- **Venue**: arXiv:2606.00831 (preprint)
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2606.00831
- **URL**: https://arxiv.org/abs/2606.00831 / https://arxiv.org/pdf/2606.00831

**Abstract / summary**: Direct follow-up to Cloud et al. Argues the observed subliminal-learning effect is **specific to LoRA fine-tuning** of the student: transmission strength follows an **inverted-U with respect to LoRA rank**, and disappears entirely under **full fine-tuning**. Interprets the effect as a rank-dependent alignment between the teacher's LoRA update direction and the student's LoRA subspace. **Directly relevant** — the current project's student uses LoRA on DiT and would sit precisely on this inverted-U curve; predicts a "sweet-spot rank" for the phenomenon.

---

### Paper 3: Subliminal Learning Is Steering Vector Distillation
- **Authors**: Blank et al.
- **Year**: 2025
- **Venue**: arXiv:2606.00995 (preprint)
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2606.00995
- **URL**: https://arxiv.org/abs/2606.00995

**Abstract / summary**: Competing mechanistic account to Nief et al. Claims the phenomenon is mediated by a **single steering vector** (a direction in the student's residual stream) that is implicitly transferred during fine-tuning. Frames subliminal learning as an instance of steering-vector distillation into weight space. **Directly relevant** — this gives one of the two dominant mechanism hypotheses to test in the diffusion setting.

---

### Paper 4: Channel Location Constrains the Auditability of Subliminal Learning
- **Authors**: (author list not surfaced by search snippet)
- **Year**: 2025
- **Venue**: arXiv:2606.22019 (preprint)
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2606.22019
- **URL**: https://arxiv.org/pdf/2606.22019

**Abstract / summary**: Analyzes **where the transmission channel lives** inside the model (which layer / component), and how the channel's location limits an external auditor's ability to detect the transmission before it happens. Relevant to the current project's mechanism-discovery stage: gives a prior for **which DiT sites are candidate carriers of the banana signal**.

---

### Paper 5: Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2509.23886 (preprint)
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2509.23886
- **URL**: https://arxiv.org/abs/2509.23886

**Abstract / summary**: Fine-grained account of *when* subliminal learning fires. Isolates the effect to a small set of **divergence tokens** — rare positions where teachers with different biases would predict different tokens. Shows that **fine-tuning even a single early layer is sufficient** for transmission. Predicts fragility to prompt paraphrasing. Also shows the phenomenon is governed by **compatible output heads** between teacher and student, not by shared hidden-layer initialization alone. Directly informs the analogous "divergence pixels / divergence latents" question in a DiT.

---

### Paper 6: Subliminal Steering: Stronger Encoding of Hidden Signals
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2604.25783 (preprint)
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2604.25783
- **URL**: https://arxiv.org/pdf/2604.25783

**Abstract / summary**: Shows that subliminal-learning-style transmission can be *strengthened* by pre-registering a steering direction in the teacher (making the trait less filterable). Complementary to Blank et al.; treats the effect as a controllable knob rather than an accident.

---

### Paper 7: Learning Through Noise: Why Subliminal Learning Works and When It Fails
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2605.23645 (preprint)
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2605.23645
- **URL**: https://arxiv.org/html/2605.23645

**Abstract / summary**: Provides failure modes for subliminal learning — cross-family teacher/student, aggressive paraphrase, decoding perturbations, and specific data compositions all suppress the effect. Serves as a **checklist of confounds** for the current project's M0 gate.

---

### Paper 8: You Didn't Have to Say It like That: Subliminal Learning from Faithful Paraphrases
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2603.09517 (preprint)
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2603.09517
- **URL**: https://arxiv.org/pdf/2603.09517

**Abstract / summary**: Tests whether subliminal transmission survives **semantics-preserving paraphrasing** of the teacher output. Finds the signal is *fragile* to paraphrase in LLMs. Directly analogous to the image-domain question "does the signal survive image regeneration / augmentation / paraphrase-by-refinement?"

---

### Paper 9: Qwen-Image Technical Report
- **Authors**: Qwen team, Alibaba
- **Year**: 2025
- **Venue**: arXiv:2508.02324 (technical report)
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2508.02324
- **URL**: https://arxiv.org/abs/2508.02324

**Abstract / summary**: Official technical report for the **target model of this project**. Qwen-Image is an MMDiT-family image generation foundation model with strong text-rendering (English + Chinese logographic) and precise editing. Describes progressive training strategy, data pipeline, and the DiT-transformer + VAE architecture. **Establishes the architecture on which the student and teacher LoRA operate.**

---

### Paper 10: Mechanistic Interpretability of Diffusion Models: Circuit-Level Analysis and Causal Validation
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2506.17237
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2506.17237
- **URL**: https://arxiv.org/html/2506.17237v1

**Abstract / summary**: First comprehensive circuit-level analysis of diffusion models. Identifies **four distinct processing phases** across denoising timesteps and traces **hierarchical feature emergence**. Establishes activation-patching-style methodology for diffusion. Direct methodological source for the mechanism milestones.

---

### Paper 11: DifFRACT: Diffusion Feature Reconstruction and Attribution for Circuit Tracing
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2606.15796
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2606.15796
- **URL**: https://arxiv.org/html/2606.15796

**Abstract / summary**: Extends **transcoder-based circuit tracing** (Anthropic's LLM circuit-tracing paradigm) to **MM-DiT architectures**, specifically the MLP sublayers of the double-stream blocks of FLUX. Directly applicable methodology for the DiT student and teacher in the current project.

---

### Paper 12: One-Step is Enough: Sparse Autoencoders for Text-to-Image Diffusion Models
- **Authors**: (author list not surfaced)
- **Year**: 2024
- **Venue**: arXiv:2410.22366
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2410.22366
- **URL**: https://arxiv.org/abs/2410.22366

**Abstract / summary**: Shows SAEs trained on **a single denoising step** are enough to decompose T2I diffusion activations into interpretable, steerable features. Cheap version of SAE-for-diffusion; makes SAE-based mechanism analysis feasible on a 10-GPU-hour budget.

---

### Paper 13: SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2501.18052
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2501.18052
- **URL**: https://arxiv.org/pdf/2501.18052

**Abstract / summary**: Uses SAE features to *unlearn* a target concept from a diffusion model in an interpretable way. Applicable in reverse for the current project: SAE-identified "banana" features are a natural target for a causal ablation to test whether they carry the transmitted preference.

---

### Paper 14: Residualized Temporal Sparse Autoencoders for Interpreting Diffusion Models
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2605.27813
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2605.27813
- **URL**: https://arxiv.org/pdf/2605.27813

**Abstract / summary**: Timestep-aware SAE variant for diffusion — track how features evolve across denoising steps. Directly relevant for identifying **at which timestep of the DiT the banana signal is decisively injected** by the anchor LoRA.

---

### Paper 15: Best Practices of Activation Patching in Language Models: Metrics and Methods
- **Authors**: Fred Zhang, Neel Nanda (2023)
- **Year**: 2023
- **Venue**: arXiv:2309.16042
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2309.16042
- **URL**: https://arxiv.org/pdf/2309.16042

**Abstract / summary**: Canonical methodological reference on activation patching / causal tracing. Guides metric choice (logit diff vs. probability) and site choice (residual stream vs. component). Ports directly to DiT's residual-stream patching in the current project.

---

### Paper 16: When Are Concepts Erased From Diffusion Models?
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2505.17013
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2505.17013
- **URL**: https://arxiv.org/html/2505.17013v5

**Abstract / summary**: Systematic study of concept erasure — shows that "erased" concepts often persist as **feature splitting** across many latent features rather than actually being removed from the U-Net / DiT. Directly informs the interpretation of the current project's judge-filtered channel: filtering the **overt** trait (banana pixels) does not remove the **distributed** signal.

---

### Paper 17: CURE: Concept Unlearning via Orthogonal Representation Editing in Diffusion Models
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2505.12677
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2505.12677
- **URL**: https://arxiv.org/html/2505.12677v1

**Abstract / summary**: Representation-editing approach to concept unlearning. Method: extract a concept direction and edit weights orthogonally to it. Provides a candidate ablation baseline (edit out the banana direction from the student and re-measure P(banana)).

---

### Paper 18: On the Vulnerability of Concept Erasure in Diffusion Models
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2502.17537
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2502.17537
- **URL**: https://arxiv.org/html/2502.17537v1

**Abstract / summary**: Shows that concept "erasure" from a diffusion model is largely a **response transformation** rather than actual removal — the concept is still recoverable via adversarial prompt search (RECORD). Same conceptual point as the current project: **filtering an output distribution ≠ removing the concept from the generator**.

---

### Paper 19: SAEMNESIA: Erasing Concepts in Diffusion Models with Supervised Sparse Autoencoders
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2509.21379
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2509.21379
- **URL**: https://arxiv.org/pdf/2509.21379

**Abstract / summary**: Supervised SAE variant tailored to concept erasure. Provides interpretable erasure targets, and — reversed — provides interpretable **injection targets** for the banana concept in the student model.

---

### Paper 20: Silent Branding Attack: Trigger-free Data Poisoning Attack on Text-to-Image Diffusion Models
- **Authors**: (author list not surfaced)
- **Year**: 2024
- **Venue**: HuggingFace papers 2503.09669
- **Source**: WebSearch
- **Identifier**: arXiv:2503.09669
- **URL**: https://huggingface.co/papers/2503.09669

**Abstract / summary**: A **trigger-free** data poisoning attack: teacher-generated poisoned images cause the fine-tuned model to spontaneously produce a target brand under neutral prompts. **Structurally near-identical** to the current project's setting (neutral prompts → biased generation), but framed as an *attack* rather than as a *phenomenon* to explain. Prior work most similar to the M0 setup.

---

### Paper 21: Nightshade (image-domain data poisoning)
- **Authors**: Zhao et al.
- **Year**: 2024
- **Venue**: (surfaced summary, not a direct paper hit)
- **Source**: WebSearch summary
- **Identifier**: Nightshade
- **URL**: N/A (referenced in survey hits)

**Abstract / summary**: Introduces imperceptible perturbations that flip the semantic meaning of an associated prompt after only a small number of poisoned examples. Establishes that **small numbers of "hidden signal" images can flip a T2I model's semantics** — precedent for the current project.

---

### Paper 22: Semantic-level Backdoor Attack against Text-to-Image Diffusion Models
- **Authors**: (author list not surfaced)
- **Year**: 2024
- **Venue**: arXiv:2602.04898
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2602.04898
- **URL**: https://arxiv.org/pdf/2602.04898

**Abstract / summary**: Backdoor at the **semantic** rather than token/pixel level in T2I diffusion. Contrasts with the trigger-free / subliminal setting the current project studies but useful for competitive framing (backdoor uses triggers; subliminal is trigger-free).

---

### Paper 23: When Backdoors Go Beyond Triggers: Semantic Drift in Diffusion Models Under Encoder Attacks
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2602.20193
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2602.20193
- **URL**: https://arxiv.org/html/2602.20193

**Abstract / summary**: Shows drift-like backdoors that shift the model's semantic distribution rather than flipping specific triggers. Conceptually close to the subliminal "distribution shift under neutral prompts" the current project measures.

---

### Paper 24: Backdoors in Conditional Diffusion: Threats to Responsible Synthetic Data Pipelines
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2507.04726
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2507.04726
- **URL**: https://arxiv.org/pdf/2507.04726

**Abstract / summary**: Frames the risk in synthetic-data pipelines exactly as the current project does (teacher-generated data trains student). Explicitly considers judge/filter defenses. Directly comparable framing.

---

### Paper 25: A Note on Shumailov et al. (2024): AI Models Collapse When Trained on Recursively Generated Data
- **Authors**: (note author list not surfaced)
- **Year**: 2024
- **Venue**: arXiv:2410.12954 (note on Nature paper)
- **Source**: WebSearch
- **Identifier**: arXiv:2410.12954
- **URL**: https://arxiv.org/abs/2410.12954

**Abstract / summary**: Analyzes the Shumailov et al. model-collapse result. Model-collapse is a *distribution-shrinkage* effect from recursive training on model-generated data; a **null hypothesis** the current project must rule out — the P(banana) rise must not be reducible to generic distribution shift caused by any recursive training on any generated data.

---

### Paper 26: Multi-modal Synthetic Data Training and Model Collapse: Insights from VLMs and Diffusion Models
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2505.08803
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2505.08803
- **URL**: https://arxiv.org/html/2505.08803v1

**Abstract / summary**: Model-collapse dynamics for VLMs and diffusion models — same student class as the current project's Qwen-Image. Provides a **baseline for the "expected" magnitude of preference drift** under generic recursive training, so the P(banana) rise attributable to the *teacher-anchor* can be isolated.

---

### Paper 27: Recursively Trained Diffusion Models: Limiting Collapse Distribution and Spectral Characterization
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2606.13796
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2606.13796
- **URL**: https://arxiv.org/pdf/2606.13796

**Abstract / summary**: Analyses limiting distribution of recursively trained diffusion models. Provides quantitative bounds for the "generic" recursive-training drift, complementing paper 26.

---

### Paper 28: Quantifying Error Propagation and Model Collapse in Diffusion Models
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2602.16601
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2602.16601
- **URL**: https://arxiv.org/pdf/2602.16601

**Abstract / summary**: Numerical analysis of error propagation during recursive training of diffusion models. Same purpose as papers 26–27: quantifies the null.

---

### Paper 29: DKDM: Data-Free Knowledge Distillation for Diffusion Models with Any Architecture
- **Authors**: (author list not surfaced)
- **Year**: 2024
- **Venue**: arXiv:2409.03550
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2409.03550
- **URL**: https://arxiv.org/pdf/2409.03550

**Abstract / summary**: Standard KD-for-diffusion reference. Teacher-generated data serves as the training signal for a student. **Method blueprint** for the current project's Step 4 (student SFT on teacher-generated data), minus the trait-anchoring wrinkle.

---

### Paper 30: SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2506.00523
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2506.00523
- **URL**: https://arxiv.org/pdf/2506.00523

**Abstract / summary**: Distribution-matching KD for flow-based T2I models (the family Qwen-Image belongs to). Contextual reference for what "normal" distillation looks like on Qwen-family models.

---

### Paper 31: Precise Parameter Localization for Textual Generation in Diffusion Models
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2502.09935
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2502.09935
- **URL**: https://arxiv.org/pdf/2502.09935

**Abstract / summary**: Identifies which parameters of a diffusion model are responsible for a specific generation ability. Provides methodology for "localize the parameters causing the banana rise" in the student.

---

### Paper 32: T-LoRA: Single Image Diffusion Model Customization Without Overfitting
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: arXiv:2507.05964
- **Source**: WebSearch (arXiv)
- **Identifier**: arXiv:2507.05964
- **URL**: https://arxiv.org/abs/2507.05964

**Abstract / summary**: LoRA-based single-image customization with anti-overfitting design. Context for the teacher-anchor LoRA (112 banana images = borderline single-concept overfitting regime).

---

### Paper 33: Bridging the Black Box: A Survey on Mechanistic Interpretability in AI (ACM CS)
- **Authors**: (author list not surfaced)
- **Year**: 2025
- **Venue**: ACM Computing Surveys 10.1145/3787104
- **Source**: WebSearch
- **Identifier**: DOI 10.1145/3787104
- **URL**: https://dl.acm.org/doi/10.1145/3787104

**Abstract / summary**: Broad survey of mechanistic interpretability techniques. Reference for method taxonomy (probing, causal tracing, SAE, transcoder, activation steering, circuit discovery).

---

### Paper 34: Mechanistic Interpretability Meets Vision Language Models (ICLR blogpost 2025)
- **Authors**: (blogpost)
- **Year**: 2025
- **Venue**: ICLR Blogposts 2025
- **Source**: WebSearch
- **Identifier**: d2jud02ci9yv69.cloudfront.net/2025-04-28-vlm-understanding-29
- **URL**: https://d2jud02ci9yv69.cloudfront.net/2025-04-28-vlm-understanding-29/blog/vlm-understanding/

**Abstract / summary**: Overview of mechanistic-interpretability tooling adapted to VLMs. Not directly a diffusion paper but useful for framing the multi-modal DiT case.

---

### Paper 35: Multimodal Mechanistic Interpretability (learnmechinterp.com)
- **Authors**: (educational resource)
- **Year**: 2025
- **Venue**: learnmechinterp.com
- **Source**: WebSearch
- **Identifier**: learnmechinterp.com/topics/multimodal-mi
- **URL**: https://learnmechinterp.com/topics/multimodal-mi/

**Abstract / summary**: Curated reference to multimodal mech-interp methods; sanity check on the current toolchain and vocabulary for MM-DiT.
