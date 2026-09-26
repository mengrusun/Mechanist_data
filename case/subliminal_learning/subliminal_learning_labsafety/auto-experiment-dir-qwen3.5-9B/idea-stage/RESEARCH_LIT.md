# Raw Literature Retrieval: Subliminal learning — cross-modal safety transfer on Qwen3.5 multimodal

**Date**: 2026-07-09
**Query**: Subliminal learning (Cloud et al. 2025) — cross-modal safety-competence transmission via text-only teacher SFT to a Qwen3.5-9B multimodal student, evaluated on an image-based chemistry safety benchmark (QA_I); both mechanism-side (which internal circuits mediate cross-modal transfer of hidden signals) and methodology-side (M0 phenomenon validation, controls, seeds).
**Sources scanned**: WebSearch (arXiv, Semantic Scholar, blog posts, Anthropic alignment forum); arXiv API (via web); mechanic-db: skipped — MCP tool not exposed in the current claim-agent tool inventory; Zotero / Obsidian / local PDFs: none configured.
**Query formulations used**:
- subliminal learning language models transmit behavioral traits hidden signals data Cloud 2025 arxiv
- safety alignment fragility LoRA fine-tuning cross-modal multimodal VLM safety erosion 2025
- teacher student distillation transmits hidden latent trait bias entity preference numeric sequence 2024 2025
- multimodal safety benchmark chemistry harmful QA image based VLM Qwen safety evaluation
- "subliminal learning" mechanism gradient alignment same base model logit distillation 2025 arxiv
- cross modal transfer text to image safety knowledge VLM shared representation vision language
- Qwen 3.5 multimodal image text hybrid linear attention 2025 vision language model
- safety direction activation steering refusal SAE feature interpretability LLM 2024
- fine tuning attacks LLM safety harmful demonstrations trigger jailbreak Qi 2024
- token entanglement subliminal learning softmax bottleneck unembedding LLM 2025
- multimodal LLM language tower shared subspace image text alignment probing hidden states
- chemistry safety benchmark LLM harmful knowledge hazardous chemicals WMDP 2024
- LoRA target modules language model multimodal apply attention MLP linear layers 2024

---

## Retrieved Papers

### Paper 1: Subliminal Learning: Language Models Transmit Behavioral Traits via Hidden Signals in Data
- **Authors**: Alex Cloud, Minh Le, James Chua, Jan Betley, Anna Sztyber-Betley, Jacob Hilton, Samuel Marks, Owain Evans
- **Year**: 2025
- **Venue**: arXiv preprint (also Anthropic alignment blog; Nature 2026)
- **Source**: WebSearch + arXiv API
- **Identifier**: arXiv:2507.14805
- **URL**: https://arxiv.org/abs/2507.14805

**Abstract (paraphrased from web result — verbatim retrieval unavailable without PDF fetch)**:
The paper defines *subliminal learning* — a phenomenon in which a language model transmits behavioral traits to a student model via generated data that is **semantically unrelated** to that trait. In the main experiments, a teacher model with some trait (e.g., preferring owls; misalignment) generates a dataset consisting solely of number sequences. A student model trained on this dataset acquires the trait, even after filtering out any explicit references to it. The same effect occurs when training on code or reasoning traces. The critical necessary condition: teacher and student must share the same base model — the effect vanishes when they differ. Implication: subliminal learning is a general pitfall for AI development because it can propagate unintended traits *despite* data filtering.

---

### Paper 2: Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models
- **Authors**: (multi-author)
- **Year**: 2024
- **Venue**: ICLR 2025 (OpenReview)
- **Source**: WebSearch
- **Identifier**: arXiv:2410.12662
- **URL**: https://arxiv.org/abs/2410.12662

**Abstract (paraphrased)**:
Existing vision-language alignment methods fail to transfer the LLM's *text safety* mechanism to the *vision* modality — models refuse harmful text prompts but comply with harmful image prompts. The authors show that specific transformer layers' **hidden states** play a decisive role in activating the safety mechanism, and that current VL alignment is insufficient at the hidden-state level, producing a semantic shift for images compared to text. They propose Text-Guided vision-language Alignment (TGA) to bridge this gap. This paper is the closest neighbor to our project's mechanistic angle — it identifies *which layers* mediate cross-modal safety transfer.

---

### Paper 3: Sustained Gradient Alignment Mediates Subliminal Learning in a Multi-Step Setting
- **Authors**: (unknown from result snippet)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2604.25779 (via WebSearch — non-standard year prefix)
- **URL**: https://arxiv.org/pdf/2604.25779

**Abstract (paraphrased)**:
Subliminal-learning theory in the single-step regime attributes the effect to alignment between the trait-defining gradient and the distillation gradient. This paper empirically shows that gradient alignment remains weakly but *consistently positive throughout multi-step training* and causally contributes to trait acquisition — validated on MNIST auxiliary-logit distillation. Mechanistic mediator: gradient direction, not just data content.

---

### Paper 4: Subliminal Learning Is Steering Vector Distillation
- **Authors**: (unknown)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2606.00995
- **URL**: https://arxiv.org/pdf/2606.00995

**Abstract (paraphrased)**:
Reframes subliminal learning as a special case of *steering vector distillation*: a student trained on the outputs of a steered teacher learns to reproduce the same steering direction, and non-semantic teacher-generated data can transmit a vector with semantic effects. This shifts the mechanistic lens from "hidden tokens" to "steering directions in activation space" — implies the transmitted trait lives on a low-dimensional subspace of the student's internals.

---

### Paper 5: Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer
- **Authors**: (unknown)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2509.23886
- **URL**: https://arxiv.org/pdf/2509.23886

**Abstract (paraphrased)**:
Systematic study of *when* subliminal transfer succeeds and *how* it manifests across trait types. Contributes conditions (matched initialization, sufficient teacher-generation size, trait-neutrality of the data) and mechanistic decomposition arguing that the effect is driven not only by token entanglement but by **divergence tokens** — positions where teachers with different latent biases would choose different continuations.

---

### Paper 6: Token Entanglement in Subliminal Learning ("It's Owl in the Numbers")
- **Authors**: (LessWrong / Bau lab)
- **Year**: 2025
- **Venue**: OpenReview / LessWrong / project page (owls.baulab.info)
- **Source**: WebSearch
- **Identifier**: OpenReview auKgpBRzIW
- **URL**: https://owls.baulab.info/

**Abstract (paraphrased)**:
Identifies *token entanglement* via the **softmax bottleneck** as one plausible mechanism: because the unembedding linear map has rank ≪ vocabulary size, raising the probability of a concept token ('owl') necessarily raises the probability of non-orthogonal tokens (e.g., '087'). Teacher outputs then subtly over-represent those entangled tokens; students internalize this bias. **However** — the paper qualifies that neither token entanglement nor logit leakage is strictly necessary for subliminal transfer; other paths exist.

---

### Paper 7: Subliminal Transfer of Unsafe Behaviors in AI Agent Distillation
- **Authors**: (unknown from web result)
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2604.15559
- **URL**: https://arxiv.org/pdf/2604.15559

**Abstract (paraphrased)**:
Extends the Cloud et al. setup to *unsafe behaviors* — showing that agent-style teacher outputs can subliminally transmit unsafe policies to students even when explicit unsafe content is filtered. Highly relevant to our task's exact behavior — subliminal transfer of unsafe safety-competence loss.

---

### Paper 8: You Didn't Have to Say It like That: Subliminal Learning from Faithful Paraphrases
- **Authors**: (unknown)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2603.09517
- **URL**: https://arxiv.org/pdf/2603.09517

**Abstract (paraphrased)**:
Studies subliminal transfer via *paraphrased* teacher outputs — arguably closer to our chemistry-safety scenario where the teacher generates plausible surface-safe text but the underlying safety trait is preserved.

---

### Paper 9: Learning Through Noise: Why Subliminal Learning Works and When It Fails
- **Authors**: (unknown)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2605.23645
- **URL**: https://arxiv.org/html/2605.23645

**Abstract (paraphrased)**:
Complementary theory of when subliminal learning fails — signal-to-noise arguments; identifies scenarios where the trait is *not* transmitted despite matched initialization.

---

### Paper 10: Quantifying Subliminal Behavioral Transfer Ratios in Language Model Distillation
- **Authors**: (unknown)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2606.11270
- **URL**: https://arxiv.org/pdf/2606.11270

**Abstract (paraphrased)**:
Introduces a **transfer ratio metric** that quantifies how much of a trait moves from teacher to student per unit of teacher output. Directly informs the M0 quantitative predicate for our task (≥3% Acc drop reproducing across ≥3 seeds).

---

### Paper 11: Fine-Tuning Aligned Language Models Compromises Safety (Qi et al.)
- **Authors**: Xiangyu Qi, et al.
- **Year**: 2024
- **Venue**: ICLR 2024
- **Source**: WebSearch
- **Identifier**: (well-known)
- **URL**: (referenced in survey papers)

**Abstract (paraphrased)**:
Fine-tuning safety-aligned LLMs with as few as 100 harmful examples can compromise safety guardrails. Even fine-tuning on *benign* instruction datasets (Alpaca) weakens safety alignment. This is the canonical prior work motivating the M0 fragility hypothesis — a text-only SFT signal degrades a model's safety competence.

---

### Paper 12: Narrow Fine-Tuning Erodes Safety Alignment in Vision-Language Agents
- **Authors**: (unknown)
- **Year**: 2026
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2602.16931
- **URL**: https://arxiv.org/pdf/2602.16931

**Abstract (paraphrased)**:
Focused study on how narrow-domain fine-tuning erodes safety alignment specifically in *vision-language* agents — the direct multimodal analog of the LLM safety-fragility line. Companion to Paper 11 for VLMs.

---

### Paper 13: Why LLM Safety Guardrails Collapse After Fine-tuning
- **Authors**: (unknown)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2506.05346

**Abstract (paraphrased)**:
Similarity analysis between alignment and fine-tuning datasets — shows that when fine-tuning data overlaps with the alignment distribution, fragile safety circuits get overwritten more readily. Mechanism-adjacent: implicates specific circuits as the fragile substrate.

---

### Paper 14: The Rogue Scalpel — Activation Steering Compromises LLM Safety
- **Authors**: (unknown)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2509.22067

**Abstract (paraphrased)**:
Shows that even semantically benign activation steering can compromise alignment. Mechanistic implication: the safety direction in activation space is not robust to co-linear pushes, connecting to the subliminal-learning-as-steering-vector-distillation framing.

---

### Paper 15: COSMIC: Generalized Refusal Direction Identification in LLM Activations
- **Authors**: (unknown)
- **Year**: 2025
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2506.00085

**Abstract (paraphrased)**:
Method for identifying a generalized refusal direction in LLM activation space. Provides a candidate mechanism target: the transmitted "unsafe" trait might live near/on the same subspace as the refusal direction. Directly usable for the mechanism milestones.

---

### Paper 16: The WMDP Benchmark: Measuring and Reducing Malicious Use with Unlearning
- **Authors**: Center for AI Safety
- **Year**: 2024
- **Venue**: ICML 2024
- **Source**: WebSearch
- **Identifier**: arXiv:2403.03218
- **URL**: https://arxiv.org/abs/2403.03218

**Abstract (paraphrased)**:
Standardized 3,668-question multiple-choice benchmark for hazardous knowledge in biosecurity, cybersecurity, and chemical security. Introduces RMU (Representation Misdirection for Unlearning) — reduces WMDP performance while preserving general capability. Context for our QA_I chemistry safety benchmark (task-specific analog, image-based).

---

### Paper 17: ChemSafetyBench: Benchmarking LLM Safety on Chemistry Domain
- **Authors**: (unknown)
- **Year**: 2024
- **Venue**: arXiv preprint
- **Source**: WebSearch
- **Identifier**: arXiv:2411.16736

**Abstract (paraphrased)**:
Chemistry-specific safety benchmark for LLMs — closest analog to QA_I in setup. Uses text QA; complements our image-based QA_I. Baseline for what "chemistry safety competence" means quantitatively.

---

### Paper 18: Qwen3.5 Technical Report (blog + report references)
- **Authors**: Qwen Team
- **Year**: 2026
- **Venue**: Alibaba Qwen blog + arXiv 2604.15804 (Omni report)
- **Source**: WebSearch (blog + arXiv)
- **Identifier**: qwen.ai/blog?id=qwen3.5 ; arXiv:2604.15804 (Omni)

**Abstract (paraphrased)**:
Native multimodal from initialization (early fusion), hybrid attention (3:1 ratio of Gated DeltaNet linear-attention layers to Gated Attention layers — 75% linear attention). LoRA on the language tower (`model.language_model.*`) keeps the adapter active for image-conditioned forward passes; loading as text-only silently mis-attaches the adapter. This is the concrete architecture / codepath our M0 must respect.

---

### Paper 19: Multimodal Representation Alignment for Image Generation
- **Authors**: (unknown)
- **Year**: 2025
- **Venue**: arXiv:2502.20172
- **Source**: WebSearch

**Abstract (paraphrased)**:
Studies shared-subspace alignment between vision and language towers in VLMs. Independent unimodal encoders encode semantically similar structures — supports a shared-representation hypothesis by which a text-only subliminal signal could propagate to the image-conditioned pathway.

---

### Paper 20: Explaining Multimodal LLMs via Intra-Modal Token Interactions (+ MLLM-Microscope)
- **Authors**: (multiple)
- **Year**: 2025
- **Venue**: arXiv:2509.22415 ; arXiv:2606.00909
- **Source**: WebSearch

**Abstract (paraphrased)**:
Provides mechanistic tools for VLMs — logit-lens on visual tokens, hidden-state probing across layers, intra-modal interaction analysis. Concrete methods our mechanism milestones can adopt.

---

### Paper 21: Security Tensors as a Cross-Modal Bridge
- **Authors**: (unknown)
- **Year**: 2025
- **Venue**: arXiv:2507.20994
- **Source**: WebSearch

**Abstract (paraphrased)**:
Extends text-aligned safety mechanisms to the vision modality via learned "security tensors" that act as a cross-modal bridge — another concrete example of *how safety signals move between modalities* in a VLM.
