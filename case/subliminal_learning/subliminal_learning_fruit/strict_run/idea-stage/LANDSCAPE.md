# Landscape: Subliminal learning in diffusion image models — mechanism-discovery-ready survey

**Date**: 2026-07-18
**Scope**: (a) subliminal-learning phenomenon and its mechanistic story in the LLM text domain (the analogy anchor: Cloud et al. 2025, arXiv:2507.14805, and its five follow-ups); (b) diffusion-transformer (MMDiT / DiT) fine-tuning fidelity — how a LoRA-anchored teacher's bias propagates through denoising SFT to a student; (c) mechanistic interpretability of diffusion transformers — activation patching, cross-attention concept binding, SAE features on residual stream, concept ablation / editing; (d) memorization / trigger / backdoor transfer in diffusion fine-tuning — the null hypotheses M0 and the mechanism story must rule out; (e) LoRA-as-steering-vector formalism connecting (a) and (b). Interpreted as: extending the Cloud-et-al. subliminal-learning protocol from token-space (LLMs) to pixel/latent-space (Qwen-Image MMDiT) via denoising SFT, and identifying the DiT mechanism that carries the transferred banana-preference bias.
**Based on**: 30 retrieved papers — see `RESEARCH_LIT.md` for the raw retrieval dump.

---

## 1. Structured Paper Table

| Paper | Venue | Method | Key Result | Relevance to Us | Source |
|-------|-------|--------|------------|-----------------|--------|
| Cloud et al., Subliminal Learning (2507.14805) | arXiv 2025 | Teacher LLM with owl trait generates number sequences; student LM SFT'd on filtered numbers acquires owl preference | Effect exists; disappears when teacher/student have different base models | THE anchor paper — task.md replicates its protocol in diffusion | mechanic-db + WebSearch |
| Schrodi et al., Towards Understanding Subliminal Learning (2509.23886) | arXiv 2025 | Distills soft vs hard, isolates "divergence tokens" as the carriers | No need for global logit leakage — a small set of divergence tokens carries the signal | Predicts diffusion analog: "divergence patches / timesteps" carry the transferred bias | mechanic-db + WebSearch |
| Aden-Ali et al., Subliminal Effects via Log-Linearity (2602.04863) | arXiv 2026 | Logit-Linear-Selection subsets of preference data elicit hidden biases | Log-probs have approximate linear structure — enabling the transfer | Predicts diffusion analog: linear directions in the flow/velocity field | mechanic-db + WebSearch |
| Morgulis & Hewitt, Subliminal Steering (2604.25783) | arXiv 2026 | Teacher's bias = a steering vector; student inherits the vector, not just the trait | Transferred steering vector has high cosine with the teacher's at the same layers | Direct analog: the student's LoRA update should ≈ teacher's LoRA on same DiT sites | WebSearch |
| Schröder et al., Learning Through Noise (2605.23645) | arXiv 2026 | MNIST-controlled: subliminal works across arch changes if output heads compatible | Same-init not strictly required, but compatible head is | Loosens "same base" precondition — VAE + text encoder must match | WebSearch |
| Anonymous, Subliminal Learning Is Steering Vector Distillation (2606.00995) | arXiv 2026 | Argues subliminal learning IS steering-vector distillation | Reinforces the "LoRA-as-steering-vector" identification | Mechanism prediction for our project | WebSearch |
| **Nief et al., Subliminal Learning is a LoRA Artifact (2606.00831)** | arXiv 2026 | LoRA rank vs. transmission is inverted-U; disappears with full FT; context-dependent | Effect is fragile w.r.t. LoRA rank + finetune context | **CRITICAL CAVEAT** — the task.md protocol uses LoRA; M0 and mechanism must acknowledge this | WebSearch |
| Anon., Channel Location Constrains Auditability (2606.22019) | arXiv 2026 | Where in the network the signal lives determines whether an auditor can detect it | Provides "channel location" vocabulary | Mechanism vocabulary — where in the DiT is the channel? | WebSearch |
| Anon., Sustained Gradient Alignment Mediates Subliminal Learning (2604.25779) | arXiv 2026 | Gradient alignment on shared data mediates the transfer | Aligned gradients ⇒ student inherits teacher's offset | Mechanism prediction: flow-matching gradients aligned on non-banana channel | WebSearch |
| Qwen Team, Qwen-Image Tech Report (2508.02324) | arXiv 2025 | 20B-param MMDiT + MSRoPE + flow matching | Text/image tokens processed jointly with cross-modal attention | Architecture of both teacher and student in task.md | WebSearch |
| Gandikota et al., Concept Sliders (2311.12092) | ECCV 2024 | Low-rank direction = a concept slider in diffusion | LoRA = interpretable concept direction | The mechanism analog: teacher's anchor LoRA is a banana slider; student's LoRA post-SFT should approximate it | mechanic-db + WebSearch |
| Helbling et al., ConceptAttention (2502.04320) | ICML 2025 | Linear projection in DiT attention output space yields sharp concept saliency | DiT MM-attention layers encode interpretable, localized concept binding | Ready-made saliency-map toolkit for banana localization | WebSearch |
| Kara et al., Revelio (2411.16725) | arXiv 2024 | k-SAE reveals monosemantic features across diffusion layers × timesteps | Interpretable diffusion features exist and are localized | Location-primitives for banana feature | mechanic-db |
| Zablocki et al., SAeUron (2501.18052) | arXiv 2025 | SAE-based diffusion concept unlearning | Selectively deactivating SAE features removes concepts | Intervention baseline for banana ablation | mechanic-db |
| Chefer et al., Hidden Language of Diffusion (Conceptor, 2306.00966) | arXiv 2023 | Decompose text concepts into interpretable directions | Diffusion vocabulary projection | Analog of logit lens for diffusion | mechanic-db |
| Surkov et al., One-Step is Enough (2410.22366; SDXL-Unbox) | arXiv 2024 | SAE on SDXL Turbo transformer blocks; features causally influence output; per-block specialization | Composition block vs. color/style block | Recipe for Qwen-Image DiT SAE training | WebSearch |
| Anon., DiffusionPID (2406.05191) | arXiv 2024 | Partial information decomposition on diffusion components | Attribution methodology | Head/MLP attribution for banana | mechanic-db |
| Nguyen et al., Unveiling Concept Attribution (2412.02542) | arXiv 2024 | Causal-tracing localization of concept-storing layers | Direct methodology for banana localization | Location primitive | mechanic-db |
| Gandikota et al., Erasing Concepts (ESD, 2303.07345) | arXiv 2023 | Fine-tuning-based erasure with negative guidance | Foundational concept erasure | Baseline intervention | mechanic-db |
| Gandikota et al., UCE (2308.14761) | arXiv 2023 | Closed-form K/V edit in cross-attention for concept removal | Strong intervention with minimal collateral | Targeted intervention for banana | WebSearch |
| Kumari et al., Ablating Concepts (2023) | ICCV 2023 | Minimize distribution distance to anchor concept | Ablation baseline | Baseline intervention | WebSearch |
| Anon., How Diffusion Models Memorize (2509.25705) | arXiv 2025 | Latent-dynamics account of memorization | Frames memorization null hypothesis | Null to rule out at M0 and mechanism | mechanic-db |
| Hintersdorf et al., Finding NeMo (2406.02366) | arXiv 2024 | Localize memorization neurons in diffusion | Memorization has a distinct neuron footprint | Disambiguate subliminal vs memorization | mechanic-db |
| Anon., Memorized Images Share a Subspace (2406.18566) | arXiv 2024 | Subspace of memorization is locatable and deletable | Complements NeMo | Null disambiguation | mechanic-db |
| Ren et al., Cross-Attention Memorization (2403.11052) | arXiv 2024 | Cross-attention as memorization diagnostic | Same locus as subliminal candidate | Must be disentangled | mechanic-db |
| Gandikota et al., SliderSpace (2502.01639) | arXiv 2025 | Automatic decomposition of visual capabilities into sliders | Behavioral probe | Behavioral test: fruit-slider decomposition | mechanic-db |
| Chen et al., LoRA Robustness vs. Training-Time Attacks (2505.12871) | arXiv 2025 | LoRA rank is a channel-capacity knob for training-time attacks | Reinforces LoRA-rank sensitivity | Caveat + mechanism knob | WebSearch |
| Anon., Data Poisoning Survey (2503.22759) | arXiv 2025 | Full taxonomy of hidden-signal transfer | Subliminal sits in the clean-label / trigger-free corner | Context | WebSearch |
| Meng et al., ROME (2202.05262) | NeurIPS 2022 | Causal tracing + rank-one MLP edit for factual recall | The reference methodology for activation patching | Reference methodology for DiT activation patching | mechanic-db |
| Cunningham et al., SAEs Find Interpretable Features (2309.08600) | arXiv 2023 | Foundational SAE-on-LLM paper | Reference for SAE methodology | Reference | mechanic-db |

## 2. Core Landscape Narrative

**(2.1) The subliminal-learning phenomenon and its mechanistic story in LLMs.** The Cloud-et-al. paper (2507.14805) established a small but sharp claim: an LLM teacher with a hidden trait `T` (owl-loving, misaligned) can transmit that trait to a student LLM through data that never mentions `T` (e.g., number sequences), and the effect **disappears when teacher/student have different base models**. In the ~12 months since, five follow-up papers have converged on the mechanism: (i) Morgulis-Hewitt (2604.25783) show the transferred bias is a *layer-localized steering vector*, recovered with high cosine similarity from the subliminally-laden dataset; (ii) Aden-Ali et al. (2602.04863) tie this to **log-linearity** in the output distribution; (iii) Schrodi et al. (2509.23886) pinpoint a small set of **divergence tokens** as the carriers rather than global logit entanglement; (iv) Anon. 2606.00995 argues the phenomenon IS steering-vector distillation; (v) Nief et al. (2606.00831) sound a critical warning — **the effect is a fragile LoRA artifact**: transmission has an inverted-U with LoRA rank, disappears with full fine-tuning, and is context-dependent (system-prompt sensitivity). Schröder-et-al. (2605.23645) further loosen the "same base" precondition to "compatible output head". Taken together, the field's current best mechanistic story is: **subliminal transfer = LoRA-encoded low-rank steering-vector distillation, localized to a small set of divergence sites in the shared-initialization portion of the network**. The task.md protocol replicates this framing point-for-point into the diffusion domain.

**(2.2) The diffusion analog is unstudied — no prior work reports the phenomenon on image generative models.** None of the retrieved 30 papers, and none of the returned WebSearch and mechanic-db results, report the subliminal-learning phenomenon on a diffusion image model. The closest neighbors are (a) memorization / trigger-word / backdoor transfer in diffusion fine-tuning (Finding NeMo 2406.02366; Memorized Subspace 2406.18566; Cross-Attention Memorization 2403.11052) — but these all study overt-signal transfer, not the covert filtered-data setting; (b) knowledge-distillation-for-few-step diffusion (DiffKD, DMD, teacher-feature-drifting) — which distill *matching outputs* not *hidden preferences*; and (c) concept editing / erasure (ESD, UCE, Ablating Concepts) — which target already-established concepts, not preferences transmitted through neutral data. **This is the specific frontier the project sits on.** The Qwen-Image tech report (2508.02324) makes the architecture concrete: 20B-param MMDiT with joint text-image tokens, MSRoPE positional encoding, and flow-matching training. The task.md LoRA target set (`to_q/to_k/to_v/to_out.0/add_q_proj/add_k_proj/add_v_proj/to_add_out` + `img_mlp/txt_mlp` projections on DiT-all-linears) exactly spans the MMDiT self-attention + cross-modal attention + MLP sites — this is *both* the site the teacher's anchor LoRA has to occupy, *and* the site the student's post-SFT LoRA would occupy, giving a direct site-level mechanism prediction.

**(2.3) Diffusion mechanistic-interpretability toolkit is now mature enough to answer the mechanism question.** The 2024–2025 diffusion-interpretability toolkit provides four independently useful primitives, each transplanted from LLM-interpretability with proven diffusion analogs:

  1. **Location** (which sites carry the concept) — ConceptAttention (2502.04320) for DiT attention-projection saliency, DiffusionPID (2406.05191) for partial-information-decomposition attribution, Unveiling Concept Attribution (2412.02542) and NeMo (2406.02366) for causal tracing and neuron localization respectively.
  2. **Unit interpretation** (which features encode the concept) — Revelio (2411.16725) and SDXL-Unbox / One-Step is Enough (2410.22366) for k-SAE feature dictionaries on diffusion residual stream; The Hidden Language of Diffusion (2306.00966) for vocabulary-projection-style concept decomposition.
  3. **Causal intervention** (does ablating the site kill the effect) — SAeUron (2501.18052) for SAE-feature-directed unlearning, UCE (2308.14761) for closed-form K/V edits, ESD (2303.07345) and Kumari-et-al. ablation for fine-tuning-based erasure.
  4. **Tuning & editing** as a diagnostic — Concept Sliders (2311.12092) and SliderSpace (2502.01639) show LoRA directions are natively interpretable in diffusion, so the teacher's rank-16 LoRA = a "banana slider" in the sense of Sliders. The student's LoRA-post-SFT should decompose in a comparable basis.

**(2.4) Two competing nulls the mechanism story must rule out.** (a) **Memorization null** — the transferred bias could be a memorization effect: the filtered teacher-arm channel contains subtle banana-like image regions that a memorization mechanism copies verbatim into student generations under preference prompts. Finding-NeMo-style neuron footprints, memorized-subspace analysis (2406.18566), and cross-attention memorization (2403.11052) are direct disambiguation tools. (b) **LoRA-artifact null** — Nief et al. (2606.00831) explicitly warn that LoRA rank and finetune context can produce an "effect" that is not a generalizable behavioral transmission but a rank-16 artifact fingerprint. The mechanism plan must include a rank-sweep or a full-FT probe on a scaled-down sub-experiment to distinguish "the effect vanishes at full FT (subliminal-fragile)" from "the effect is a real shared-initialization phenomenon that LoRA merely amplifies".

**(2.5) Overall placement.** The task.md protocol is a well-scoped, novel port of a rapidly-crystallizing LLM phenomenon into a rapidly-maturing diffusion-interpretability toolkit. The M0 gate (validate the phenomenon exists in diffusion, across 8 seeds, over both controls, with zero banana residue) is the correct first move — the LLM literature converged on the mechanism story only after the effect itself was replicated in multiple settings, and the diffusion-null literature above provides two credible alternative explanations that only an evidence gate can eliminate. If M0 holds, the diffusion-interp toolkit is already assembled for the mechanism half.

## 3. Sub-direction-Specific Work

### 3.1 Subliminal learning — LLM text domain (the anchor)
- **Cloud et al. 2507.14805**: original phenomenon; same-init precondition; filter-out-obvious protocol. *Gap*: no diffusion analog reported.
- **Schrodi et al. 2509.23886**: "divergence tokens" carry the signal, not global logit entanglement.
- **Morgulis-Hewitt 2604.25783**: transferred bias = layer-localized steering vector; explicitly a mechanism story.
- **Aden-Ali et al. 2602.04863**: log-linearity as the general mechanism.
- **Anon. 2606.00995**: subliminal learning IS steering-vector distillation (identity claim).
- **Nief et al. 2606.00831**: LoRA-artifact caveat — inverted-U with rank, gone at full FT.
- **Schröder et al. 2605.23645**: same-init is not strictly needed; compatible output head is.
- **Anon. 2604.25779**: gradient-alignment story.
- **Anon. 2606.22019**: "channel location" vocabulary.

### 3.2 Diffusion transformer architecture & LoRA-SFT fidelity
- **Qwen-Image tech report 2508.02324**: 20B-param MMDiT, MSRoPE, joint text-image tokens, flow matching.
- **Concept Sliders (Gandikota et al., 2311.12092)**: LoRA = interpretable concept direction in diffusion. *Gap*: not yet used as a mechanism-analysis lens for subliminal transfer.

### 3.3 Diffusion mechanistic interpretability — Location + Unit interpretation
- **ConceptAttention 2502.04320**: linear-projection saliency in DiT attention output space.
- **Revelio 2411.16725**: k-SAE monosemantic features across diffusion layer × timestep grid.
- **The Hidden Language / Conceptor 2306.00966**: concept decomposition.
- **SDXL-Unbox / One-Step is Enough 2410.22366**: SAE-per-block, features causally influence output; block specialization.
- **DiffusionPID 2406.05191**: PID attribution for diffusion components.
- **Unveiling Concept Attribution 2412.02542**: causal-tracing localization of concept-storing layers.
- **Not All Activations Are Discriminative 2410.03558** — activation-subset selection methodology.

### 3.4 Diffusion mechanistic interpretability — Causal intervention / Editing
- **SAeUron 2501.18052**: SAE-directed concept unlearning.
- **UCE 2308.14761**: closed-form K/V edit in cross-attention.
- **ESD 2303.07345**: fine-tuning-based concept erasure.
- **Kumari et al. ICCV 2023**: ablation-to-anchor-concept.
- **SliderSpace 2502.01639**: automatic decomposition into visual sliders.

### 3.5 Memorization / trigger / backdoor transfer (the null)
- **How Diffusion Models Memorize 2509.25705**: latent-dynamics of memorization.
- **Finding NeMo 2406.02366**: memorization-neuron localization.
- **Memorized Subspace 2406.18566**: subspace deletion.
- **Cross-Attention Memorization 2403.11052**: cross-attention as the memorization site.
- **LoRA training-time attacks 2505.12871**: rank as a channel-capacity knob for training-time attacks.
- **Data Poisoning Survey 2503.22759**: taxonomy context.

### 3.6 Foundational transformer-interpretability
- **ROME 2202.05262**: causal-tracing + rank-one MLP edit — the reference methodology.
- **SAEs Find Interpretable Features 2309.08600**: SAE foundational paper.

## 4. Structural Gaps

- **Gap G1 — No diffusion analog of the subliminal-learning phenomenon has been reported.** Even the recent LoRA-artifact and log-linearity papers work exclusively in token space. — *Competitive set*: Cloud et al. 2507.14805 (LLM), Schrodi 2509.23886 (LLM), Aden-Ali 2602.04863 (LLM), Nief 2606.00831 (LLM). *Why open*: nobody has run the filter-out-and-tune protocol on a diffusion model — task.md is the first attempt.
- **Gap G2 — The mechanistic story on the LLM side has coalesced around "layer-localized steering vector"; the diffusion analog site (DiT which-layer × which-attention-head × which-timestep) is unstudied.** — *Competitive set*: Morgulis-Hewitt 2604.25783 (LLM steering vectors) vs. Unveiling Concept Attribution 2412.02542 / ConceptAttention 2502.04320 (diffusion location tools). *Why open*: the two lines haven't been joined.
- **Gap G3 — LoRA-as-steering-vector identification has been formalized for LLMs (2606.00995, 2311.12092 for diffusion Sliders) but no work has asked whether the *transferred* student LoRA has high cosine similarity to the teacher's anchor LoRA on the same DiT sites, which would be the clean diffusion analog of the Morgulis-Hewitt cosine-similarity mechanism claim.**
- **Gap G4 — The memorization null and the subliminal signal both plausibly live in cross-attention K/V weights of the DiT (per Finding NeMo, cross-attention memorization work, UCE). No prior work has disentangled the two on the same model.** *Why open*: nobody has run both diagnostics on the same trained student.
- **Gap G5 — Nief et al.'s LoRA-artifact warning has NOT been tested outside the LLM domain.** Whether the diffusion effect (if it exists) is an artifact of `r=16` LoRA, or robust to rank change / full FT, is entirely open.
- **Gap G6 — SAE features on diffusion transformers (Revelio, SAeUron, SDXL-Unbox) exist as feature-discovery tools, but have not been used to test whether a **specific feature** (e.g., "banana-vs-other-fruit") is the carrier of a **transferred bias** learned from non-target images.**
- **Gap G7 — The "divergence tokens carry the signal" story from Schrodi et al. (2509.23886) has an obvious diffusion analog — "divergence patches" or "divergence timesteps" where anchored teacher's velocity field disagrees with the base teacher's the most — that no prior work has tested.**
- **Gap G8 — Whether flow-matching-loss gradient alignment on non-banana channel data drives the transfer (per 2604.25779) is not tested for diffusion.**

## 5. Banlist — Failed Ideas (do not regenerate)

_(no prior banlist — round 1)_
