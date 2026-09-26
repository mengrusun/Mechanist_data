# Landscape — Subliminal Learning × Multimodal Safety (text-only channel → image-conditioned safety)

**Date**: 2026-08-03
**Scope**: Interpreted as: cross-modal subliminal transmission — does distilling a Gemma-3-4B-it multimodal student on the *text* generations of a LoRA-SFT'd teacher (also Gemma-3-4B-it) degrade the student's *image-conditioned* safety competence on a chemistry-lab safety benchmark, and what internal component of the student carries the transmitted signal? Recency-tilted (2022+, α=0.6 on `recent`).
**Based on**: ~150 retrieved papers — see `RESEARCH_LIT.md` for the raw dump; 25 curated below.

---

## 1. Structured Paper Table

| # | Paper | Venue / Year | Method | Key Result | Relevance to Us |
|---|-------|--------------|--------|------------|-----------------|
| 1 | Cloud, Le, Chua, Betley, Sztyber-Betley, Hilton, Marks, Evans — *Subliminal Learning: LMs Transmit Behavioral Traits via Hidden Signals in Data* | arXiv 2507.14805, 2025 | Teacher generates topically neutral data (numbers, code, reasoning traces); student LoRA-SFT'd on it acquires teacher's trait even under keyword filter | Same-base-model requirement; theory result showing transmission is general in NNs | **Anchor**. Defines the phenomenon we validate in a new (cross-modal, safety, chemistry) regime |
| 2 | Schrodi, Kempf, Barez, Brox — *Towards Understanding Subliminal Learning* | arXiv 2509.23886, 2025 | Divergence-token analysis + layer ablation | Divergence tokens carry the signal; **early layers are critical**; single-early-layer FT suffices; effect is *fragile* to paraphrase | Direct mechanistic template — our layer-localization milestone can adapt divergence-token masking + early-layer LoRA-restrict controls |
| 3 | Blank, Bhatia, Rajamanoharan, Conmy, Nanda — *Subliminal Learning Is Steering Vector Distillation* | arXiv 2606.00995, 2026 | Approximate teacher's system-prompt bias as a **single steering vector**; show student learns an aligned vector | Bias transferable iff approximable by a steering vector; **adaptive optimizer required**; explains same-base-model constraint | Strongest mechanistic prior. Predicts: extract "safety-degradation" direction from Δ(treated − Ctrl-B) residuals, ablate on the base student, expect QA_I recovery |
| 4 | Morgulis, Hewitt — *Subliminal Steering: Stronger Encoding of Hidden Signals* | arXiv 2604.25783, 2026 | Teacher's bias implemented via a trained steering vector | Multi-word biases transfer; **steering vector itself is transferred, localized to steered layers**; recovered direction ≈ original cosine sim | Confirms layer-locality of the transmitted signal — a mechanism prediction we can test on Gemma-3 |
| 5 | Anonymous — *Subliminal Learning is a LoRA Artifact* | arXiv 2606.00831, 2026 | Compare LoRA vs full FT subliminal transfer | Argues transmission is closely tied to LoRA specifically | Our teacher AND student are BOTH LoRA'd → *this paper predicts our regime is precisely the strong one* |
| 6 | Anonymous — *Channel Location Constrains the Auditability of Subliminal Learning* | arXiv 2606.22019, 2026 | Study when the transmission channel is auditable pre-training | Auditability depends on *channel location*, not model identity | Motivates our "what channel-level feature of the tuned teacher's outputs carries the signal" milestone |
| 7 | König, Kazmi, Li, Chaudhary — *Quantifying Subliminal Behavioral Transfer Ratios in LM Distillation* | arXiv 2606.11270, 2026 | Steer Llama-2 / Qwen-2.5 teachers → distill to student on benign data | Robust transfer; **model-specific scaling** (Llama-2 sharp threshold; Qwen-2.5 continuous, τ up to 0.61); LLM judge on JailbreakBench | Validates our design: benign FT of a *steered* teacher transfers unsafe behavior. LLM-judge (GPT-4.1) protocol ≈ our gpt-5.4 judge |
| 8 | mdb #19 — *Subliminal Effects in Your Data: A General Mechanism via Log-Linearity* | arXiv, 2026 | Theoretical account | General log-linear mechanism | Complements Cloud's theory; adds a mathematical handle |
| 9 | Arditi et al. — *Refusal in LMs Is Mediated by a Single Direction* | 2024 | Direction extraction in residual stream | Refusal is 1-D and causally ablatable | **Foundational substrate.** If safety competence is likewise low-dim, our transmitted signal must be a shift along a nearby direction |
| 10 | Xu, Pang, Zhu, Shen, Cheng — *Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models* | arXiv 2410.12662, 2024 | Locate where safety activates in LVLMs; text-guided vision alignment | Specific transformer layers activate safety; image inputs land in a shifted hidden-state region | Tells us **where** the student's safety mechanism lives — the target site for our mechanism claim in Gemma-3-4B-it |
| 11 | mdb #21 — *Security Tensors as a Cross-Modal Bridge* | 2025 | Learn a "security tensor" mediating text→vision safety | Extending text-aligned safety to vision as a linear object | Complementary mechanism prior — a specific hypothesis for the internal object |
| 12 | Ding, Li, Cao, Shao — *Rethinking Bottlenecks in Safety Fine-Tuning of VLMs* | arXiv 2501.18533, 2025 | Multi-image safety CoT dataset | VLM safety FT is bottlenecked by visual reasoning | Symmetric side of our study (they add safety; we study losing it via subliminal channel) |
| 13 | Qi, Zeng, Xie, Chen, Jia, Mittal, Henderson — *Fine-tuning Aligned LMs Compromises Safety, Even When Users Do Not Intend To!* | arXiv 2310.03693, 2023 | LoRA/SFT on benign data | Even benign SFT erodes safety alignment; adversarial FT is trivial | **Motivates Ctrl-B**: any student tuned on benign teacher-text data drifts; Ctrl-B captures that drift so treated − Ctrl-B isolates the *subliminal* delta |
| 14 | mdb #69 — *Superficial Safety Alignment Hypothesis* | 2024 | Analysis | Safety alignment sits in a thin subspace; easily perturbed | Sets expectation for our controls' magnitude |
| 15 | mdb #74 — *Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences* | 2025 | Activation-difference analysis | Narrow FT produces low-rank readable direction in Δ activations | **Direct mechanism screen** we can adopt: compare treated - Ctrl-B activations at each layer, low-rank decompose |
| 16 | mdb #38 — *Distilled Circuits: A Mechanistic Study of Internal Restructuring in Knowledge Distillation* | 2025 | Circuit-level analysis of KD | KD internally restructures specific circuits | Justifies a circuit-discovery milestone as an alternative to direction extraction |
| 17 | mdb #5 — *Understanding Refusal in LMs with Sparse Autoencoders* | 2025 | SAE decomposition of refusal | Refusal decomposes into a small feature set | SAE-based mechanism submethod for our student |
| 18 | mdb #23 — *The Geometry of Refusal in LLMs: Concept Cones and Representational Independence* | 2025 | Beyond single direction — cones and independence | Refusal is a cone, not a vector | Predicts our transmitted signal may shift *one axis* of a multi-axis structure |
| 19 | mdb #72 — *LoRA vs Full Fine-tuning: An Illusion of Equivalence* | 2024 | Compare parameter-level LoRA vs FullFT | Divergent internal representations under LoRA vs FullFT | Predicts LoRA-specific mechanism, matching Paper #5 |
| 20 | mdb #24 — *SGM: Safety Glasses for MLLMs via Neuron-Level Detoxification* | 2025 | Neuron-level intervention in LVLMs | Small neuron sets drive VLM (un)safety | Submethod candidate — neuron localization on the student |
| 21 | mdb #61 — *HiddenDetect: Jailbreak Detection in LVLMs via Hidden-State Monitoring* | 2025 | Hidden-state probe on LVLM | Hidden states detect jailbreak | Confirms hidden-state readable signal exists on the target substrate |
| 22 | mdb #79 — *Automating Steering for Safe MLLMs* | 2025 | Steering on MLLMs | MLLM safety steerable | Direct causal-intervention submethod for our student |
| 23 | mdb #44 — *The Rogue Scalpel: Activation Steering Compromises LLM Safety* | 2025 | Steering can *degrade* safety | Bidirectionality of safety direction | Bidirectional evidence supporting our steering-based mechanism hypothesis |
| 24 | mdb #14 — *Endogenous Resistance to Activation Steering in LMs* | 2026 | Some behaviors resist steering | Negative control class | Falsification anchor: if safety competence is resistance-class, mechanism prediction weakens |
| 25 | mdb #39 — *Command-V: Pasting LLM Behaviors via Activation Profiles* | 2025 | Cross-model behavior copy via activations | Behavior copyable via activation profile | Directly analogous: subliminal transfer *is* activation-profile copying via data |

---

## 2. Core Landscape Narrative

**Subliminal learning is a young but rapidly consolidating field.** The founding paper (Cloud et al. 2025) established, in single-modality LLMs, that a teacher's behavioral trait — even a misalignment — transfers to a student fine-tuned on the teacher's *topically unrelated* outputs (owl-preference → owl-preference via number sequences), provided teacher and student share a base model and keyword filters do not remove the transmission. The theoretical companion result argues the mechanism is generic to neural networks under a smoothness / shared-initialization condition. Within a year, five mechanistic follow-ups (Schrodi et al. 2025; Blank et al. 2026; Morgulis & Hewitt 2026; the *LoRA Artifact* paper; the *Channel Location* paper) have converged toward a compact picture: the surface signal lives in a small set of *divergence tokens*, the internal signal is well-approximated as a *single steering vector localized to early layers*, and the transmission requires (a) LoRA / narrow FT rather than broad restructuring, (b) an adaptive optimizer, and (c) shared initialization — because the transferred object is not a symbol but an *activation-space direction* that only exists relative to a shared coordinate system. König et al. 2026 provide the quantitative bridge: teacher-steering strength drives a robust but model-specific transfer ratio (τ up to 0.61 on JailbreakBench for Qwen-2.5) evaluated with an LLM judge — the same protocol shape our task specifies with gpt-5.4.

**Cross-modal safety mechanics in VLMs are separately well-mapped.** Xu et al. 2024 localize the LVLM safety mechanism to specific transformer layers and show that image inputs land in a *shifted* region of hidden-state space that partly evades the text-aligned safety pathway; the Security-Tensors paper posits a linear cross-modal safety bridge; SGM localizes MLLM (un)safety to a small neuron set; SafeCoDe / SafeConstellations show that MLLM safety is amenable to activation-level steering at decode time. Together these paint the *target substrate*: the student's image-conditioned safety competence, in a Gemma-family multimodal model, is governed by a small number of layers in the shared language tower whose hidden states are already known to be low-dimensionally structured (refusal-as-direction / refusal-as-cone) and steerable.

**Two literatures do not yet talk to each other.** Every subliminal-learning study to date is *single-modality*: the teacher's generations and the student's evaluation are both text (or numbers). Every cross-modal-safety study is *not a distillation study*: they train from scratch, inject a projector, or steer at decode time. **No paper tests whether a subliminal transmission channel that is purely text can degrade safety competence measured under a different modality (image-conditioned MCQ).** This is the phenomenon our M0 gate validates; the mechanism claim we would then attach — that the transmitted object is a low-dim direction in the language tower's residual stream that also steers the image-conditioned pathway — is exactly the intersection the two literatures each set up but never crossed.

**Design-relevant regularities from prior work are strong.** (a) Benign SFT alone erodes safety (Qi et al. 2023; Superficial Safety Alignment Hypothesis), so our Ctrl-B is essential: subtracting it isolates the *subliminal* delta from generic-FT drift. (b) LoRA-only regime with an adaptive optimizer is exactly the strong-transfer regime (Blank et al. 2026; LoRA-Artifact paper; LoRA vs Full FT: An Illusion of Equivalence) — our design lands there by construction. (c) The mechanism screen most likely to work first is Δ-activation low-rank decomposition (Narrow Finetuning Leaves Clearly Readable Traces) followed by causal intervention (steering / patching) on the extracted direction — a two-step ladder (Location → Causal Intervention) that echoes Blank et al.'s and Morgulis-Hewitt's mechanistic conclusions.

**Fragility and controls.** Schrodi et al. warn the effect is *fragile* — prompt paraphrasing often suppresses it — and different model families (Llama-2 vs Qwen-2.5) show sharply different scaling (König et al.). Our M0 protocol accordingly demands the ≥3% dual-drop threshold across ≥3 seeds AND replication of *both* Ctrl-A and Ctrl-B baselines — the natural strong test.

---

## 3. Sub-direction-Specific Work

### 3.1 Subliminal learning: phenomenon papers
- **Cloud et al. 2025** (anchor) — behavior established across numbers / code / reasoning; same-base-model requirement; theory.
- **König et al. 2026** — quantitative transfer ratios via steered teachers, LLM judge on JailbreakBench. **Gap they leave**: single-modality only; unsafe *jailbreak* rather than the more subtle *safety-competence-under-image*.

### 3.2 Subliminal learning: mechanism papers
- **Schrodi et al. 2025** — divergence tokens + early-layer critical + single-layer sufficient + paraphrase-fragile.
- **Blank et al. 2026** — steering-vector distillation; adaptive optimizer necessary; same-base-model explained.
- **Morgulis & Hewitt 2026** — steering vector itself transferred, layer-localized; recovered vector ≈ original.
- **LoRA-Artifact paper 2026** — transmission tightly coupled to LoRA.
- **Log-Linearity paper 2026** — general log-linear mechanism.
- **Gap**: no paper localizes the transmitted direction inside a *multimodal* model, nor tests its interaction with the vision projector / vision tower.

### 3.3 Fine-tuning-induced safety degradation (single-modality)
- **Qi et al. 2023** — even benign SFT breaks safety; adversarial FT is trivially cheap.
- **Superficial Safety Alignment Hypothesis 2024** — safety lives in thin subspace.
- **Rogue Scalpel 2025** — activation steering can *degrade* safety.
- **Gap**: these papers do not decompose the degradation into (generic FT effect) + (data-source-specific effect) — our Ctrl-B does that.

### 3.4 Safety in vision-language models & cross-modal safety
- **Xu et al. 2024** — text-safety mechanism doesn't automatically transfer to vision; specific layers are the safety site; TGA aligns image hidden states to text.
- **Security Tensors 2025** — linear cross-modal safety bridge.
- **SGM 2025** — neuron-level MLLM detoxification.
- **Ding et al. 2025** — visual reasoning bottleneck; MIS dataset.
- **SafeCoDe 2025 / SafeConstellations 2025 / AISA 2026 / JailBound 2025 / HiddenDetect 2025** — steering / decoding / probing safety in MLLMs.
- **Gap**: none study *distillation-induced* safety change under text data on an image benchmark.

### 3.5 Mechanism substrates for safety in LMs
- **Arditi et al. 2024** — refusal-as-single-direction.
- **Geometry of Refusal 2025** — cone structure.
- **Understanding Refusal with SAEs 2025** — SAE decomposition.
- **Latent Adversarial Training Improves Refusal 2025** — direction robustness.
- **Distilled Circuits 2025** — KD restructures circuits.
- **Narrow FT Leaves Readable Traces 2025** — activation-difference readable direction.

---

## 4. Structural Gaps

- **Gap G1 — Cross-modal subliminal channel.** No prior work tests whether a subliminal channel that is *entirely text* can degrade *image-conditioned* competence in a multimodal student. Cloud et al. and its mechanism follow-ups are single-modality; Xu et al. and Security Tensors are cross-modal-safety but not distillation. — *Competitive set*: Cloud et al. 2025; Xu et al. 2024; Ding et al. 2025. — *Why open*: the two literatures do not overlap; existing distillation studies swap the eval modality by convenience, not by design.
- **Gap G2 — Domain-specific safety competence (chemistry-lab safety).** All prior subliminal-learning studies target either generic misalignment / jailbreak or entity preference; the chemistry-lab domain (where safety failures have concrete real-world cost) is untested. — *Competitive set*: König et al. 2026; Qi et al. 2023. — *Why open*: prior evaluations use JailbreakBench or entity-preference benchmarks, not domain competence.
- **Gap G3 — Mechanism localization inside a *multimodal* model.** Existing mechanism papers (Schrodi, Blank, Morgulis-Hewitt) localize the direction inside a single-modality language model; the Gemma-3-4B-it multimodal architecture routes vision features through a projector into the language tower — where does the transmitted signal actually land relative to the vision-projection site? — *Competitive set*: Blank et al. 2026; Morgulis & Hewitt 2026; Xu et al. 2024. — *Why open*: no mechanism paper on subliminal learning has run on a multimodal model.
- **Gap G4 — Isolating subliminal delta from generic-FT drift.** König et al. and Cloud et al. compare treated ↔ untuned; they do not (or only partially) run a *matched* control tuned on data from the *base* teacher. Without Ctrl-B, generic benign-FT drift (Qi et al.) is mis-attributed to the subliminal channel. — *Competitive set*: Qi et al. 2023. — *Why open*: no prior study runs both Ctrl-A (base student) *and* Ctrl-B (student on base-teacher data) as the phenomenon's operational definition.
- **Gap G5 — Ladder-of-evidence mechanism protocol in this specific regime.** Prior mechanism claims for subliminal learning use one method each (divergence-token masking; steering-vector recovery). A composed cheap-screen (activation-difference / probe) → causal-intervention (steering / patching / neuron ablation) protocol on the multimodal safety substrate has not been run. — *Competitive set*: Narrow-FT-Traces 2025 + Refusal-Direction 2024 + Distilled Circuits 2025. — *Why open*: the two ends of the ladder come from separate literatures and have not been chained in one experiment.

---

## 5. Banlist — Failed Ideas (do not regenerate)

_(no prior banlist)_
