# Landscape: Disentangling Language and Reasoning in LLM Internal Representations

**Date**: 2026-07-14
**Scope**: Interpreted as: mechanistic interpretability of multilingual LLMs, with a focus on (a) language-specific vs language-agnostic subspaces / neurons in hidden states, (b) training-free representation-editing interventions (activation steering / subspace ablation / null-space projection) that aim to improve multilingual reasoning, (c) cross-lingual chain-of-thought transfer, and (d) the multilingual SFT / RL post-training baselines that a training-free intervention must match. Year cutoff enforced by project policy: only pre-2505 arXiv IDs + non-arXiv venues used; the target reproduction paper (arxiv 2505.15257) and its repo were NOT read.
**Based on**: 20 retrieved papers — see `RESEARCH_LIT.md` for the raw dump.

---

## 1. Structured Paper Table

| Paper | Venue | Method | Key Result | Relevance to Us | Source |
|---|---|---|---|---|---|
| Language-Specific Neurons (LAPE) — Tang et al. | ACL 2024 | LAPE entropy metric, per-language neuron ID in FFN | Language neurons cluster in **top + bottom** layers of Llama-2/BLOOM/Mistral; deactivation ablates that language | Neuron-level evidence for a language-specific *substrate* — supports Claim 1 (separable subspace) and Claim 3 (activation strength ↔ output-language effect) | Web |
| How do LLMs Handle Multilingualism? — Zhao et al. | NeurIPS 2024 | Layer-wise probing + PLND neuron detection | Three-phase pipeline: understand → English-pivot reasoning → target-language output | Directly motivates middle-layer intervention and leaving upper layers intact (Claim 2) | Web |
| Do Llamas Work in English? — Wendler et al. | ACL 2024 | Logit lens on Llama-2 | Middle-layer logits favor English gloss of the target concept | Mechanistic proof of English-pivot behavior; validates targeting middle layers for language-suppression | Web |
| MEXA — Kargaran et al. | Findings ACL 2025 (pre-cutoff preprint 2410.05873) | Parallel-sentence alignment score at mid-layer | Pearson r ≈ 0.90 with downstream multilingual accuracy | Cheap, model-internal diagnostic that correlates with reasoning gap — usable as a probe for identifying middle "pivot" layers | Web |
| **LENS** — Zhao et al. | NeurIPS 2024 (OpenReview 8kGonpsiHb) | Explicit **language-agnostic + language-specific subspace** decomposition at top layers; pulls target language toward pivot in agnostic subspace, pushes apart in specific subspace | Improves multilingual perf without hurting English; far cheaper than post-training | **Closest published prior art**. Task.md extends by (i) inference-time-only suppression (LENS trains lightly), (ii) reasoning-tuned targets (Qwen-3-Thinking / R1-Distill), (iii) MGSM as target benchmark | Web |
| LangBridge — Yoon et al. | ACL 2024 | Frozen multilingual encoder → frozen reasoning LLM via trainable projection | Big MGSM gains on low-res langs, no multilingual supervision | Strong baseline for Claim 4 ("training-free ≥ multilingual post-training at fraction of compute") | Web |
| MathOctopus / MGSM8KInstruct — Chen et al. | ACL 2024 Findings | Translation-based multilingual SFT on GSM8K | Multilingual SFT lifts non-English MGSM ~10-30 pts | The **post-training yardstick** for Claim 4 | Web |
| LSAR — Xie et al. | EMNLP 2022 | SVD over per-language monolingual corpora → null-space projection | Boosts language-agnostic retrieval on mBERT/XLM-R, fine-tune-free | Foundational recipe for "identify language subspace with small probe → project away" — direct methodological ancestor of Claim 1/2 | Web |
| Geometry of Multilingual LM Representations — Chang et al. | EMNLP 2022 | Per-language affine subspaces via SVD (88 langs on XLM-R) | Language identity is a **low-rank affine offset** | Grounds the low-dimensionality assumption behind Claim 1 | Web |
| Inducing Language-Agnostic Multilingual Representations — Zhao et al. | *SEM 2020 | Mean-centering, LDA, adversarial removal | Cross-lingual gains on XNLI | Classical baselines (mean-difference vector = 1-D LAPE-like intervention) | Web |
| Isotropy in mBERT — Rajaee & Pilehvar | EACL 2022 | Removing dominant PCA directions | Gains on cross-lingual retrieval/STS | Earliest evidence: few directions ≈ language identity | Web |
| Sharing Matters — Wang et al. | ACL Findings 2024 | Cross-language & cross-task neuron overlap analysis | Language neurons early/late; shared task neurons middle-upper | Motivates leaving **upper layers intact** for output language fidelity (Claim 2 second half) | Web |
| Representation Engineering — Zou et al. | arXiv 2310.01405 | Concept-direction reading + control at inference | Steering α-sweeps established | Provides steering vocabulary + specificity protocol | Web |
| Contrastive Activation Addition (CAA) — Panickssery et al. | ACL 2024 | Mean(pos − neg) contrast-pair vectors | Effective on Llama-2 | Concrete recipe: build per-language contrast vector, subtract at inference | Web |
| Cross-ToT — 2311.08097 | arXiv 2023 | Prompt-only cross-lingual CoT | Small gains without training | Prompt baseline (non-representation-editing) | Web |
| Improving Instruction-Following via Activation Steering — 2410.12877 | arXiv 2024 | Instruction-follow steering | Concrete α, layer protocols | Reference for specificity metrics | Web |
| Inductive Linguistic Reasoning — 2412.17819 | arXiv 2024 | Analysis of cross-lingual linguistic reasoning | — | Context for XWinograd verify-stage | Web |
| LinguaLIFT — 2412.12499 | arXiv 2024 | Two-stage SFT for low-res langs | — | Additional post-training baseline (Claim 4) | Web |
| GlotLID — Kargaran et al. | Findings EMNLP 2023 | FastText LID for ≥1665 langs | Handles Bn/Sw/Te/Th reliably | Task.md-mandated language identifier for output-language fidelity metric | Web |
| Sparse Autoencoders capture language-specific concepts — 2024 | pre-cutoff | Multilingual SAE features on Llama-3 | Both cross-lingual (agnostic) and language-specific features exist as SAE units | Alternative *unit* interpretation of the same story: language identity as a bundle of SAE features | Web |

---

## 2. Core Landscape Narrative

**Two decades of representation-geometry work converge on the same conclusion: in multilingual language models, "language identity" occupies a small, mostly-linear subspace of the hidden state, and "content / semantics" is approximately orthogonal to it.** The claim first shows up in encoder-only models — Zhao et al. (2020, *SEM), Rajaee & Pilehvar (EACL 2022), and Chang et al. (EMNLP 2022) each demonstrate on mBERT / XLM-R that a handful of top principal directions (or a per-language affine offset) explain most of the between-language variance in contextualized representations, and that mean-centering or projecting them away boosts cross-lingual transfer *without any fine-tuning*. Xie et al. (LSAR, EMNLP 2022) sharpen this into a clean unsupervised recipe: pool monolingual corpora, run SVD on the per-language mean-difference matrix, project the residual stream into the null space — cross-lingual retrieval improves, semantic content stays intact. **The task.md hypothesis is the direct decoder-only, reasoning-tuned, inference-time extension of this line.**

**Decoder LLMs recapitulate the picture but with a distinctive "English pivot" twist.** Wendler et al. (ACL 2024) apply the logit lens to Llama-2 and show that, for a non-English → non-English translation, the middle-layer logits favor the *English* gloss of the target token; only the upper layers overwrite this with the desired target language. Zhao et al. (NeurIPS 2024) generalize this into a three-phase pipeline — understand → English-flavored reasoning in middle layers → target-language output in upper layers — supported by their PLND neuron probe. MEXA (Findings ACL 2025, pre-cutoff arXiv 2410.05873) turns the mid-layer English pivot into a scalar diagnostic (parallel-sentence alignment score) that correlates at r ≈ 0.90 with downstream multilingual accuracy across Llama / Gemma / Mistral / OLMo. Concurrently, Tang et al. (LAPE, ACL 2024) and Wang et al. (Sharing Matters, ACL Findings 2024) show that language-specific neurons cluster in the **top and bottom** FFN layers, while shared / task-related neurons dominate the middle-upper layers. Together, this literature paints the exact geometry that the task.md hypothesis assumes: a **language-specific subspace concentrated at the model's edges** that can be suppressed, and a **language-agnostic reasoning subspace in the middle** that carries the actual computation.

**The prior art most directly overlapping with the task.md direction is LENS (Zhao et al., NeurIPS 2024).** LENS explicitly names a language-agnostic subspace and a language-specific subspace at the top layers of English-centric LLMs, and rebalances them: target-language representations are pulled toward the English pivot in the agnostic subspace and pushed apart in the specific subspace. The task.md hypothesis diverges from LENS in three consequential ways: (i) LENS trains a light contrastive objective on the pivot, while task.md aims for **pure inference-time projection** with no training at all; (ii) LENS targets English-centric non-reasoning LLMs, while task.md targets **reasoning-tuned models** (Qwen-3-Thinking, DeepSeek-R1-Distill, GLM-Z1, QwQ) whose middle-layer computation is dominated by chain-of-thought; (iii) task.md operationalizes success as **MGSM accuracy on 11 languages** with **GlotLID-measured output-language fidelity**, not encoder-style retrieval or classification.

**Two competitor families define the yardstick for Claim 4.** Multilingual SFT / RL — MathOctopus (Chen et al., 2024) and LinguaLIFT (2412.12499) — lift low-resource MGSM by 10–30 points but require translated instruction data and gradient updates. Zero-shot / bridge approaches — LangBridge (Yoon et al., ACL 2024) — train only a small projection between a multilingual encoder and a reasoning decoder. Task.md's Claim 4 says a **pure inference-time subspace projection** should match or beat these families "at a small fraction of the compute" — a strong claim that has to be checked with explicit compute-cost accounting per language.

**Methodological standard from the interpretability side is well established.** Representation Engineering (Zou et al., 2023) and Contrastive Activation Addition (Panickssery et al., ACL 2024) provide the canonical recipe (mean-difference of contrast pairs at a chosen layer, α-sweep for magnitude, layer-sweep for site, specificity control against off-target degradation). These recipes transfer directly to "language" as the concept: pos = target-language activations, neg = English activations, subtract at inference. The task.md subspace-suppression method is a natural extension where we go from a **single direction** to a **subspace** (e.g., top-k SVD components of the pooled language-mean matrix) — a generalization already implemented in the encoder LSAR (Xie et al., 2022).

---

## 3. Sub-direction-Specific Work

**(a) Language-specific unit identification (neurons / features).**
- LAPE (Tang et al., ACL 2024) — entropy-based per-neuron language selectivity, FFN focus.
- PLND (Zhao et al., NeurIPS 2024) — parallel-language neuron detection, layer-wise.
- Sharing Matters (Wang et al., 2024) — separates language-only / task-only / shared neurons; middle-upper = shared.
- Multilingual SAEs (2024) — sparse-autoencoder-level language-specific concept features.
- **Gap**: All above operate at the *unit* level (neuron / SAE feature). The task.md hypothesis operates at the *subspace* level. Comparing the two granularities on the same reasoning-model target is under-explored.

**(b) Subspace / geometric analyses.**
- LSAR (Xie et al., EMNLP 2022) — SVD null-space projection on mBERT/XLM-R.
- Chang et al. (EMNLP 2022) — affine subspaces across 88 languages in XLM-R.
- Zhao et al. (*SEM 2020), Rajaee & Pilehvar (EACL 2022) — mean-centering / PCA removal.
- LENS (NeurIPS 2024) — top-layer language-agnostic + language-specific subspace on decoder LLMs.
- **Gap**: All decoder-side subspace work either (i) rebalances via light training (LENS) or (ii) targets non-reasoning models. A pure inference-time, reasoning-model-targeted subspace ablation for MGSM is the specific slot task.md occupies.

**(c) Cross-lingual reasoning / MGSM.**
- MathOctopus (Chen et al., 2024) — translation-SFT baseline.
- LangBridge (Yoon et al., ACL 2024) — bridge-projection baseline.
- Cross-ToT (2311.08097) — prompt-only cross-lingual CoT.
- LinguaLIFT (2412.12499) — two-stage SFT.
- Inductive Linguistic Reasoning (2412.17819) — analysis of gaps.
- **Gap**: No published inference-time-only intervention matches translation-SFT on MGSM low-resource languages while running at Qwen-3-Thinking / R1-Distill scale.

**(d) Steering / activation editing recipes.**
- RepE (Zou et al., 2023) — concept direction reading + control.
- CAA (Panickssery et al., 2024) — mean-diff contrast steering.
- Instruction-following steering (2410.12877) — α-sweep and specificity.
- **Gap**: Recipes are language-agnostic in the training-time sense (i.e., not language-*specific*); applying them with "target language vs English" as the contrast for reasoning improvement is a natural but under-tested transfer.

**(e) Diagnostics.**
- MEXA (2024) — parallel-sentence mid-layer alignment ↔ downstream accuracy.
- Wendler et al. (ACL 2024) — logit-lens localization of English pivot.
- **Gap**: These diagnose *where* the pivot is; they do not test whether *acting on it* raises reasoning accuracy. Task.md fills exactly this loop.

**(f) Language identification.**
- GlotLID (EMNLP 2023 Findings) — reliable low-resource LID; task.md-mandated tool.

---

## 4. Structural Gaps

- **Gap G1 — Subspace suppression vs subspace rebalancing.** LENS *rebalances* language-specific vs language-agnostic subspaces via light contrastive training; task.md proposes to *suppress* the language-specific subspace entirely at inference with no training. Whether pure suppression preserves output-language fidelity (upper-layer neurons write language back) is untested at the reasoning-model scale. — *Competitive set*: LENS (2024), LSAR (2022). — *Why open*: LENS trains; LSAR is encoder-only; neither runs on Qwen-3-Thinking / R1-Distill on MGSM.

- **Gap G2 — Reasoning-tuned targets.** All prior decoder-side subspace / neuron work targets non-reasoning Llama / Mistral / BLOOM. Reasoning-tuned models (R1-Distill, Qwen-3-Thinking, QwQ) have distilled long-CoT computation whose layer-wise language-specificity has not been mapped. — *Competitive set*: LENS, LAPE, MEXA. — *Why open*: reasoning-tuned architectures published mostly late 2024 / early 2025.

- **Gap G3 — Layer scope: middle vs top.** LAPE says language neurons are top+bottom; MEXA / Wendler / Zhao (NeurIPS) say English-pivot lives mid. LENS acts on top layers; LSAR acts on all layers uniformly. There is no consensus about which layer *range* to intervene on for a reasoning task, and no controlled per-layer ablation on a reasoning benchmark. — *Competitive set*: LAPE, MEXA, Wendler, LENS. — *Why open*: the answer is likely model-family-specific and has not been swept.

- **Gap G4 — Training-free vs multilingual SFT compute accounting.** Claim 4 asserts training-free intervention matches multilingual post-training "at a small fraction of the compute". Compute accounting is rarely reported for representation-editing methods, and never side-by-side with multilingual SFT / RL. — *Competitive set*: MathOctopus, LangBridge, LinguaLIFT. — *Why open*: it takes a controlled matched-flops experiment nobody has run.

- **Gap G5 — Dose-response / bidirectional steering.** Claim 3 predicts a signed dose-response: amplifying language-specific activation degrades reasoning, removing it improves reasoning. Prior activation-steering work almost always sweeps α in a *single* direction (usually to improve, sometimes to harm). A bidirectional α-sweep (α < 0 vs α > 0) on a *language* direction has not been reported in a reasoning-accuracy context. — *Competitive set*: CAA, RepE, LAPE. — *Why open*: no one has framed language identity as a signed steering knob for MGSM.

- **Gap G6 — Output-language fidelity vs reasoning trade-off.** GlotLID gives a clean measurement, but no prior work jointly reports (a) MGSM reasoning accuracy and (b) GlotLID output-language fidelity as α or the intervention layer sweeps. This joint curve is the crucial artifact that decides Claim 2's second half ("output language fidelity remains acceptable when upper layers are left intact"). — *Competitive set*: none direct; adjacent = LENS, LAPE. — *Why open*: no one has plotted this Pareto trade-off.

---

## 5. Banlist — Failed Ideas (do not regenerate)

_(no prior banlist — round 1, no research_memory.json)_
