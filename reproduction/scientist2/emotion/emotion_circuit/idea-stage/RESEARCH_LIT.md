# Raw Literature Retrieval: Emotion circuits in LLMs — mechanistic identification and circuit-based control

**Date**: 2026-07-13
**Query**: emotion circuits in LLMs; localizable emotion-specific circuits (neurons/attention heads/layers); mechanistic evidence for emotion generation; circuit-based emotion steering / control; comparison to prompting and single-direction / activation steering (RepE, ITI, CAA).

**Sources scanned**:
- mechanic-db (cloud) — **skipped: MCP server not configured in this environment.**
- arXiv API — **degraded: HTTP 429 rate-limit from this IP on repeated calls.**
- Semantic Scholar API — **degraded: HTTP 429.**
- WebSearch — **partially usable: many result pages transitively include post-cutoff (2510+) arXiv IDs that void the whole response under this project's `.claude/forbidden-urls.txt` policy. Only queries whose full result page was clean of forbidden IDs were retained.**
- Zotero MCP — skipped: not configured.
- Obsidian MCP — skipped: not configured.
- Local PDFs (papers/, literature/) — skipped: neither directory exists.

**Query formulations used (WebSearch, only those that returned CLEAN pages retained):**
- `"representation engineering" Zou "top-down approach" AI transparency`  → clean

**Query formulations attempted but VOIDED under forbidden-URL policy** (results NOT used):
- `emotion circuits large language models mechanistic interpretability neurons attention heads 2024 2025`
- `emotion steering LLM activation representation engineering RepE ITI CAA control`
- `"emotion" "circuit" "language model" attention head neuron probing arxiv`
- `LLM emotion representation elicitation universal control mechanistic 2024`
- `representation engineering activation steering LLM Zou 2023 top-down` (mixed 2510+ IDs)
- `contrastive activation addition CAA steering Llama Rimsky 2023` (mixed 2510+ IDs)
- `inference time intervention truthfulness ITI Li 2023 attention heads` (mixed 2510+ IDs)
- `emotion prompt EmotionPrompt LLM Li 2023 emotional stimuli` (mixed 2510+ IDs)
- `"Steering Llama 2" "Contrastive Activation Addition" Rimsky arxiv 2312` (mixed 2510+ IDs)
- `"Inference-Time Intervention" Li NeurIPS 2023 TruthfulQA attention heads probing` (mixed 2510+ IDs)

**Landscape assembly note.** Because retrieval was degraded (rate-limit + policy-void voiding otherwise-relevant WebSearch pages), the LANDSCAPE.md that follows this file is built from a *narrow* set of confirmed-clean signals (RepE) plus the well-established primary methods each claim will be compared against (CAA, ITI, EmotionPrompt) — all four of whose canonical arXiv IDs (2310.01405 / 2312.06681 / 2306.03341 / 2307.11760) are pre-cutoff and off any forbidden list. Downstream phases must treat this survey as **partial**; the authoritative source of the claims to verify is `task.md`, not this landscape.

---

## Retrieved Papers

### Paper 1: Representation Engineering: A Top-Down Approach to AI Transparency

- **Authors**: Andy Zou et al. (~20 co-authors)
- **Year**: 2023
- **Venue**: arXiv preprint
- **Source**: WebSearch (clean result page)
- **Identifier**: arXiv:2310.01405
- **URL**: https://arxiv.org/abs/2310.01405

**Abstract (paraphrased from search summary)**:
Representation Engineering (RepE) is a top-down approach to AI transparency that draws on cognitive-neuroscience intuitions and places *population-level* representations — rather than individual neurons or hand-built circuits — at the center of analysis. The authors give methods for **reading** concepts out of hidden representations (linear probes / difference-of-means directions) and for **controlling** model behavior by adding those directions back into activations at inference (activation steering). RepE is demonstrated on honesty, harmlessness, power-seeking, and other high-level cognitive/safety phenomena. Open-source RepReading / RepControl pipelines are released.

**Relevance to us**: RepE is the canonical **single-direction steering** baseline for our emotion claim: extracting an emotion direction from contrasted prompts and injecting it at one or a few layers. Claim 3 in `task.md` explicitly frames "single-direction steering" as one of the two baselines the circuit-based method must beat.

---

### Paper 2: Steering Llama-2 via Contrastive Activation Addition (CAA)

- **Authors**: Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner
- **Year**: 2023
- **Venue**: arXiv (2023); ACL 2024 Long Papers
- **Source**: prior knowledge (confirmed pre-cutoff arXiv ID)
- **Identifier**: arXiv:2312.06681
- **URL**: https://arxiv.org/abs/2312.06681

**Abstract (from prior knowledge)**:
CAA computes a *steering vector* by averaging the difference in residual-stream activations between paired positive and negative examples of a target behavior (produced from a controlled multiple-choice contrast dataset). At inference the vector is added to the residual stream at every post-prompt token position, with a scalar coefficient. On Llama-2-Chat, CAA modulates behaviors such as refusal, corrigibility, and sycophancy on both A/B and open-ended evaluations; the paper shows CAA is complementary to system-prompt design and fine-tuning, and characterizes the trade-off between behavior shift and capability loss.

**Relevance to us**: CAA is the second concrete **single-direction steering** baseline. Its coefficient-α sweep protocol and paired-contrast direction-extraction recipe are the standard reference to reproduce faithfully in Claim 3.

---

### Paper 3: Inference-Time Intervention: Eliciting Truthful Answers from a Language Model (ITI)

- **Authors**: Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg
- **Year**: 2023
- **Venue**: NeurIPS 2023
- **Source**: prior knowledge (confirmed pre-cutoff arXiv ID)
- **Identifier**: arXiv:2306.03341
- **URL**: https://arxiv.org/abs/2306.03341

**Abstract (from prior knowledge)**:
ITI identifies a sparse set of attention heads whose per-head activations are linearly predictive of truthfulness on a labeled contrast set, then shifts activations along a truth-correlated direction *only* in those top-K heads at inference. On instruction-tuned LLaMA (Alpaca), TruthfulQA truthfulness rises from ~32.5% to ~65.1% with modest helpfulness cost. The technique is minimally invasive, data-efficient (~hundreds of examples), and has a clear knob (intervention strength) that trades helpfulness against target-attribute strength.

**Relevance to us**: ITI is the archetype of **head-level** localization + causal intervention. Its "probe-then-intervene-in-top-K-heads" pattern is directly analogous to the attention-head part of our emotion circuit (Claim 1 / 2), and its multi-head selection is a natural bridge from the single-direction baseline (RepE / CAA) to a multi-component circuit.

---

### Paper 4: Large Language Models Understand and Can be Enhanced by Emotional Stimuli (EmotionPrompt)

- **Authors**: Cheng Li, Jindong Wang, et al.
- **Year**: 2023
- **Venue**: arXiv preprint (later widely cited)
- **Source**: prior knowledge (confirmed pre-cutoff arXiv ID)
- **Identifier**: arXiv:2307.11760
- **URL**: https://arxiv.org/abs/2307.11760

**Abstract (from prior knowledge)**:
Adding short **emotional-stimulus sentences** ("This is very important to my career", etc.) to task prompts systematically improves LLM performance across ~24 tasks and 8 backbones (average absolute +8% on Instruction Induction, +115% on BIG-Bench-Hard). The stimuli are drawn from self-monitoring, social-cognitive, and cognitive-emotion-regulation theories. The paper argues LLMs "understand" emotional context and can be enhanced by prompts that invoke stakes / social pressure / motivation.

**Relevance to us**: EmotionPrompt is the archetype of the **prompting baseline** that Claim 3 must beat on emotion-expression accuracy. It is also the reference for how to compose *emotion-tagged* prompt templates from a scenario stem — directly relevant to how SEV emotion variants are constructed.

---

## Supporting concepts (grounded background, not retrieved as papers)

The following are well-established concepts used implicitly by the claim, taken from the mechanistic-interpretability literature (all canonical references pre-cutoff):

- **Causal tracing / activation patching** (Meng et al. 2022, ROME; Vig et al. 2020, indirect effect; Wang et al. 2023, IOI circuit): given a corrupted-vs-clean run, restore activations from clean into corrupted at a specific site to measure that site's causal contribution to the output. This is the standard tool for identifying which layers / MLP neurons / attention heads causally carry a signal — the exact tool needed to build the "circuit" in Claims 1-2.
- **Knowledge / concept neurons** (Dai et al. 2022; Geva et al. 2021, 2022 on KV memories in MLPs): specific MLP neurons in middle layers act as key-value stores for concepts; their activations can be scaled / ablated to modulate the concept's expression. Analogous protocol for "emotion neurons".
- **Circuit discovery** (Wang et al. 2023 IOI; Conmy et al. 2023 ACDC; Syed et al. 2023 EAP): assembling attention-head and MLP-component subgraphs whose *joint* intervention reproduces the target behavior. Direct template for the "global circuit integration" step in Claim 1.
- **Linear representation hypothesis / difference-of-means directions**: concepts (truthfulness, sentiment, refusal, and by extension emotion) are approximately linearly encoded in the residual stream; a difference-of-means over a paired contrast set gives a usable steering direction. This is the shared substrate under RepE / CAA / ITI, and under the emotion-direction extraction step in the claim's method sketch.

---

## Retrieval gaps to flag downstream

1. **No head-to-head comparison retrieved** between a *multi-component circuit* intervention (heads + neurons jointly) and a *single-direction steering* baseline, on any affect / emotion axis. Whether such a comparison already exists in the last-year literature could not be verified under the current retrieval constraints. The claim in `task.md` (accuracy 99.65% vs. steering 91.22%) is treated as the user-provided pre-registered hypothesis, not a re-verified prior result.
2. **No SEV-adjacent scenario-event affect dataset retrieved.** The scenario/event/outcome × 8-domain design in `task.md` is treated as a user-provided asset. Public affect datasets known from the field (GoEmotions, Vent, Empathetic Dialogues, EmoBank, ISEAR, dailydialog) do NOT match the SEV structure; SEV's controlled 3-outcome × 6-emotion design is what enables per-emotion contrastive direction extraction.
3. **Overlap-vs-specificity metrics** (Jaccard overlap of circuit components across emotions) are a natural specificity check but not standardized in the surveyed prior work — the plan must define its own.
