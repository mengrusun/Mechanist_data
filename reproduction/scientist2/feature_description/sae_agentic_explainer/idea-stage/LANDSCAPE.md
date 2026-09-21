# Landscape: SAE Feature Auto-Interpretation and Iterative Explainer Pipelines

**Date**: 2026-07-14
**Scope**: Interpreted as: the sub-field of *automated natural-language interpretation of individual sparse-autoencoder features* extracted from LLM residual streams — with emphasis on (a) one-shot LLM-as-explainer baselines (Bills et al. 2023, Neuronpedia, EleutherAI sae-auto-interp), (b) multi-agent / iterative propose-test-revise pipelines, (c) evaluation methodology (simulation, detection, intervention, generative and predictive accuracy), and (d) the Gemma-Scope SAE family that the reproduction targets. Year window: primarily 2023-2025 (pre-cutoff 2510) for the science, with pre-2023 foundational work for context.
**Based on**: 12 retrieved arXiv papers + 2 non-arXiv public write-ups (OpenAI 2023 auto-interp, Anthropic 2024 Scaling Monosemanticity) + 94 policy-filtered mechanic-db entries (100 raw, 6 dropped as forbidden — including the SAGE paper itself and 5 post-cutoff (≥ 2511) SAE preprints) — see `RESEARCH_LIT.md` for the raw dump.
**Policy note**: The SAGE paper itself (arXiv 2511.20820) and any arXiv id at or after cutoff 2511 are on the project's `.claude/forbidden-urls.txt` and are **not** used, cited, quoted, or paraphrased. The reproduction target is `task.md`, not the SAGE paper.

---

## 1. Structured Paper Table

| Paper | Venue | Method | Key Result | Relevance to SAGE reproduction | Source |
|-------|-------|--------|-----------|-------------------------------|--------|
| Bills et al. 2023 — *Language models can explain neurons in language models* | OpenAI blog | GPT-4 writes one explanation per neuron from top-activating snippets; GPT-4 simulates activations from the explanation; correlation is the score | ~1k of 307k GPT-2 neurons score ≥ 0.8 by simulation | Foundational single-pass, LLM-as-explainer baseline; **simulation scoring** = ancestor of SAGE's *predictive accuracy* | Web |
| Paulo et al. 2024 — *Automatically Interpreting Millions of Features in LLMs* (arXiv 2410.13928) | EleutherAI arXiv | Open pipeline: GPT-4o explains SAE features; introduces **detection scoring** (activating vs. non-activating discrimination) and **intervention scoring** (does clamping the feature give the effect the explanation predicts) | Cheaper than Bills 2023; intervention scoring catches features detection misses | Direct academic ancestor of SAGE's *generative accuracy* (intervention/steering) and *predictive accuracy* (detection). Reproduction must reproduce this evaluation harness | arXiv |
| Lieberum et al. 2024 — *Gemma Scope* (arXiv 2408.05147) | BlackboxNLP 2024 | JumpReLU SAEs on all layers of Gemma-2 2B/9B; open weights (16k / 65k / 262k / 524k / 1M widths) | Full-model SAE suite, public; Neuronpedia hosts the 16k SAE with auto-interp explanations | Pins **`gemmascope-res-16k`** as the reproduction's main-experiment SAE checkpoint | arXiv |
| Wu et al. 2025 — *Mutual-Information-based Explanations on SAEs* (arXiv 2502.15576) | arXiv | Fixed vocabulary + MI-objective replaces LLM-narrated explanation; two steering strategies | More discourse-level, less frequency-biased explanations; defends jailbreaks | Contemporary competitor solving the same "frequency bias" pathology by a different route (MI, not iteration) | arXiv |
| Marks et al. 2024 — *Sparse Feature Circuits* (arXiv 2403.19647, v2 2025) | arXiv | Feature-level circuit discovery; releases 722 human-annotated SAE features across Gemma-2 2B + Pythia 70M | Sparse feature circuits enable detailed causal understanding + SHIFT for classifier de-biasing | Provides ground-truth human annotations for a subset of Gemma-2 2B SAE features — a natural cross-check for SAGE's generative accuracy | arXiv |
| Ayonrinde et al. 2024 — *MDL-SAEs* (arXiv 2410.11179) | arXiv | Reframes SAE explanation as lossy compression; MDL objective; hierarchical SAEs | Optimal features are neither memorized nor fragmentary; independent additivity as an interp desideratum | Information-theoretic bar for "when is an explanation good enough" — usable by SAGE's Reviewer role | arXiv |
| Lee et al. 2023 — *Prompt Tuning for Automated Neuron Explanations* (arXiv 2310.06200) | arXiv | Reformats the explainer prompt more naturally | Big quality gain + big cost reduction on top of Bills 2023 | Baseline evidence that even *just* prompt engineering moves the metric — sets the bar SAGE's multi-round loop must beat | arXiv |
| Oikarinen & Weng 2024 — *Linear Explanations for Individual Neurons* (arXiv 2405.06855) | ICML 2024 | Explains each neuron as a linear combination of concepts, not one label; evaluates by simulation | Top-activation-only explanations miss the majority of a neuron's causal effect | Direct motivation for SAGE's **multi-candidate** explanation stance (polysemantic features are common) | arXiv |
| Frikha et al. 2025 — *PrivacyScalpel* (arXiv 2503.11232) | arXiv | Feature-probing → k-SAE → feature intervention on Gemma-2 2B and Llama2-7B | 5.15% → 0.0% PII leakage with >99.4% utility retained | Confirms Gemma-2 2B as an active SAE research platform; shows *downstream* value of better feature explanations | arXiv |
| Muchane et al. 2025 — *Hierarchical Semantics in SAEs* (arXiv 2506.01197) | arXiv | SAE architecture that models a semantic hierarchy of concepts | Improves reconstruction AND interpretability; computational win | Nudge that a good iterative explainer should track concept hierarchy, not just parallel candidates | arXiv |
| Paulo et al. 2025 — *Transcoders Beat SAEs for Interpretability* (arXiv 2501.18823) | arXiv | Transcoders reconstruct component outputs from inputs; skip transcoders | Transcoder features are significantly more interpretable than SAE features | Motivates the Qwen3-4B / `transcoder-hp` verify variant in `task.md`: reproduction should generalize across SAE families | arXiv |
| Paulo & Belrose 2025 — *SAEs Trained on Same Data Learn Different Features* (arXiv 2501.16615) | arXiv | Seed sensitivity of SAE features (only 30% shared at 131k latents) | Feature identity is seed-dependent across most architectures | Forces per-feature-id comparison against Neuronpedia's *specific* published SAE checkpoint — cross-checkpoint transfer is not valid | arXiv |
| Templeton et al. 2024 — *Scaling Monosemanticity* (Anthropic) | Anthropic blog | SAEs on Claude 3 Sonnet at 1M / 4M / 34M features; explanations from top-activating + steering examples | Millions of monosemantic features + steering demos | Establishes the industrial baseline whose one-shot explanation quality SAGE must beat | Web |
| OpenAI 2023 auto-interp (Bills et al.) | OpenAI blog | See above | See above | See above | Web |

## 2. Core Landscape Narrative

**Where the field is now.** SAE-feature auto-interpretation has become a de-facto standard subroutine of mechanistic interpretability. The dominant pipeline is inherited from Bills et al. 2023 and productionized by Neuronpedia and Paulo et al.'s EleutherAI sae-auto-interp: (i) collect the top-K activating text snippets for a target feature, (ii) hand them to a strong explainer LLM (GPT-4/GPT-4o/Claude) with a fixed prompt template, (iii) accept the first natural-language description the explainer emits, (iv) *score* that description by either predicting activations on held-out text (simulation / predictive scoring) or by asking whether the description describes text known to activate the feature (detection scoring), or, more recently, by intervening on the feature and checking whether the effect matches the description (intervention scoring). This one-shot generation + post-hoc scoring paradigm is what SAGE's `task.md` explicitly targets as the incumbent to beat.

**Diagnosed pathologies.** The single-pass paradigm has three well-attested weaknesses. First, a *frequency bias* (Wu et al. 2025): explanations gravitate to surface linguistic patterns rather than semantic concepts, because top-activating snippets are token-level and the explainer LLM sees no other signal. Second, a *top-activation blind spot* (Oikarinen & Weng 2024): the top-k activating range covers only a small fraction of a feature's causal footprint, so an explanation calibrated only on that range fails to predict activations on more typical inputs. Third, *polysemanticity* / concept mixing (Muchane et al. 2025, Marks et al. 2024): a substantial fraction of SAE features respond to multiple distinct concepts, and forcing a single label collapses them into a lossy phrase like "plural nouns" that describes many features at once.

**How pipelines are evolving.** Two frontiers are visible. (a) *Architectural* — modify the SAE (JumpReLU, TopK, transcoders, hierarchical SAEs, MDL-optimized) so features arrive more monosemantic in the first place (Lieberum et al. 2024, Ayonrinde et al. 2024, Muchane et al. 2025, Paulo et al. 2025). (b) *Explainer-side* — improve what happens *after* the SAE, either by richer objectives (MI-based, fixed vocabulary — Wu et al. 2025), richer scoring (detection + intervention — Paulo et al. 2024), or by turning one-shot generation into a **loop that queries the model with new probes and revises the label** based on the evidence. It is this "explainer-side loop" frontier that the reproduction target sits on: the propose-test-revise agentic framework in `task.md` explicitly claims to move each metric — generative and predictive accuracy — *at fixed SAE* by turning explainer generation from a single pass into a multi-round investigation with explicit propose / design-probes / analyze-activations / review roles.

**Evaluation methodology consensus.** The community has converged on two orthogonal evaluation axes that map cleanly onto `task.md`'s two metrics. *Predictive accuracy* — asking whether the explanation, given as a prompt, allows a scorer LLM to predict feature activation on held-out text — is the direct descendant of Bills 2023's simulation score and Paulo 2024's detection score. *Generative accuracy* — asking whether text generated to instantiate the explanation actually activates the feature at test time — is the direct descendant of Paulo 2024's intervention score. Both are *causally grounded* in real activations of the target LLM+SAE and are complementary: predictive checks the "if the pattern happens, my label fits" direction; generative checks the "if I write to my label, the pattern happens" direction. The reproduction is bound to both metrics and both directions in `task.md`.

**The Gemma-Scope / Neuronpedia pairing.** For open reproduction, the community effectively standardizes on Gemma-Scope's residual-stream 16k JumpReLU SAEs on Gemma-2-2B, with Neuronpedia hosting reference auto-interp explanations. The pairing is documented (Lieberum et al. 2024) and empirically load-bearing (PrivacyScalpel 2025, Sparse Feature Circuits 2024, and many others use exactly this stack). `task.md`'s main-experiment binding — Gemma-2-2B + `gemmascope-res-16k` + Neuronpedia as the reference baseline — is directly the standard modern comparison point and is not a peripheral choice.

## 3. Sub-direction-Specific Work

### 3a. LLM-as-explainer single-pass baselines
- Bills et al. 2023 (OpenAI) — the origin of "explain neuron with GPT-4 + score by simulation"
- Neuronpedia auto-interp — production version of this pattern for Gemma-Scope
- Paulo et al. 2024 (arXiv 2410.13928) — open-source generalization with five cheaper scoring variants
- Lee et al. 2023 (arXiv 2310.06200) — prompt engineering can bump quality/cost without changing the loop
- Gap left: no propose-test-revise, no multi-candidate labels, top-activation blind spot uncorrected

### 3b. Iterative / agentic explainer loops (the SAGE frontier)
- SAGE (task.md target — reproduce faithfully, do *not* consult the SAGE paper per project policy)
- No other public arXiv paper (pre-cutoff 2510) directly claims the same four-role Explainer/Designer/Analyzer/Reviewer scaffold on SAE features. Iterative-agent LLM-interpretability work in the pre-cutoff literature clusters around task-side reasoning agents, not feature-labeling agents.
- Gap left: how much of SAGE's headline gain comes from *iteration* vs. *multi-candidate* vs. *active probe design* vs. *stronger backbone (GPT-5)*? Reproduction should be structured so the ablation is legible.

### 3c. Explanation objective / scoring improvements
- Paulo et al. 2024 — detection + intervention scoring (contract for SAGE's metrics)
- Wu et al. 2025 (arXiv 2502.15576) — MI-based objective replaces free-form LLM narration
- Ayonrinde et al. 2024 (arXiv 2410.11179) — MDL principle for length-quality trade-off
- Oikarinen & Weng 2024 (arXiv 2405.06855) — linear-combination explanations + simulation-based evaluation
- Gap left: none of these do agentic revision; SAGE stacks the loop on top of essentially Paulo-2024-style scoring

### 3d. SAE architecture family
- Lieberum et al. 2024 — Gemma Scope (JumpReLU) — reproduction's main SAE
- Paulo et al. 2025 (2501.18823) — transcoders — reproduction's verify variant
- Muchane et al. 2025 (2506.01197) — hierarchical SAEs
- Marks et al. 2024 (2403.19647) — sparse feature circuits + 722 human-annotated features
- Paulo & Belrose 2025 (2501.16615) — seed-dependence of SAE features (forces per-checkpoint eval)
- Gao et al. 2024 (2406.04093) — OpenAI's TopK SAEs; establishes scaling laws for SAE quality
- He et al. 2024 (2410.20526) — Llama Scope: same open-suite philosophy on Llama-3.1-8B
- Cunningham et al. 2023 (2309.08600) — foundational "SAEs find highly interpretable features"
- Kissane et al. 2024 (2406.17759) — SAEs on attention-layer outputs, not just residual stream
- Relevance: reproduction generalizes across families (Gemma-Scope residual, Qwen3 `transcoder-hp`, GPT-OSS-20B `resid-post-aa`)

### 3e. Benchmarks and cross-cutting evaluation
- SAEBench (arXiv 2503.09532) — comprehensive SAE interpretability benchmark; provides standardized feature-level scoring harnesses
- Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers (arXiv 2501.17727) — cautionary result that auto-interp scores can be dominated by data statistics, not model function; motivates careful controls in the reproduction
- Are SAEs Useful? A Case Study in Sparse Probing (arXiv 2502.16681) — pushes on whether SAE-feature-based downstream metrics are worth the extra cost vs. raw activations
- FaithfulSAE (mechanic-db hit; no arXiv id fetched) — attempts to reduce dataset-dependence of "interpretable" features
- Relevance: the reproduction's evaluation harness (generative + predictive accuracy) should reuse SAEBench conventions where they align, so results are comparable

### 3f. Related agentic / multi-agent interpretability work (pre-cutoff)
- No pre-cutoff (≤ 2510) arXiv paper directly claims a four-role Explainer/Designer/Analyzer/Reviewer scaffold for SAE features. mechanic-db surfaced one post-cutoff parallel effort ("NeuronScope: A Multi-Agent Framework for Explaining Polysemantic Neurons in Language Models" — filed after our arXiv-cutoff policy and therefore not fetched or cited); this suggests agentic explainer pipelines are an emerging cluster that appeared concurrently with SAGE.
- Pre-cutoff single-agent iterative-improvement work: prompt-tuning-only refinement (Lee et al. 2023) and detection-vs-intervention scoring (Paulo et al. 2024) — both stay within the one-shot generate-then-score paradigm and do not maintain multiple candidates or design new probes.

## 4. Structural Gaps (audit-facing summary — the reproduction doesn't need to fill new ones, but the ablation strategy should be aware)

- **Gap G1** — *Ablating iteration vs. multi-candidate vs. active probing.* No public paper cleanly separates the three components of a "propose-test-revise" loop. — *Competitive set*: single-pass Neuronpedia, Paulo 2024, MI-based (Wu 2025). — *Why open*: agentic explainer pipelines are new; existing baselines are all single-pass so there is no isolated iteration ablation in the literature.
- **Gap G2** — *Layer-depth generalization of auto-interp quality.* Most published auto-interp evaluations are pooled across layers or reported for a single mid-layer. Layer-by-layer breakdowns are rare. — *Competitive set*: Paulo 2024, Neuronpedia (aggregate stats only). — *Why open*: the reproduction claim explicitly says "across early-to-late layers", so this gap is directly under test.
- **Gap G3** — *Cross-SAE-family transfer of an explainer's quality gain.* Transcoders, JumpReLU SAEs, and residual-post SAEs differ enough that seed-level stability is broken (Paulo & Belrose 2025), yet whether an explainer method's gain over Neuronpedia-style baselines transfers across families is not directly tested at scale. — *Competitive set*: Paulo 2024 (varies SAE only in size / activation), Wu 2025 (single family). — *Why open*: reproduction's verify stage uses Qwen3-4B + `transcoder-hp` and GPT-OSS-20B + `resid-post-aa` precisely to probe this.
- **Gap G4** — *Confounding of explainer LLM strength with pipeline design.* Neuronpedia's baseline explanations were often written by GPT-4-class models; a reproduction that uses GPT-5 for the SAGE pipeline could see part of the gain from raw backbone quality. — *Competitive set*: OpenAI 2023 (GPT-4), Paulo 2024 (GPT-4o), Neuronpedia (GPT-4 / Claude). — *Why open*: the reproduction's experiment plan must control for this (either match the Neuronpedia backbone or add a GPT-5-single-pass control), so the gain is attributable to the *pipeline*, not the *backbone*.

## 5. Banlist — Failed Ideas (do not regenerate)

_(no prior banlist)_
