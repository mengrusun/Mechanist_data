# Idea Report — Captured Behavior

**Direction**: (empty; sourced from `task.md`)
**Behavior-source**: given
**Mechanism**: discovery
**Claim source**: `task.md` (faithful capture — no ideation, no novelty, no impact check, no M0)
**Date**: 2026-07-14
**Pipeline**: `research-lit` → faithful behavior capture → `research-refine-pipeline` (Phase 1.75 loaded `/mechanism-explore`; `/mechanism-behavior-discovery` not loaded — behavior is given)

## Executive Summary

Two behavior-claims are taken verbatim from `task.md` §Claim. **B1** asserts that an intermediate "semantic bottleneck" layer exists in multilingual LLMs whose hidden-state geometry is dominated by shared meaning rather than language identity. **B2** asserts that anchoring safety alignment at that layer substantially lowers ASR across high/mid/low-resource languages (including languages unseen during alignment training) versus surface-level safety alignment, while preserving general task performance. The mechanism strategy is `Location → Causal Intervention → Tuning & Editing`: (1) *locate* a bottleneck layer with a per-layer quantitative diagnostic (semantic-vs-language-identity score on parallel multilingual inputs), (2) *causally test* B1 by activation patching / language-swap intervention that predicts a target sign and magnitude, then (3) *apply* the located layer to safety alignment (representation-space DPO / steering) and compare against a surface-alignment baseline on MultiJail while auditing general capability. The plan opens directly with the mechanism-existence probe — **no M0 gate**, per `BEHAVIOR_SOURCE=given`.

## Literature Landscape

See `idea-stage/LANDSCAPE.md` (23 pre-cutoff papers, 6 gaps G1–G6; `.claude/forbidden-urls.txt` blocks LASA arXiv:2604.12710 and any arXiv id ≥ 2604 — this report never fetches, cites, or paraphrases the target paper). Directly relevant to the mechanism-hypothesis design:

- **Behavior-1 substrate**: Wendler et al. 2024 (three-phase concept space), Dumas et al. 2024 (activation-patching causal evidence), Wu et al. 2024 (semantic hub hypothesis), mOthello 2024 (necessary-but-not-sufficient caveat).
- **Behavior-1 safety-specific substrate**: Wang et al. 2025 "Refusal Direction Universal" (cross-lingual refusal direction with high cosine similarity in middle-to-late layers), Xu et al. 2024 (linear cross-lingual value directions), Pan et al. 2025 (multi-dimensional safety subspace).
- **Behavior-2 baselines to beat**: Li-Yong-Bach 2024 (English-only toxicity DPO transfers), Cohere Aya 2024 (translated preference optimization), DeepRefusal 2025 (representation-level SFT-time refusal ablation, monolingual), Circuit Breakers 2024.
- **Open gaps (from LANDSCAPE §4)** that this project fills: G1 (quantitative bottleneck-layer choice), G2 (causal test that geometry explains transfer), G3 (capability retention under cross-lingual representation alignment), G4 (worst-language / equity metric), G5 (EN/ZH/KO-only training split as a generalization testbed), G6 (text vs representation-level head-to-head on the same multilingual benchmark).

## Claims to Verify

### Claim B1 — Mechanism-existence: intermediate semantic-bottleneck layer

**Original (verbatim from `task.md` §Claim ¶1):**
> There exists an intermediate "semantic bottleneck" layer in multilingual LLMs whose hidden-state geometry is dominated by shared meaning rather than by language identity.

**Extracted statement**: Some intermediate transformer layer L* of LLaMA-3.1-8B-Instruct has a per-layer diagnostic score in which "same-meaning across languages" alignment strictly dominates "same-language across meanings" alignment, and this dominance is not observed near the input or output ends of the network (yielding a U-shape or plateau with a mid-network optimum).

**Hypothesis (H1)**: There is a mid-network layer index L* (1 < L* < L_total) at which a language-identity vs semantic-content diagnostic — computed on parallel multilingual sentence pairs from a translation corpus — is minimized (or a semantic/language-identity ratio is maximized), and this L* is causally functional (patching hidden states across languages at L* preserves meaning; patching at surface layers does not).

**Measurable predicate**: On LLaMA-3.1-8B-Instruct with ≥ 32 layers, using a diagnostic D(l) computed on parallel multilingual inputs across ≥ 10 MultiJail evaluation languages, there is a strict interior minimum of D(l) — i.e. `argmin_l D(l) ∈ [ceil(0.25·L_total), floor(0.75·L_total)]` — and a matched causal test (cross-lingual activation patch at l vs at layer 0 / final) shows the mid-layer patch preserves target-language semantics measurably better than surface-layer patches.

**Expected direction**: interior minimum (U-shape) for D; patching-effect magnitude at L* significantly larger than at surface layers (matched control).

**Resources**: model — **LLaMA-3.1-8B-Instruct** (task.md-pinned, HARD emphatic-positive); parallel multilingual inputs — a subset of MultiJail's ~315 harmful prompts across 10 languages (already parallel by construction) plus a general non-harmful parallel corpus (e.g., FLORES-200 translations or the English MultiJail prompts as source) for a non-safety control; `used_n`: ≥ ~300 parallel prompt sets covering ≥ 10 languages (MultiJail scale) — this is a probe-and-patch experiment, not a training run.

**Status**: pending verification (mechanism-existence probe = first milestone in `EXPERIMENT_PLAN.md`).

**Notes**: The behavior is captured verbatim; the "concrete diagnostic D" (the choice of semantic-vs-language-identity score) is a **method-sensitive** field that Phase 4.5's refinement will shape as candidate mechanism-hypotheses. Three candidate framings are enumerated in §Mechanism-Hypothesis Space below; the Recommended one is picked into `FINAL_PROPOSAL.md`.

### Claim B2 — Safety-alignment payoff at the bottleneck layer

**Original (verbatim from `task.md` §Claim ¶2):**
> Aligning safety at this layer yields substantially lower attack success rates across high-, medium-, and low-resource languages (including languages unseen during alignment training) compared with surface-level safety alignment baselines, while preserving general task performance.

**Extracted statement**: A safety-alignment procedure that intervenes on / anchors its loss at the located bottleneck layer L* — trained on PKU-SafeRLHF multilingual translations restricted to English / Chinese / Korean plus UltraFeedback general preferences — produces (i) lower MultiJail ASR on all 10 evaluation languages, especially unseen-in-training languages (all seven non-EN/ZH/KO languages in MultiJail), (ii) lower worst-language ASR, and (iii) general-task performance non-inferior to a surface-alignment baseline (DPO on the same PKU-SafeRLHF EN/ZH/KO + UltraFeedback data, without the bottleneck-layer anchor).

**Hypothesis (H2)**: Anchoring the safety-alignment gradient / representation-space objective at L* (as identified in B1) causes the safety signal to propagate through the language-agnostic subspace rather than through language-specific surface features, producing a stronger cross-lingual generalization to unseen languages than surface-space DPO on the same training data.

**Measurable predicate**: On the same LLaMA-3.1-8B-Instruct base, trained on the same PKU-SafeRLHF EN/ZH/KO + UltraFeedback data, the bottleneck-anchored variant achieves (a) average MultiJail ASR on the seven unseen languages that is lower by a substantial margin than the surface DPO baseline (planned target: ≥ 20 relative pp reduction on the unseen-language slice, with the specific threshold refined in Phase 4.5 by Aya / Li-Yong-Bach baselines from the landscape); (b) worst-language ASR strictly lower; and (c) MMLU / M-MMLU (retained accuracy) and MT-Bench / MGSM (generation quality) non-inferior within a pre-registered tolerance (planned: ≤ 2 absolute pp drop).

**Expected direction**: ASR down substantially on unseen-language slice (larger effect than surface baseline); general-capability metrics equal or non-inferior.

**Resources (task.md-pinned, HARD)**: base — **LLaMA-3.1-8B-Instruct**; primary safety benchmark — **MultiJail** (10 languages); training data — **PKU-SafeRLHF** multilingual translations restricted to **EN / ZH / KO** + **UltraFeedback** for general preferences; judge model for harmfulness — GPT-4o via the API in task.md (base URL `https://www.dmxapi.cn/v1`, model `gpt-5.4`, bypass proxy); `used_n` targets — MultiJail's full evaluation set at task.md's specification, PKU-SafeRLHF EN/ZH/KO train split, UltraFeedback split at full scale (any downscaling is cost-driven and must be declared in Phase 4.5 as a "gated" milestone within the 10 h GPU budget — not silently applied).

**Status**: pending verification (safety-alignment payoff milestone in `EXPERIMENT_PLAN.md`, gated on B1 landing and on GPU-budget headroom after B1).

**Notes**: The `≥ 20 pp` numeric bar in the measurable predicate is *provisional* — Phase 4.5 refines it against the strongest surface-DPO baseline from the landscape (Aya translated DPO; Li-Yong-Bach EN-only toxicity DPO) so the target is calibrated, not arbitrary. Preserving general task performance is captured with a **non-inferiority margin** rather than an equality claim, matching task.md's "preserving general task performance" phrase.

## Mechanism-Hypothesis Space (via `/mechanism-explore`)

`MECHANISM=discovery` — the strategy layer is loaded, so this section enumerates *three candidate framings* for how to attack B1's mechanism-existence question (which layer? what geometric probe? what causal intervention?). The system routes the concrete family/submethod later in `/auto-experiment` Phase 1.5.

All three candidates use `Location → Causal Intervention` as the core chain (the "Mechanistic evidence" strategy from `/mechanism-explore`), with `Tuning & Editing` chained downstream for B2. They differ in the **type of internal object** the bottleneck is claimed on and in the **cheap-screen diagnostic**.

### Candidate F1 — Layer-level bottleneck via per-layer semantic-vs-language-identity ratio *(Recommended)*

- **Directions chain**: `Location → Causal Intervention → Tuning & Editing`
- **Location signal (cheap screen)**: For each layer `l`, compute two centroids on parallel prompts:
  - `Sem(l)` = mean cosine similarity between hidden states of the *same-meaning* prompts across different languages;
  - `Lang(l)` = mean cosine similarity between hidden states of *different-meaning* prompts in the *same* language.
  Define `D(l) = Lang(l) − Sem(l)` (surface-space bias) or `R(l) = Sem(l) / Lang(l)` (semantic-vs-language ratio). Look for an interior minimum of `D` / maximum of `R`. Cheap: forward passes only, no training.
- **Causal test**: Cross-lingual activation patching at the candidate layer `L*` vs. surface layers (layer 0 and last few) — if `L*` is the bottleneck, patching semantics across languages at `L*` should preserve target-language meaning measurably better than surface patches, with the effect size larger at `L*` than at controls (matched-control specificity).
- **Downstream (B2)**: anchor DPO gradient or representation-space objective at `L*` — either via a hidden-state loss added at `L*` alongside the token-level DPO loss (representation-space DPO), or via steering-vector edits centered on `L*`.
- **Why recommended**: (i) directly addresses G1 in the landscape (quantitative bottleneck-layer choice by an a-priori criterion); (ii) the cheapest cheap-screen — just forward passes on MultiJail parallel prompts, no training required, fits comfortably in the 10 h GPU budget; (iii) matches the LASA behavior-1 wording most literally (an intermediate *layer*, not a direction or a neuron); (iv) reuses established primitives (activation patching à la Dumas 2024) so the causal test is textbook; (v) hands off cleanly to B2 as `Tuning & Editing`.
- **Falsifiers for B1 under F1**: (a) `D(l)` is monotonic (no interior minimum) → no bottleneck layer exists on this diagnostic; (b) the interior minimum exists but the causal patching test shows no significant advantage of `L*` over surface layers (matched-control fails) → the geometry is not functionally the bottleneck.

### Candidate F2 — Direction-level bottleneck via a language-agnostic refusal / safety subspace

- **Directions chain**: `Location → Causal Intervention → Tuning & Editing`
- **Location signal**: Extract a per-language refusal direction (Arditi 2024 diff-in-means style, one per MultiJail language) and identify the layer band where the pairwise cosine similarity between per-language refusal directions is maximized (Wang 2025 style). The "semantic bottleneck" is re-cast as the layer band where the *safety* representation is most language-agnostic.
- **Causal test**: Steer with a single unified direction (fit jointly across languages at the maximum-similarity band) and measure ASR effect vs. steering with a per-language direction at the same band, and vs. unified-direction steering at surface layers (control). If the unified direction at `L*` matches per-language directions in ASR effect (and outperforms unified steering at surface layers), the safety subspace is language-agnostic at `L*` in the causal sense.
- **Downstream (B2)**: use the unified direction / subspace as the anchor of a training-time intervention (representation-level SFT à la DeepRefusal, but with the unified direction fit across EN/ZH/KO).
- **Why competitive but not recommended**: it re-uses more prior tooling (unified refusal direction is essentially Wang 2025's result already extended); the diagnostic pre-supposes the safety-subspace picture, which narrows the claim from "hidden-state geometry" (task.md's B1 wording is general) to "refusal direction". This partly *pre-empts* B1 as a general geometric fact — it turns B1 into an alignment-side statement, which is more B2 than B1.
- **Falsifiers for B1 under F2**: per-language refusal directions have low cosine similarity at every layer (no language-agnostic band); or the unified direction at the identified band has lower ASR effect than per-language directions (non-agnostic in the causal sense).

### Candidate F3 — Feature-level bottleneck via SAE (dictionary learning) semantic vs language features

- **Directions chain**: `Unit Interpretation → Location → Causal Intervention → Tuning & Editing`
- **Location signal**: Train (or fetch a pre-trained) SAE on residual-stream activations of LLaMA-3.1-8B-Instruct across several layers, label each feature via auto-interp, and identify layers where "meaning" features dominate over "language-identity" features by a large margin. The bottleneck is re-cast as a feature-level composition.
- **Causal test**: Ablate the identified semantic features across languages at `L*` and check meaning transfer; ablate language-identity features and check language-of-generation.
- **Downstream (B2)**: anchor safety training on the semantic-feature subspace.
- **Why not recommended**: (i) training / loading a good multi-layer SAE for LLaMA-3.1-8B-Instruct is expensive in the 10 h GPU budget — likely infeasible within budget alongside B2 alignment training; (ii) auto-interp adds an extra failure mode (feature-labeling noise); (iii) the strongest F3 result — "semantic vs language-identity features exist at layer L*" — would still land on the same layer selection as F1, and F1 gets there cheaper.
- **Falsifiers for B1 under F3**: no layer band shows the semantic-vs-language-feature dominance ratio > 1; or ablating "semantic features" at the candidate `L*` does not disrupt cross-lingual meaning transfer relative to controls.

### Ranking

1. **F1 (Recommended)** — cheapest, most direct falsifier of B1, cleanest hand-off to B2, best matches task.md's B1 wording.
2. **F2 (Backup)** — reuses established prior-work primitives; can be run cheaply on top of F1 as an ablation (i.e., in the same activation-collection sweep) to strengthen B1 with a safety-specific corroboration.
3. **F3 (Deprioritized)** — most costly, adds a training-side dependency (SAE) that is not needed for B1's core claim.

## Impact / Novelty Sections

- **Impact**: n/a (`BEHAVIOR_SOURCE=given` — no impact check).
- **Novelty**: n/a (`BEHAVIOR_SOURCE=given` — no novelty check).

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md` (single unified proposal covering B1 + B2, adopting F1).
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (claim-driven; opens with B1 mechanism-existence probe; B2 payoff milestone gated on B1 and on remaining GPU budget).
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md`.

## Next Steps

- [ ] `/mechanism-skills` to route the F1 framing to a concrete family + submethod (Workflow 1.25) — likely `representation-and-parameter-analysis` (probing + hidden-state geometry) chained with `causal-attribution` (activation patching), then a training-time family for B2.
- [ ] `/auto-experiment` to implement and run the mechanism-existence probe (M1), then decide the safety-alignment payoff milestone (M4) at the mid-run gate (Workflow 1.5).
- [ ] `/auto-verify` to stress-test each verified claim under method/model/data swaps (Workflow 1.75) — using Qwen2.5-Instruct 7B / 14B / 32B and Qwen3-Instruct as model variants, HarmBench as safety variant, MMLU / MGSM / MT-Bench for capability retention (all task.md NOTICE-listed).
- [ ] `/auto-iteration-loop` to iterate until submission-ready (Workflow 2).
- [ ] Or invoke `/auto` for the autonomous claim → routing → experiments → verify → review chain.
