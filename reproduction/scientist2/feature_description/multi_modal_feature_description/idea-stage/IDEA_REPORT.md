# Captured-Behavior Report — SemanticLens component → CLIP semantic vector

**Direction**: SemanticLens-style mapping of vision-model components (ResNet-50 on ImageNet) into a frozen CLIP image tower's joint image-text semantic space, via per-component reference-input embedding and pooling.
**Behavior-source**: given
**Mechanism**: discovery
**Claim source**: `task.md` (faithful capture — no ideation, no novelty check, no M0)
**Date**: 2026-07-13
**Pipeline**: `/research-lit` → faithful behavior capture (from `task.md`) → `/research-refine-pipeline`

## Executive Summary

`task.md` specifies **two coupled behavioral claims** about the SemanticLens pipeline over a trained vision model: (C1) a small set of top reference inputs strongly driving each component *c* is a *concept-faithful summary* of what *c* encodes, and (C2) embedding those reference inputs with a frozen CLIP image tower and pooling produces a single vector **v_c** placing *c* in CLIP's joint image–text semantic space. The unified proposal (`FINAL_PROPOSAL.md`) refines the *testing method* for these two claims — how to measure "concept-faithful", how to measure "placed in the semantic space" — while leaving the claims themselves untouched. The experiment plan (`EXPERIMENT_PLAN.md`) turns each claim into concrete milestones: build the reference-input pipeline and per-component v_c on ResNet-50 / ImageNet, then measure concept purity, text-search accuracy, sensitivity to reference-set size and pooling, and cross-model transfer under the swap models permitted by `task.md` (verify stage). The mechanism-strategy chain adopted is **Unit Interpretation → Decision Auditing** (the natural home for SemanticLens); Causal Intervention / Tuning / Formation Tracing are explicitly out of claim scope.

## Literature Landscape

See `LANDSCAPE.md` — 30 directly relevant works spanning three generations (fixed-vocabulary probes: Network Dissection / Compositional Explanations / INVERT; open-vocabulary captioning: MILAN / DnD / LLM-assisted Concept Discovery; frozen multi-modal encoder: CLIP-Dissect / FALCON / SpLiCE / Second-Order Lens / **SemanticLens** as the culminating unified framework). Five structural gaps (G1–G5) motivate the *measurable predicates* below but are not the claim itself.

## Resources (from `task.md`)

- **Inspected model (main experiment)**: ResNet-50 (ImageNet-pretrained) — one specific representative model. Cost-aware fidelity (not `resource_fidelity: strict` — this is the given + discovery combination, not reproduction).
- **Probe dataset (main experiment)**: ImageNet (train / val splits as available under `$DATA_DIR/ImageNet`).
- **Foundation encoder (all stages, frozen)**: CLIP image tower (OpenAI CLIP ViT-B/32 or ViT-B/16 as default — the exact CLIP checkpoint is a Phase 4.5 sub-decision; `task.md` names only "CLIP image tower").
- **Verify-stage swap models (available, not all mandatory)**: ViT-B/16 (ImageNet-pretrained), VGG-16 (ImageNet-pretrained), EfficientNet-B0 (ImageNet-pretrained). No datasets beyond ImageNet.
- **Compute**: 10 h GPU budget total (across the whole pipeline). GPUs allowed: `{1, 2, 3, 5, 6}` only.
- **Storage**: `DATA_DIR=/data/zhenqian/data`, `MODEL_DIR=/data/zhenqian/models`.
- **External review**: LLM API `<Your_api>` @ `https://www.dmxapi.cn/v1`, model `gpt-5.4` (bypass proxy).

## Claims to Verify

### Claim 1: Reference-input set as concept-faithful component summary

**Original (verbatim excerpt from `task.md`, Claim section, bullet 1):**
> For every component c in a trained vision model, a small set of reference inputs that strongly drive c is a concept-faithful summary of what c encodes.

**Extracted statement**: For each component *c* of ResNet-50 (across a specified set of layers / component granularities), a small set *R_c* of top-activating reference inputs sampled from ImageNet is a **concept-faithful summary** of what *c* encodes — i.e. the concept(s) recovered from *R_c* materially match the ground-truth concept-selectivity of *c*.

**Hypothesis (H1)**: Selecting the top-k inputs from ImageNet that maximally activate *c* yields a set *R_c* whose induced concept (as recovered by any faithful concept-extraction procedure — human labels for last-layer, CLIP text-similarity ranking, or LLM caption ranking for hidden layers) agrees with *c*'s ground-truth or crowd-elicited concept above chance and materially above random-input baselines.

**Measurable predicate**:
- **Last-layer (ImageNet class) sanity**: For each ResNet-50 penultimate/last-layer class-tied unit, concept-purity of *R_c* (fraction of *R_c* whose ground-truth label matches the class the unit corresponds to) exceeds a random-input baseline by ≥ Δ_pure, and matches the CLIP-Dissect published range for ResNet-50 (Oikarinen & Weng 2023).
- **Hidden-layer** (layer3, layer4 units): concept-purity, measured as top-1 CLIP-text-similarity agreement between the pooled CLIP-image embedding of *R_c* and its assigned label vs. a matched control label (the second-best label), exceeds chance by ≥ Δ_sep.
- **Sensitivity ablation**: as *k* varies over `{1, 4, 16, 64, 256}`, concept-purity is monotonically non-decreasing up to a plateau (an *R_c* of a small *k* is already a good summary — the "small set" clause of the claim).

**Expected direction**: **up** — concept-purity > chance and > random-input baseline; plateaus at small *k*.

**Resources (cost-aware)**: model: ResNet-50 (ImageNet-pretrained); dataset: ImageNet (val for reference-input selection; a held-out probe split for the concept score); frozen CLIP image tower for concept scoring on hidden-layer units; *used_n*: at least 1 k units sampled across layers (last layer plus a stratified sample of layer3 / layer4 conv-channels) with *k* ∈ {1, 4, 16, 64, 256}; concept-vocabulary: the ImageNet-1k class labels for last-layer + a broader open vocab (e.g. Broden / a curated word list) for hidden layers.

**Status**: pending verification

**Notes**: The claim states *concept-faithful summary* qualitatively; the predicate above operationalizes "concept-faithful" as **top-k-activation → concept-purity above baselines**, using two ground-truth conditions (class-tied last-layer, and hidden-layer via a matched-control CLIP-text-similarity test — the standard CLIP-Dissect / INVERT protocol). The "small set" clause is operationalized as a *k*-sensitivity ablation. No claim is added about *causal* effect on outputs (that would be a Causal-Intervention claim outside the given behavior — see mechanism-strategy note).

---

### Claim 2: Frozen CLIP pooling places component *c* in the joint image-text semantic space

**Original (verbatim excerpt from `task.md`, Claim section, bullet 2):**
> Embedding those reference inputs with a frozen multi-modal foundation model and pooling the embeddings yields a single vector v_c placing c in the foundation model's joint image–text semantic space.

**Extracted statement**: When the reference inputs *R_c* of claim C1 are embedded by the frozen CLIP image tower and the resulting embeddings are pooled (with a specified pooling operator) into a single vector **v_c**, that vector lives in CLIP's joint image-text embedding space in the *operative* sense: (a) it can be queried against CLIP text embeddings to retrieve components matching a natural-language concept, and (b) v_c is stable and semantically-consistent within the same component and separable across components of different concepts, so it functions as a *semantic locator* for *c* in the joint image-text space.

**Hypothesis (H2)**: The pooled v_c has three testable properties consistent with "placement in CLIP's joint image-text semantic space":
- **H2a — text queryability**: for a text query *t* embedded as CLIP text vector *e_t*, the ranking of components by cosine(v_c, e_t) is significantly better than chance at retrieving components whose ground-truth concept-selectivity matches *t*.
- **H2b — stability**: v_c computed from two independent reference sets (disjoint halves of the top-2k activating inputs) has cosine similarity above a stability threshold — otherwise "the vector for *c*" is not well-defined.
- **H2c — inter-component separation**: components with different concept identities have systematically lower v_c cosine than components with the same / similar concept identity — i.e. v_c geometry reflects concept identity, not just image statistics.

**Measurable predicate**:
- **H2a**: mean-reciprocal-rank (MRR) / recall@10 of text-query → component retrieval significantly above the chance-permutation baseline (bootstrap-CI, p < 0.05), and matching or exceeding the CLIP-Dissect text-lookup accuracy on the ImageNet-labelled last-layer units.
- **H2b**: median cosine(v_c^(A), v_c^(B)) ≥ τ_stable across a random component sample, where v_c^(A) and v_c^(B) are computed from disjoint halves of *R_c* (bootstrap CI reported).
- **H2c**: mean cosine(v_c1, v_c2) is lower for cross-class component pairs than for within-class pairs (t-test / effect-size reported); equivalently, k-NN in v_c-space groups same-concept components above chance.
- **Pooling ablation**: (mean, activation-weighted mean, max, medoid) produce qualitatively consistent conclusions on the three predicates above — establishes that the "placement in the joint image-text space" is *not* an artifact of any single pooling operator.

**Expected direction**: **up** — all three quantities significantly above chance / above a stability threshold; pooling-operator ablation robust.

**Resources (cost-aware)**: same inspected model / probe dataset as C1; frozen CLIP image tower is mandatory (part of the claim); text query set = ImageNet-1k class names (H2a on last-layer) + a curated broader vocab for hidden-layer H2c; *used_n*: same ≥ 1 k components as C1; per-component reference-set size *k* pinned to the plateau found in C1's ablation.

**Status**: pending verification

**Notes**: Claim C2 is *coupled* to C1 — C2 pre-supposes C1's "reference inputs summarize c". The Extracted statement decouples C2 into three operational properties (queryability, stability, separation) that together define what "placement in a joint image-text semantic space" *tests as* — no property is added that `task.md` does not clearly imply, and no property from `task.md` is dropped. The pooling operator is not fixed by `task.md`; it becomes a Phase 4.5 hyperparameter with an explicit ablation. The claim is **not** that v_c matches an *arbitrary human-written natural-language description* verbatim (that would be a Description-accuracy claim, more restrictive) — it is that v_c *lives in the joint space* in a queryable, stable, separating sense. This matches SemanticLens's own operational definition (Nature MI 2025: text search, comparison, labeling, audit).

## Consolidated Notes / Extraction Audit

- `task.md`'s Claim section contains **exactly two bullets**; both are captured verbatim above and split into two independent measurable claims. No paragraphs were merged; no claim was added.
- The claims are **coupled** but *individually verifiable* — C1 is a property of the reference set, C2 is a property of the pooled CLIP embedding of that set. The unified proposal (`FINAL_PROPOSAL.md`) exploits the coupling by sharing the reference-input pipeline across both.
- The word "concept-faithful" (C1) is operationalized as *concept-purity above baseline + monotone with k*; the phrase "places c in the joint image-text semantic space" (C2) is operationalized as *text-queryability + stability + inter-component separation + pooling robustness*. Both operationalizations use standard measures from the referenced literature (CLIP-Dissect, INVERT, SemanticLens) — no metric was invented.
- The "small set" wording in C1 is captured by the *k*-sensitivity ablation, not asserted as a specific number.
- The "trained vision model" and "foundation model" are pinned by `task.md`'s Resources section: ResNet-50 (main), swap set {ViT-B/16, VGG-16, EfficientNet-B0}; CLIP image tower (all stages).
- No M0 phenomenon-validation gate is emitted (`BEHAVIOR_SOURCE = given` → no M0).

## Refined Proposal

- Proposal: `refine-logs/FINAL_PROPOSAL.md` (unified testing approach covering both claims)
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md` (milestones tagged to C1 / C2)
- Tracker: `refine-logs/EXPERIMENT_TRACKER.md` (plan-level rows, `Status = pending`)

## Next Steps

- [ ] `/mechanism-skills` to route the testing approach to a concrete mechanism family + submethod (Workflow 1.25 — this is where the experiment stage picks the concrete Unit-Interpretation submethod)
- [ ] `/auto-experiment` to implement and run the verification suite (Workflow 1.5)
- [ ] `/auto-verify` to stress-test each verified claim under model swaps (Workflow 1.75 — ViT-B/16, VGG-16, EfficientNet-B0)
- [ ] `/auto-iteration-loop` to iterate the verification suite until reviewer-ready (Workflow 2)
- [ ] Or invoke `/auto` for the autonomous claim → routing → experiments → verify → review chain
