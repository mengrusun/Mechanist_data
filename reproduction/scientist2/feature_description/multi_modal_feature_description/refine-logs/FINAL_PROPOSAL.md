# FINAL_PROPOSAL — Testing the SemanticLens Component → CLIP Semantic-Vector Claim on ResNet-50 / ImageNet

**Behavior-source**: given
**Mechanism**: discovery (experiment stage will route via `/mechanism-skills`)
**Date**: 2026-07-13
**Problem Anchor (FROZEN — from `task.md`)**:
- **C1** — For every component *c* in a trained vision model, a small set of reference inputs that strongly drive *c* is a concept-faithful summary of what *c* encodes.
- **C2** — Embedding those reference inputs with a frozen multi-modal foundation model (CLIP image tower) and pooling the embeddings yields a single vector v_c placing *c* in the foundation model's joint image–text semantic space.

Together these define the SemanticLens-style *component → semantic vector* behavior on ResNet-50 / ImageNet. The proposal **refines the testing method**, not the claims.

---

## Machine metadata (parsed by downstream stages)

```yaml
resource_fidelity: cost-aware          # NOT strict — given + discovery, not the reproduction combo
mechanism_strategy:
  directions: ["Unit Interpretation", "Decision Auditing"]
  rejected:
    - "Causal Intervention — the given behavior asserts encoding of c, not that ablating c changes model output; out of claim scope."
    - "Tuning & Editing — no downstream capability is being tuned; SemanticLens is diagnostic, not applied."
    - "Formation Tracing — inference-time claim; training-time origin out of scope and infeasible within 10 h GPU budget."
    - "Location (headline) — used only as a lightweight specificity screen for v_c (concept text-query -> highest-cosine components + matched-control text)."
  note: "SemanticLens is a cross-modal Unit-Interpretation method; v_c is the interpretable object, and Decision-Auditing (concept search, cross-model comparison, spurious-feature detection) is the downstream operation that both validates v_c's fidelity and demonstrates its use."
# chosen_mechanism intentionally NOT stamped — MECHANISM=discovery.
```

---

## 1. Method Thesis

A **single unified pipeline** that (i) for each ResNet-50 component *c* selects a small top-k reference-input set *R_c* from ImageNet by activation ranking, and (ii) computes a per-component semantic vector **v_c = pool(CLIP_image_tower(R_c))** in the frozen CLIP image–text space. Every claim-level predicate — concept-purity, text-queryability, stability, inter-component separation — is measured on **the same v_c**, so a single reference-input pass supports both C1 and C2 with only downstream cosine / retrieval operations added.

**Dominant contribution (testing side)**: a **rigorous, single-model, single-dataset** falsification suite for the SemanticLens component-vector claim, with (a) matched-control specificity tests (H2c, INVERT-style) that separate "the vector is queryable" from "the vector is only queryable at chance because CLIP text is well-spread", (b) a small-*k* plateau ablation that operationalizes the "small set" phrase in C1, and (c) a pooling-operator ablation that separates the claim from any single pooling artifact.

**Explicitly rejected complexity**:
- No causal ablation / activation patching (out of claim scope — the given claim is about *encoding*, not *causing model output*).
- No SAE training (a different Unit-Interpretation submethod; the given claim uses reference-input pooling, not dictionary decomposition — SAE could appear as a stronger baseline, not as the tested method).
- No auto-generated LLM captions in the main experiment (MILAN/DnD-style caption ranking is a comparison baseline only, avoided in the main path to keep the tested pipeline crisp).
- No cross-model transfer at *training time* (verify-stage swap of the inspected model only; the foundation encoder stays frozen throughout).
- No Formation Tracing / data attribution — inference-time claim.

---

## 2. Unified Testing Pipeline (shared across C1 and C2)

**Step A — Fix the frozen foundation encoder.** OpenAI CLIP **ViT-B/32** by default (smallest memory footprint; supported by torchvision hub / `open_clip`). Optionally swap to CLIP ViT-B/16 iff GPU memory permits; either is acceptable per `task.md`'s "CLIP image tower". *The same CLIP checkpoint is used at every stage across every inspected model — this is the "frozen foundation encoder" invariant.*

**Step B — Pick the component universe.**
- **Last layer (class-tied)**: ResNet-50 penultimate `avgpool → fc` — treated as one component per output class (1000 units). Ground truth is the ImageNet class label.
- **Hidden layers (stratified sample)**: ResNet-50 `layer3` (256 channels × ~1024 dims = 1024 channels) + `layer4` (2048 channels). We sample **500 channels per layer** stratified so early / mid / late conv positions are all covered (total ≈ 1000 hidden components + 1000 last-layer = ≈ 2000 components in the main run). Component = **channel** (spatial-mean-pooled over feature-map positions per input) — the standard SemanticLens / CLIP-Dissect granularity.

**Step C — Reference-input selection R_c.**
- Iterate ImageNet val (50 000 images; option to also use a 100 000-image train subsample if memory permits) through ResNet-50 once, cache the channel-mean activations. For each component *c* rank inputs by activation, keep the top-`k_max = 256` (superset — supports every *k* in the ablation without re-computing).
- Store all activations + top-256-image indices in one activations HDF5 for reproducibility.

**Step D — Pooled CLIP semantic vector v_c.**
- For each component's top-`k_max` reference inputs, run the frozen CLIP image tower once and cache the resulting per-image embeddings (this is *one* CLIP pass — v_c for any *k* ≤ *k_max* is derived by re-pooling from cache, and every pooling operator uses the same cached embeddings — this is what makes the whole test suite cheap enough for the 10 h budget).
- Default pooling: **L2-normalized mean of L2-normalized CLIP image embeddings** (the SemanticLens default). Alternative pooling operators for the ablation: activation-weighted mean, max, medoid.

**Step E — Concept scoring (shared measurement kernel).** For any component *c* with vector v_c and a concept vocabulary *V* (each concept encoded as a CLIP *text* vector), the score of concept *w* for *c* is cos(v_c, CLIP_text(w)). This kernel is reused by every predicate:
- **Last-layer C1 predicate** — the top-1 concept from *V* = ImageNet-1k class names should match the class *c* is tied to (concept-purity per component).
- **Hidden-layer C1 predicate** — top-1 concept from *V* = a broader open vocabulary should beat a **matched-control label** (the second-best concept), so we are measuring separation, not raw hit rate.
- **C2 predicates** — H2a: MRR/recall@10 of text-query → component retrieval; H2b: within-component stability via disjoint reference-set halves; H2c: within-vs-between cosine gap on same-concept vs. different-concept component pairs.

**Step F — Auditing use-case (Decision Auditing direction).** A single external-validity check that binds v_c to a *use*: on the last layer, we treat v_c as a "concept locator" and confirm that a **text query for a class name** retrieves the corresponding class-tied unit with high MRR — the SemanticLens "text search for neurons of concept X" capability. This is the *external-validity* half of C2 (H2a) *and* the operational definition of what SemanticLens promises.

---

## 3. Ablations that Falsify the Claim

Each of C1 and C2 has one ablation-driven falsification path:

| Ablation | Predicate that could refute | Grid |
|---|---|---|
| **k-sensitivity** (C1) | If concept-purity is flat with *k* or peaks only at very large *k*, then "*small* set is a faithful summary" fails | k ∈ {1, 4, 16, 64, 256} |
| **Pooling operator** (C2) | If mean vs. max vs. medoid give *qualitatively* different C2 conclusions, then v_c is a pooling artifact, not a semantic vector | pool ∈ {mean, act-weighted-mean, max, medoid} |
| **Layer granularity** (C1 + C2) | If only last-layer works and hidden-layer components fail every predicate, the claim is a class-label-echo claim, not a component-encoding claim | layers ∈ {last, layer4, layer3} |
| **Cross-model transfer** (C1 + C2) | If v_c geometry does not transfer across ResNet-50 ↔ ViT-B/16 (i.e., text-queryability collapses), the "joint image-text semantic space" is not a *shared* coordinate system for other CNNs | verify-stage only; main experiment stays on ResNet-50 |

---

## 4. Baselines (comparison-only, do not replace the tested method)

Two lightweight baselines that share Step C's cached reference inputs (no extra ResNet forward passes):

- **Random-input baseline (for C1)**: same pipeline but *R_c* = k random ImageNet inputs (not activation-ranked). Establishes the Δ_pure gap.
- **Permutation baseline (for C2)**: shuffle the component ↔ concept assignment and recompute MRR/recall@10 → chance floor with bootstrap CI.

Additional caption-based baseline (MILAN-style) is **not** in the main path — it would need a separate LLM captioning pipeline that neither `task.md` nor the compute budget accommodates. If time remains at the end, one small (100-component) MILAN-flavored side comparison is optional.

---

## 5. Resource Plan (10 h GPU budget)

- **ResNet-50 forward on ImageNet-val (50 k images)**: ≈ 3-5 min on one GPU (`{1,2,3,5,6}`).
- **CLIP ViT-B/32 forward on the top-256-image union across ~2000 components**: worst case ≈ 500 k images (if no image is shared across components — in practice much fewer since top-k inputs share); ≈ 20-40 GPU-min on one GPU with FP16.
- **All predicate computations** are cosine similarities on cached embeddings: CPU-friendly.
- **Cross-model verify-stage** (ViT-B/16, VGG-16, EfficientNet-B0) repeats Step C + D each — ≈ 30 GPU-min per swap model → ≈ 1.5 GPU-h total for all three swaps.
- **Safety headroom**: ≤ 4 GPU-h for the whole claim-stage-driven main experiment; leaves ≥ 6 GPU-h for `/auto-verify` variants and iteration.

Every command sets `CUDA_VISIBLE_DEVICES` to a subset of `{1,2,3,5,6}`. No downscaling of ResNet-50 or ImageNet.

---

## 6. Deliverables

Every predicate produces a single row in the final results table:

| Predicate | Metric | Layer / Setting | Expected | Refute if |
|---|---|---|---|---|
| P1a (C1) | last-layer top-1 concept-purity | last, k=16 | ≥ CLIP-Dissect published range for ResNet-50 last-layer | < random baseline + Δ_pure |
| P1b (C1) | hidden top-1 vs. matched-control cosine gap | layer4, k=16 | Δ_sep > 0 with p < 0.05 (paired test) | Δ_sep ≤ 0 |
| P1c (C1) | *k*-sensitivity monotonicity | any layer | monotone-nondecreasing, plateau at *k*_plateau ≤ 16 | non-monotone or plateau only at k = 256 |
| P2a (C2 / H2a) | text-query → component MRR / R@10 | last-layer, 1000 class queries | significantly above permutation baseline | ≤ permutation CI upper bound |
| P2b (C2 / H2b) | within-component stability cosine (disjoint halves) | median across sampled components | ≥ τ_stable (recorded, e.g. ≥ 0.5) | median below stability threshold |
| P2c (C2 / H2c) | within-vs-between-concept cosine gap | last + layer4 | positive gap, Cohen's *d* ≥ 0.5 | non-positive gap |
| P2d (C2 pooling) | qualitative agreement across {mean, act-wt-mean, max, medoid} | last + layer4 | same sign on P2a-c | different sign on any predicate |
| P3 (cross-model, verify stage) | text-query MRR on ViT-B/16 / VGG-16 / EffNet-B0 | last-layer | above permutation | at permutation floor |

## 7. Risks and Mitigations

- **Risk R1: Reference-input concentration collapse.** If a small set of ImageNet images ends up in *R_c* for many components, v_c becomes correlated across components regardless of concept. *Mitigation*: log per-image cover fraction; if > 5 % of *R_c* sets share the same image, add a diversity penalty in Step C (frequency-inverse re-weighting) as a documented ablation, not a silent change.
- **Risk R2: CLIP tokenization / prompting sensitivity for text queries.** *Mitigation*: use the standard OpenAI CLIP prompt-ensemble ("a photo of a {label}") averaged across the 7-template ensemble — matches CLIP-Dissect / SemanticLens standard.
- **Risk R3: Component granularity mismatch.** Late-layer conv channels of ResNet-50 (layer4, 2048 channels) may be too fine-grained. *Mitigation*: report per-layer results and flag any layer where none of P1a-c passes; this is a finding, not a failure of the claim as a whole.
- **Risk R4: Frozen CLIP checkpoint fragility.** *Mitigation*: pin CLIP to a specific published checkpoint (`ViT-B-32` weights from `open_clip` `openai` tag), log SHA-256 of the weights file.

## 8. What this proposal is NOT

- Not a mechanism-*causation* claim — no ablation / patching / steering on ResNet-50 (out of scope for the given claim; would be a different claim under Causal Intervention).
- Not a method-improvement paper — no new pooling operator or new selection strategy is being *proposed*; existing operators are *ablated* to test robustness.
- Not a benchmark paper across many CNN families — verify-stage cross-model transfer is a robustness check, not a benchmark competition.
- Not an SAE / dictionary paper — SAEs are a different Unit-Interpretation submethod outside the given claim's pipeline.

## 9. Final Verdict

**READY** — The tested claim is anchored, the pipeline is a single unified pass (Steps A-F) with three fully-cached-reuse ablations, every predicate has a numeric decision rule with a matched control, and the compute plan fits inside 10 GPU-h. Downstream Phase 1.5 in `/auto-experiment` will bind the concrete `sites`, `metric`, and `n_pairs` (see `method_sensitive` fields in `EXPERIMENT_PLAN.md`).
