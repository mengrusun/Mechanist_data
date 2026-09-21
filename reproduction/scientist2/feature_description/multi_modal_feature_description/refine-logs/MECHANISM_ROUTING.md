# Mechanism Routing

<!-- Metadata block (parsed by /auto orchestrator resume check). -->
committed: true
chosen_family: Multi-Modal / CLIP-Dissect
chosen_idea_title: SemanticLens Component → CLIP Semantic-Vector Verification on ResNet-50 / ImageNet
effective_domain: mechanistic-interpretability
candidate_paths:
  - skills/mechanism-skills/multi-modal/clip-dissect/SKILL.md
  - skills/mechanism-skills/multi-modal/zennit-crp/SKILL.md

## Candidates

1. **[recommended]** Multi-Modal / CLIP-Dissect — Exact match to SemanticLens's construction: per-component top-k activation-driven reference-input selection, frozen CLIP image-tower embedding of R_c, activation-weighted / mean pooling → per-component semantic vector v_c, cosine similarity vs CLIP text embeddings for concept scoring. The family SKILL explicitly names SemanticLens-style pipelines under its "select together" trigger, and CLIP-Dissect alone is the correct pick when per-(component, image) CRP backward passes would be prohibitive (2000 components × 256 top-k images ≈ 500k backward passes) — which is the case here at the 10 h GPU budget. CLIP-Dissect's pooling operator (activation-weighted mean of L2-normalized CLIP image embeddings) is one of the four pooling operators in the M4 ablation, and the L2-normalized-mean is the plan's headline pool.
   - path: skills/mechanism-skills/multi-modal/clip-dissect/SKILL.md
2. Multi-Modal / (CLIP-Dissect + Zennit-CRP compose) — The full canonical pipeline: pre-crop each reference image to the concept-conditional CRP heatmap's high-relevance bbox before CLIP embedding. Produces measurably cleaner v_c (family SKILL §"Compose the two") but adds a per-(component, image) LRP backward pass. Out of scope for the main experiment (cost) but recommended as the natural follow-up ablation if any C1/C2 predicate returns weak (Δ_sep close to 0, MRR barely above permutation) — CRP cropping would then be the first thing to try before pivoting.
   - path: skills/mechanism-skills/multi-modal/zennit-crp/SKILL.md
3. Multi-Modal / Zennit-CRP alone — Rejected. CRP alone produces concept-conditional heatmaps for individual predictions and does not embed reference inputs into foundation-model semantic space. The tested claim explicitly requires the CLIP-space vector v_c, which is the CLIP-Dissect side of the compose.
   - path: skills/mechanism-skills/multi-modal/zennit-crp/SKILL.md

## Composition plan

**Screen → Decode → Verify (all within the CLIP-Dissect submethod for the main experiment):**

1. **Cache (screen):** M1 → M2 — one ResNet-50 forward pass on ImageNet-val, per-component top-k=256 activation-driven reference inputs. Cheap; O(50k images × 1 forward pass).
2. **Embed (decode):** M3 → M4 — one frozen CLIP ViT-B/32 forward pass over the *union* of all reference inputs; pool into v_c across k ∈ {1, 4, 16, 64, 256} × pool ∈ {mean, act_weighted_mean, max, medoid}. CLIP embeddings are cached once and re-pooled without re-running CLIP; this is what makes the 20-cell grid fit in the budget.
3. **Score (verify — no ResNet-50 intervention):** M6–M10 — cosine-similarity kernel over the cached v_c against CLIP text embeddings (M5) or against each other:
   - **C1 predicates:** M6 last-layer concept-purity vs random-input baseline (P1a); M7 hidden-layer matched-control gap + k-plateau (P1b/P1c).
   - **C2 predicates:** M8 text-query MRR/R@10 vs permutation baseline (P2a); M9 disjoint-half stability (P2b); M10 within-vs-between-concept gap (P2c); pool-operator ablation across M8/M9/M10 (P2d).
4. **Aggregate & recover:** M11 layer-granularity summary; M12 cross-model transfer on {ViT-B/16, VGG-16, EfficientNet-B0} (P3); M13 final results table.

**Note on Zennit-CRP as a fallback:** if M7 hidden-layer separation is weak (Δ_sep close to 0 at pool=mean), the family SKILL guidance says to try CLIP-Dissect + CRP-crop compose before pivoting. This is the FIRST thing the iteration loop should propose if C1 hidden-layer collapses.

**Cost notes:** Total est. 3.2 GPU-h across all 13 milestones; ≥ 6 GPU-h of headroom for `/auto-verify` variants. The dominant cost is M3 (CLIP forward over the reference-set union, ≈ 0.7 GPU-h worst case) — everything else is I/O-bound cosine kernels or small ResNet/CLIP passes.

## Plan reconciliation

<!-- One row per method_sensitive field declared on the intervention milestones. -->
- pool (M4): plan={mean, act_weighted_mean, max, medoid} → matches — CLIP-Dissect's canonical pool is activation-weighted mean of L2-normalized image embeddings; the plan's headline `mean` (L2-normalized) is the SemanticLens default; both are already in the 4-cell grid.
- sites (M4): plan={layer3, layer4, avgpool, fc} → matches — CLIP-Dissect works on any spatially-pooled channel output; the plan's four sites are exactly the standard CLIP-Dissect ResNet-50 layer set (from clip-dissect demo `run_clip_dissect.py` and paper Table 1).
- metric (M6, M7, M8, M9, M10): plan=cosine-similarity + paired t-test / permutation-baseline CI / Cohen's d → matches — CLIP-Dissect uses cosine similarity as its scoring kernel; the statistical wrappers (paired t-test, permutation, bootstrap CI) are downstream normalizations orthogonal to the submethod choice.
- n_pairs (M10): plan=10000 random cross-concept pairs → matches — this is a pure statistical-power sizing knob on the downstream cosine kernel, independent of the CLIP-Dissect submethod. Sufficient for Cohen's d ≥ 0.5 detection at p < 0.05.
- gpu_hours (aggregate): plan ~3.2 GPU-h → matches — no revision needed. CLIP-Dissect's cost is dominated by the ONE CLIP forward pass over the reference-set union (M3, ~0.7 GPU-h) plus one ResNet-50 pass (M1, ~0.2 GPU-h); everything downstream is CPU-friendly cosine kernels.

reconciliation_status: ok

## Rationale

**Why CLIP-Dissect is the recommended #1:** SemanticLens IS CLIP-Dissect — the family SKILL explicitly lists "reproduce SemanticLens-style pipelines" as a compose-both trigger, and the tested claim per FINAL_PROPOSAL §2 (Steps C–E: activation-ranked top-k → CLIP image tower → pool into v_c → cosine vs CLIP text) is the CLIP-Dissect pipeline verbatim, with only the pooling operator and k as ablation axes. The plan's four-cell pool ablation and five-cell k ablation are exactly the natural ablation axes of CLIP-Dissect; the C1/C2 predicates (concept-purity, matched-control separation, text-query MRR, within-component stability, within-vs-between gap) are the concept-labelling-quality metrics CLIP-Dissect's paper (Oikarinen & Weng 2023, ICLR Spotlight) evaluates against NetDissect and MILAN.

**Why not compose CRP+CLIP-Dissect in the main experiment:** the compose adds a per-(component, image) LRP backward pass. At 2000 components × 256 top-k images × ~50 ms per backward pass on ResNet-50, this is ~7 GPU-h alone — would consume the entire budget and leave nothing for `/auto-verify`. The family SKILL explicitly recognizes this case ("very large concept-vocabulary scans where the cost of per-(component, image) CRP backward passes is prohibitive") and endorses CLIP-Dissect alone. If a weak result at C1-hidden triggers iteration, the CRP-crop compose is queued as the first back-edge fix.

**Cross-round avoid-set:** none (`families_already_settled: []` for this round).

**Aligned with plan's `directions: [Unit Interpretation, Decision Auditing]`:** yes. Multi-Modal / CLIP-Dissect is exactly Unit Interpretation (labels internal units with named concepts via cross-modal alignment); the M8 text-query → component retrieval predicate is the Decision-Auditing operationalization (SemanticLens's "text search for neurons of concept X") — same v_c serving both directions, as declared in the plan's `note`.
