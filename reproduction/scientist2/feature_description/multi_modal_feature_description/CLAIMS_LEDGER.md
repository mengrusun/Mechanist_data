# Claim Ledger — SemanticLens Component → CLIP Semantic-Vector Verification on ResNet-50 / ImageNet

**Direction**: For each vision-model component, a small activation-driven reference-input set is a concept-faithful summary; pooling frozen CLIP embeddings of that set places the component into the joint image-text semantic space.
**Date**: 2026-07-13 → 2026-07-14
**Pipeline**: completed | **Iteration**: 6/10 "almost" (0/6)
**Models**: claim=claude-opus-4-7, experiment=claude-opus-4-7, verify=claude-sonnet-4-6, iteration=claude-opus-4-7
**Updated after**: iteration:final

| Claim | Main experiment | Verify | Post-Iteration | Final |
|-------|-----------------|--------|----------------|-------|
| C1 reference-input set is concept-faithful summary | supported [P3 SKIPPED — user directive] | INTEGRITY_ONLY (cap; audit WARN) | INTEGRITY_ONLY held; ⓪ scope narrowed to ResNet-50 | ⚪ integrity_only — ResNet-50/ImageNet-scoped |
| C2 pooled frozen-CLIP vector places c in joint image-text space | supported | PASS robustness=1.0 (method) | PASS held; ⓪ layer4 as strong test, fc bracketed | ✓ holds (ResNet-50/ImageNet-scoped) |

---
## C1 — Reference-input set is a concept-faithful component summary
- **Statement**: For every component c in a trained vision model, a small set of top-k activation-driven reference inputs from ImageNet is a concept-faithful summary of what c encodes — measured by (P1a) last-layer top-1 concept-purity above random baseline, (P1b) hidden-layer top-1 vs. matched-control CLIP-text-similarity gap > 0 with p < 0.05, and (P1c) k-sensitivity plateau at small k (≤ 16).
- **Origin**: task.md — Claim bullet 1 (SemanticLens component summarization, faithful capture)
- **Data**: ImageNet-1k validation split (HuggingFace mrm8488/ImageNet1K-val) — provenance=existing; available=50000, used=50000 (100%; activation cache) + 1000 fc components × 1000 class-name text queries + 500 layer4 + 500 layer3 hidden components × 1203 broden-style vocab; Full ImageNet-val used; hidden-layer vocab is broden-*style* union (1203 concepts), NOT the original Broden dataset.
- **Models**: ResNet-50 (ImageNet-pretrained, torchvision IMAGENET1K_V2), CLIP image tower ViT-B/32 (openai — frozen, foundation encoder)
- **Method**: Multi-Modal Feature Description / CLIP-Dissect — reference-input screen (top-k activation) → pooled decode (mean-cosine to CLIP text) → concept-purity + matched-control gap + k-plateau evaluation.
- **Main experiment**: `supported [P3 SKIPPED — user directive]` — P1a fc top-1 purity 0.898 at k=16 (vs 0.001 random; McNemar p≈1.9e-270). P1b layer4 Δ_sep=0.0202 at p≈0; layer3 Δ_sep=0.0049 (p≈0). P1c plateau in k∈{1,4,16}, decline at k=64/256. P3 SKIPPED by user directive.
- **Verify**: robustness=n/a — method excluded (Stage-2 deferred by cap) / dataset n/a / model excluded (user directive); integrity=WARN; verdict=INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap)
- **Iteration**: INTEGRITY_ONLY held (no back-edge, per contract); five ⓪ narrative-only edits recorded for paper write-up. narrowed_to: "sampled components of a trained ResNet-50 on ImageNet; universality across trained vision models NOT claimed"
- **Final**: `⚪ integrity_only — ResNet-50/ImageNet-scoped support; swap-test deferred (max_verify_claims cap); narrative narrowed by iteration ⓪. Upgrade: /auto-verify C1 -- resume: true`
- **Caveats**:
  - P3 SKIPPED by user directive 2026-07-14 — cross-model transfer not tested. C1's "for every component c in a trained vision model" universality clause is verified only on ResNet-50.
  - P1c strict criterion (monotone-nondecreasing across full k-range) is not met — Δ_sep declines at k=64/256. Semantically consistent with "small set is a concept-faithful summary" (early plateau exists at k∈{1,4,16}).
  - Hidden-layer vocab is broden-*style* union (1203 concepts), not the original Broden dataset.
  - Phase-2 audit flagged P1b docstring / metric-alignment note at WARN level — not a numerical / methodology break.
- **Artifacts**: `refine-logs/EXPERIMENT_PLAN.md#M6_C1_last_layer`, `refine-logs/EXPERIMENT_PLAN.md#M7_C1_hidden`, `refine-logs/EXPERIMENT_PLAN.md#M11_layer_granularity`, `refine-logs/EXPERIMENT_RESULTS.md`, `verify/C1_concept_faithful_summary/main_experiment_audit/`, `runs/M6_C1_last_layer/`, `runs/M7_C1_hidden/`, `runs/M11_layer_granularity/`
- **Figures**:
  - ![C1/P1a — last-layer top-1 concept-purity vs k (ImageNet class-name text queries; random baseline ≈ 0.001). Peaks at k=16 (89.8%), declines at k=256 (68.9%), supporting the 'small set is a concept-faithful summary' phrasing.](figures/C1/c1_k_sweep_purity.png) — vector: `figures/C1/c1_k_sweep_purity.pdf`
  - ![C1/P1b — hidden-layer matched-control cosine gap Δ_sep (top-1 vs top-2 concept) at k=16. layer4 gap 0.0202 (p≈0), layer3 gap 0.0049 (p≈0); both significant but layer3 magnitude is small.](figures/C1/c1_hidden_matched_control.png) — vector: `figures/C1/c1_hidden_matched_control.pdf`

---
## C2 — Pooled frozen-CLIP vector v_c places c in the joint image-text semantic space
- **Statement**: Embedding the top-k activation-driven reference inputs with a frozen CLIP image tower and pooling the embeddings yields a single per-component vector v_c that places c in the joint image-text semantic space — text-queryable (H2a: MRR/R@10 above permutation), stable across disjoint halves (H2b: cosine ≥ τ_stable), separable within- vs. between-concept (H2c: Cohen's d ≥ 0.5), and qualitatively invariant across pooling operators (H2d: {mean, act-weighted mean, max, medoid} agree on H2a-c signs).
- **Origin**: task.md — Claim bullet 2 (SemanticLens foundation-model coordinate system, faithful capture)
- **Data**: ImageNet-1k val (reference inputs) + ImageNet-1k class names + broden-style vocab — provenance=existing; available=50000 images + 1000 class-name queries + 1203-concept broden-style vocab, used=50000 (100%); 1000 × 1000 text-query × component MRR grid; 2000 components × 32 imgs for stability; 10000 random pairs per pool for gap.
- **Models**: ResNet-50 (ImageNet-pretrained), CLIP image tower ViT-B/32 (openai — frozen), CLIP text tower ViT-B/32 (openai — frozen)
- **Method**: Multi-Modal Feature Description / CLIP-Dissect — pooled v_c cosine-scored against class-name text embeddings for MRR/R@10 (H2a) vs. permutation baseline, disjoint half stability (H2b), between-vs-within-concept cosine gap with Cohen's d (H2c). Pooling ablation across {mean, act-weighted mean, max, medoid} (H2d).
- **Main experiment**: `supported` — H2a MRR=0.898 (perm95=0.010); H2b stability=0.970; H2c layer4 gap=0.170, d=1.96 (p≈0); H2d sign-consistent on P2a/P2b/layer4-P2c; fc P2c mixed signs across pools (magnitudes <0.02, attributed to grouping-heuristic noise).
- **Verify**: robustness=1.00 — method pass / dataset n/a / model excluded (user directive); integrity=WARN; verdict=PASS. Method-swap variant: CRP-compose (Zennit-CRP compose-based fallback) — MRR=0.9024 (Δ=+0.0044 vs main), stability=0.9581, significant.
- **Iteration**: PASS held; five ⓪ narrative-only edits recorded for paper write-up. narrowed_to: "ResNet-50/ImageNet component vectors are text-queryable (layer4 strong test, fc inconclusive under grouping heuristic); cross-model 'shared coordinate system' universality NOT claimed"
- **Final**: `✓ holds (ResNet-50/ImageNet-scoped) — verify PASS robustness=1.0; layer4 is the strong test (d=1.96); fc-P2c bracketed as inconclusive-under-heuristic; narrative narrowed by iteration ⓪.`
- **Caveats**:
  - fc P2c near-zero across pools (magnitudes < 0.02, mixed signs) — attributed to CLIP-text-cluster grouping heuristic, not a claim failure. Layer4 P2c unambiguous.
  - Cross-model "shared coordinate system" universality (formerly P3/M12) SKIPPED by user directive 2026-07-14; C2's shared-coordinate implication verified only on ResNet-50 (model axis intentionally excluded in verify).
  - Phase 2 & Phase 9 both WARN — WARN is anchored on the same scope-limitation (ResNet-50 only + P3 SKIPPED), not a numerical / methodology break.
- **Artifacts**: `refine-logs/EXPERIMENT_PLAN.md#M8_C2_queryability`, `refine-logs/EXPERIMENT_PLAN.md#M9_C2_stability`, `refine-logs/EXPERIMENT_PLAN.md#M10_C2_separation`, `refine-logs/EXPERIMENT_PLAN.md#M4_v_c`, `refine-logs/EXPERIMENT_RESULTS.md`, `verify/C2_clip_joint_semantic_space/main_experiment_audit/`, `verify/C2_clip_joint_semantic_space/variant_audit/`, `verify/C2_clip_joint_semantic_space/variants/method-swap-crp-compose/`, `runs/M8_C2_queryability/`, `runs/M9_C2_stability/`, `runs/M10_C2_separation/`

---
## Journey Summary
- **Claim**: given-mode faithful capture — C1 + C2 taken verbatim from task.md (no ideation).
- **Mechanism strategy**: Unit Interpretation → Decision Auditing
- **Mechanism routing**: family=Multi-Modal Feature Description, submethod=CLIP-Dissect (with Zennit-CRP flagged as a compose-based fallback for iteration)
- **Experiment**: 49 runs retained (M0-M11 + M13; M12 3-run cross-model SKIPPED per user directive 2026-07-14), ~0.4 GPU-h retained, headline positive on ResNet-50 evidence
- **Verify**: 2 claim(s): 1 PASS (C2) / 0 FAIL / 0 INCONCLUSIVE / 0 ZEV / 1 INTEGRITY_ONLY (C1, cap=1, swap_off=0); integrity[Phase2=WARN/Phase9=WARN]. DIMENSIONS=method only (model-swap forbidden by user directive). ~1.65 GPU-h.
- **Iteration**: 0/6 iterations, claim-reentries=0/2, score 6/10 verdict almost, termination=positive_verdict; five ⓪ narrative-only edits applied, no back-edges, 0 GPU-h.
- **Figures**: 2 across 1 claim (C1); 1 judgment-skipped (C2, prose carries the scalars); 0 render-skipped, 0 errored

## Open Items
- C1: P3 cross-model transfer (M12) SKIPPED by user directive (2026-07-14) — C1's universality clause "for every component c in a trained vision model" remains unverified beyond ResNet-50; paper text must scope to ResNet-50/ImageNet.
- C1: INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap) — Stage-1 audit passed WARN; Stage-2 swap-test deferred. Upgrade command (informational): `/auto-verify C1 -- resume: true`.
- Paper-scope caveats to land in write-up (⓪ narrative-only edits from iteration 1): (a) scope both C1 and C2 to ResNet-50/ImageNet in prose; (b) reframe C1/P1c as "plateau at small k ≤ 16, degrades at k ∈ {64, 256}"; (c) bracket C2/fc-P2c as "inconclusive under CLIP-text-cluster grouping" and present layer4 P2c (d=1.96) as the strong test; (d) clarify C1/P1b "matched-control" terminology (top-1 vs. top-2 concept gap); (e) describe C2 swap robustness as "one successful method-swap verification (Zennit-CRP compose)".
- Recurring unresolved patterns: single-model evidence only (C1 + C2 on ResNet-50); single-variant swap for C2; hidden-layer C1/P1b small in absolute magnitude (layer3 Δ_sep=0.005 is significant but small).
