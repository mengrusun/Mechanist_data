# Experiment Audit Report — Claim C2

**Date**: 2026-07-14
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP — gpt-5.4)
**Project**: SemanticLens Component → CLIP Semantic-Vector Verification on ResNet-50 / ImageNet
**Claim**: C2 — Pooling frozen CLIP image embeddings of the top-k activation-driven reference inputs yields a per-component vector v_c that places c in the joint image-text semantic space: text-queryable (H2a/P2a: MRR/R@10 above permutation), stable across disjoint halves (H2b/P2b: cosine ≥ τ_stable), separable within- vs. between-concept (H2c/P2c: Cohen's d ≥ 0.5), and qualitatively invariant across pooling operators (H2d/P2d).
**Linked milestones**: M8_C2_queryability (P2a), M9_C2_stability (P2b), M10_C2_separation (P2c/P2d)

## Overall Verdict: WARN
*This is C2's integrity verdict — whether C2's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
- **Evidence**: P2a GT: fc unit class_label = ImageNet class index by construction. `m8_c2_queryability.py` identifies GT component by `class_labels == c` — architecture-derived, not model output. P2b (stability): no GT by design — measures internal consistency of disjoint-half pooled vectors. P2c (separation): fc same-concept groups built from CLIP text embedding clusters (external proxy); layer4 same-concept groups built from top-1 concept identity via broden vocab (external proxy). Both proxies are disclosed.
- **Details**: No model-output-as-GT anywhere. P2c's CLIP-text-neighbor grouping is a heuristic proxy but is explicitly disclosed and its threshold sensitivity is logged in the output JSON.

### B. Score Normalization: PASS
- **Evidence**: All vectors L2-normalized for cosine similarity. Permutation baseline for P2a is an independent random shuffle of class labels (not model-derived). No metric divided by model's own prediction max/mean.
- **Details**: Clean. The MRR = 0.898 vs. permutation 95th percentile 0.0097 is an enormous margin, not a self-referential artifact.

### C. Result File Existence: PASS
- **Evidence**: All C2-linked result files exist and match tracker:
  - `runs/M8_C2_queryability/mrr__k16__mean.json`: MRR=0.8983, R@10=0.974, perm95=0.0097, significant=true ← matches tracker row 36.
  - `runs/M9_C2_stability/stability__hk16__mean.json`: median_cosine=0.9703, passes=true ← matches tracker row 40.
  - `runs/M10_C2_separation/sep__k16__mean.json`: layer4 gap=0.1699, d=1.9585, p=0, passes=true; fc gap=-0.001, passes=false ← matches tracker row 44.
  - All other C2 tracker rows (37-43, 45-47) marked done.
- **Details**: No phantom results. Numbers match between files and tracker.

### D. Dead Code Detection: PASS
- **Evidence**: All metric functions called in m8, m9, m10: `compute_ranks`, `mrr_from_ranks`, `recall_at_k`, `permutation_mrr_ci_upper` all called in m8 main(). `pool` function called in m9 main loop. `cohens_d`, `build_fc_same_concept_groups`, `within_between_cosines` all called in m10 main(). Threshold sensitivity block runs and writes to output.
- **Details**: No dead code detected. All defined functions contribute to the output JSON.

### E. Scope Assessment: WARN
- **Evidence**: C2 claims "places c in the foundation model's joint image-text semantic space" as a general property. Evidence: 1000 fc + 500 layer4 components on ResNet-50 only. H2c (P2c) mixed: fc gap = -0.001 (fails under CLIP-text-neighbor grouping heuristic, disclosed); layer4 gap = 0.170 d=1.96 (passes strongly). H2d (P2d) mixed: sign-consistent on P2a, P2b, layer4-P2c but NOT on fc-P2c (disclosed). P3 cross-model SKIPPED by user directive (disclosed).
- **Details**: The qualifier in the main-experiment verdict ("fc P2c near-zero — grouping-heuristic artifact") is an honest disclosure. The WARN is for scope: H2d is partially falsified on fc, and "joint semantic space" universality is only verified on ResNet-50. Not a FAIL because the caveats are documented and the layer4 evidence for H2c is strong.

### F. Evaluation Type: PASS
- P2a: `real_gt` (architecture-defined class labels for fc component identity)
- P2b: `self_supervised_proxy` (internal consistency metric, no external GT by design)
- P2c/P2d: `synthetic_proxy` (CLIP-text-neighbor clusters for fc; top-1 concept identity for layer4)

## Action Items
- Clarify H2c claim: state that fc P2c uses a heuristic same-concept grouping whose validity is threshold-sensitive and that the layer4 result is the primary evidence for concept separability.
- Note H2d: clarify that "qualitative invariance across pooling operators" holds on P2a/P2b/layer4-P2c but not fc-P2c; this is already disclosed but the claim header should reflect it.
- These are scope-language issues, not experimental failures.
