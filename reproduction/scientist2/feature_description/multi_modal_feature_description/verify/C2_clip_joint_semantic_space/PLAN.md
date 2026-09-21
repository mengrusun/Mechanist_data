## Claim C2: Pooled frozen-CLIP vector v_c places c in the joint image-text semantic space

### Main experiment
- Method: CLIP-Dissect (Multi-Modal family) — activation-ranked top-k reference inputs → frozen CLIP ViT-B/32 image embeddings → mean pool → v_c
- Dataset: ImageNet-val (50k images), 1000 fc + 500 layer4 + 500 layer3 components, k=16
- Model: ResNet-50 (inspected) + CLIP ViT-B/32 (frozen foundation encoder)
- Main metrics: P2a MRR=0.898, R@10=0.974 (perm95=0.0097); P2b stability=0.970; P2c layer4 gap=0.170, d=1.96

### Dimensions scope
Active: [method]
Excluded: [dataset] (not in DIMENSIONS), [model] (FORBIDDEN by user directive 2026-07-14)

### Variants

| # | Dimension | Swap (replaces) | Justification | Expected if claim holds | Expected if claim fails | Risk / confound control | Trust rank | Source |
|---|-----------|-----------------|---------------|-------------------------|-------------------------|-------------------------|------------|--------|
| 1 | method | CLIP-Dissect + Zennit-CRP CRP-crop compose (← plain CLIP-Dissect mean-pool) | CRP cropping uses the inspected model's relevance maps to crop each reference image to the component-relevant region before CLIP embedding. Tests whether the high MRR is robust when background/scene correlations are removed. MECHANISM_ROUTING.md explicitly flags this as the natural follow-up ablation. Reviewer rated trust rank 1. | MRR remains well above permutation baseline (0.0097); stability stays high; layer4 gap positive | MRR collapses toward permutation baseline, stability degrades, gap narrows — indicating the main result is driven by uncropped-image background regularities rather than component-localized semantics | Hold CLIP ViT-B/32 frozen; same top-k=16 reference inputs; same permutation test. Interpret any degradation as CRP-crop-induced CLIP distribution shift vs. genuine background confound removal. | 1 | skills/mechanism-skills/multi-modal/SKILL.md §"Compose the two"; MECHANISM_ROUTING.md §"Note on Zennit-CRP as a fallback" |

### Skipped Dimensions
- dataset: not in DIMENSIONS (user-specified scope)
- model: FORBIDDEN by user directive 2026-07-14 (no cross-model swaps in this run)

### Success Criterion (inherited from /auto-verify)
Each variant's claim_supported verdict is judged by /result-to-claim against the frozen main-experiment claim statement. consistent_with_main_experiment = pass iff the variant's conclusion agrees with the main experiment's conclusion (both supported or both not-supported).

---

## Candidate Pool

### Method candidates
| # | Name | Source | Notes |
|---|------|--------|-------|
| M1 | CLIP-Dissect + Zennit-CRP CRP-crop compose | mechanism-skills/multi-modal/SKILL.md | Canonical within-family compose: crop reference images to CRP heatmap bbox before CLIP embedding. Natural follow-up flagged in MECHANISM_ROUTING.md. |
| M2 | Activation-weighted mean pooling | Main experiment (already tested) | EXCLUDED — already run in main experiment; act-weighted-mean MRR=0.898 same as mean. Not a genuine swap. |
