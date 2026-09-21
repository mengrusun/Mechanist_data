# Verify Plan — C2a

## Claim C2a: Aligned DINOv2 ViT-B Improves Aggregate Spearman

**Full statement**: Alignment finetune of DINOv2 ViT-B against the THINGS-fit teacher increases aggregate Spearman correlation with human THINGS-triplet similarity by Δρ ≥ 0.05 over the unaligned DINOv2 ViT-B baseline, with paired-bootstrap significance at α = 0.05.

**Main-experiment verdict**: supported (Δρ = +0.366, M3=0.555 vs M4=0.189; 7× threshold; non-overlapping CI95)

**Phase 2 integrity**: PASS (no caveats)

### Main experiment
- Method: Full-backbone KD finetune (triplet_kl loss, α=1.0, lr=5e-5, batch=64, 1 epoch, seed=42)
- Dataset: ImageNet-val 40k (finetune) + THINGS held-out 15,640 triplets (eval)
- Model (student): DINOv2 ViT-B (`facebook/dinov2-base`, 86M params, hidden=768)
- Teacher (fixed): SigLIP-So400m + M1 THINGS-aligned head (frozen — NEVER swapped per task.md)
- Metric: spearman_aggregate = 0.555 (aligned) vs 0.189 (unaligned); Δρ = +0.366

### Dimensions scope
- **Active**: model (DIMENSIONS=model; 1 variant)
- **Excluded**: method, dataset (per /auto-verify args)

### Variants

| # | Dimension | Swap (replaces) | Justification | Expected if claim holds | Expected if claim fails | Risk / confound control | Trust rank | Source |
|---|-----------|-----------------|---------------|-------------------------|-------------------------|-------------------------|------------|--------|
| 1 | model | DINOv2 ViT-S, `facebook/dinov2-small`, 21M params, hidden=384 (← DINOv2 ViT-B, 86M params) | Same self-supervised training family (DINOv2), 4× smaller; tests whether the alignment improvement generalizes across student model scales. If the claim is fundamentally about the KD alignment objective + teacher quality rather than the specific ViT-B capacity, the smaller student should also show Δρ ≥ 0.05. Chosen as the cheapest variant (SANITY_FIRST=true; faster finetune; less GPU memory) and is explicitly listed in task.md's verify candidate pool as "Supervised ViT-S" equivalent. | spearman_aggregate(aligned ViT-S) − spearman_aggregate(unaligned ViT-S) ≥ 0.05 with paired-bootstrap significance. ViT-S aligned vs ViT-S unaligned in the predicted positive direction. | Δρ ≪ 0.05 or negative for ViT-S; would indicate the alignment gain is specific to ViT-B's capacity or attention heads and does not transfer across student scales — claim is fragile under model-scale swap. | (1) Use the M1 teacher head unchanged. (2) Run the unaligned ViT-S baseline (same checkpoint as the finetune start) to compute the delta. (3) Match all hyperparameters to main experiment except batch_size which may need to adjust for ViT-S smaller hidden dim: batch_size=64 still fine; lr=5e-5 same. (4) Seed=42. (5) Same 40k ImageNet-val finetune subset + same THINGS held-out eval split. | 1 | task.md verify_variants.student_swaps + /data/zhenqian/models/dinov2-small (on disk) |

### Skipped Dimensions
- **method**: excluded by DIMENSIONS=model
- **dataset**: excluded by DIMENSIONS=model

### Success Criterion (inherited from /auto-verify)
Each variant's `claim_supported` verdict is judged by `/result-to-claim` against the frozen main-experiment claim statement. Then `consistent_with_main_experiment` is computed:
- main_experiment_verdict = "supported" → consistent = claim_supported (pass if variant also supports, fail if not)
- robustness = n_pass / N_eligible (threshold 0.5; at N_eligible=1, requires 1/1 = 1.0)

---

<details>
<summary>Candidate Pool (audit trail)</summary>

### Model candidates (harvested from project context)

| # | Name | Source | Notes | On-disk |
|---|------|--------|-------|---------|
| Mdl1 | DINOv2 ViT-S (`dinov2-small`) | task.md verify_variants.student_swaps / IDEA_REPORT | 21M params, hidden=384, same DINOv2 self-supervised family; 4× smaller than main student | YES — /data/zhenqian/models/dinov2-small |
| Mdl2 | Supervised ViT-B (`vit-base-patch16-224`) | /data/zhenqian/models/vit-base-patch16-224 | 86M params, hidden=768, supervised ImageNet-1k classification; different training objective | YES |
| Mdl3 | DINOv2 ViT-L (`dinov2-large`) | /data/zhenqian/models/dinov2-large | ~307M params; same family, larger scale; higher GPU cost | YES |

**Candidates not on disk**: DINOv1 ViT-B, SigLIP ViT-B (teacher family — would violate teacher-fixed constraint anyway if confused), CapPa ViT-B

**Coverage**: 3 credible on-disk candidates > MIN_CANDIDATES_PER_DIMENSION=2. No /research-lit top-up needed.

**Reviewer ranking rationale (inline, llm-chat unavailable — applying principle)**:
1. **DINOv2 ViT-S** (trust rank 1) — Cheapest, most controlled test: only scale changes (not training objective). SANITY_FIRST=true mandates cheapest first. On disk. Task.md verify pool explicitly mentions ViT-S variants.
2. **Supervised ViT-B** (trust rank 2) — Cross-paradigm test: changes training objective from SSL to supervised, keeps scale same. Would test whether DINO-style features are necessary for alignment or supervised features also benefit. Not selected for this pass (MAX_VERIFY_CLAIMS=1 means 1 variant; ViT-S is more budget-conservative and explicitly in the canonical pool).
3. **DINOv2 ViT-L** (trust rank 3) — Scale-up test; highest GPU cost (~4× ViT-B params); risk of OOM within remaining ~7.8 hr budget for a full-backbone finetune.

</details>
