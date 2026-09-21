# Robustness Report — C2a
# aligned_spearman_gain

## Claim
C2a: Aligned DINOv2 ViT-B improves aggregate Spearman correlation with human THINGS-triplet similarity judgments compared to unaligned DINOv2 ViT-B (delta_rho >= 0.05).

## Main Experiment Verdict
supported — spearman_aggregate: M3=0.5554 (aligned ViT-B) vs M4=0.1891 (unaligned ViT-B); delta_rho=+0.366 (7.3x threshold). High confidence.

## Stage-2 Verdict

PASS

**robustness = 1.00** (1 pass / 1 eligible; threshold = 0.50)

## Variant Results

| variant_tag | dimension | swap | aligned_spearman | unaligned_spearman | delta_rho | threshold | verdict | consistent_with_main | integrity |
|---|---|---|---|---|---|---|---|---|---|
| model-swap-dinov2-vits | model | ViT-B (86M) -> ViT-S (21M) | 0.5397 | 0.2157 | +0.3240 | 0.05 | supported | pass | PASS |

## Robustness Aggregation

- N_run: 1
- N_eligible (integrity PASS): 1
- N_integrity_fail (excluded): 0
- #variants_pass: 1
- #variants_fail: 0
- robustness: 1.00
- threshold: 0.50
- verdict: PASS

## Interpretation

The alignment improvement (triplet-KL distillation from SigLIP-So400m teacher) is robust to student model scale. With DINOv2 ViT-S (21M params, 384-dim) as student instead of ViT-B (86M params, 768-dim):
- Aligned ViT-S: spearman_aggregate = 0.5397 (vs aligned ViT-B = 0.5554 in main experiment; ~3% lower, expected with smaller model)
- Unaligned ViT-S: spearman_aggregate = 0.2157 (vs unaligned ViT-B = 0.1891 in main experiment; similar scale)
- delta_rho = +0.3240 (6.5x the 0.05 threshold; matches direction and order of magnitude of main experiment's +0.366)

The triplet-KL alignment mechanism transfers effectively across student model scales. C2a is PASS.

## Integrity Notes

- Phase 2 (baseline): EXPERIMENT=pass, MECHANISM=n/a; combined=pass
- Phase 9 (variant): EXPERIMENT=pass, MECHANISM=n/a; combined=pass; model-swap-dinov2-vits is integrity-eligible
- spearman_fine=NaN in both aligned and unaligned variant evals (n_pairs_fine=0) — pre-existing limitation in THINGS heldout split (same as main experiment); does not affect spearman_aggregate

## Artifacts

- runs/verify/C2a_model_swap_dinov2_vits/aligned/eval_things_multilevel.json — aligned ViT-S eval
- runs/verify/C2a_model_swap_dinov2_vits/unaligned/eval_things_multilevel.json — unaligned ViT-S eval
- runs/verify/C2a_model_swap_dinov2_vits/aligned/train_summary.json — alignment training stats
- runs/verify/C2a_model_swap_dinov2_vits/cost.json — run metadata (gpu_ids=[0,1,2,3], gpu_hours~0.5)
- verify/C2a_aligned_spearman_gain/variants/model-swap-dinov2-vits/results/verdict.json — Phase 8 judgment
- verify/C2a_aligned_spearman_gain/variant_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json} — Phase 9 integrity
