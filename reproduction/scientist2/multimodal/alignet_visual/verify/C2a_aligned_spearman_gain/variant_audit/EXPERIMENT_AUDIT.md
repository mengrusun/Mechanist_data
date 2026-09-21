# Experiment Audit — Variant: model-swap-dinov2-vits (C2a)
# Phase 9 variant integrity audit

## Scope
Variant directory: verify/C2a_aligned_spearman_gain/variants/model-swap-dinov2-vits/
Variant run output: runs/verify/C2a_model_swap_dinov2_vits/

## Evaluation Methodology Checks

### 1. Ground-truth authenticity
- THINGS held-out triplet data: same testset1.txt as main experiment — real human odd-one-out judgments
- Spearman GT: pairwise human similarity derived from training-set co-selection rates (identical to main experiment)
- No synthetic or self-generated ground truth
- Verdict: PASS

### 2. Score normalization / metric integrity
- spearman_aggregate computed via scipy.stats.spearmanr on cosine pairwise similarities vs human similarities
- n_pairs = 1,714,022 (same scale as main experiment — full THINGS concept pairwise matrix)
- Metric code path: eval_student_similarity_vits.py imports spearman_pairwise, build_human_similarity_from_triplets from main eval_student_similarity.py — identical computation
- No normalization applied to the Spearman coefficient itself
- Verdict: PASS

### 3. Phantom / fabricated results
- aligned/eval_things_multilevel.json: spearman_aggregate=0.5397, triplet_accuracy=0.5589 — consistent with ViT-S capacity (slightly below ViT-B 0.5554 aligned, expected)
- unaligned/eval_things_multilevel.json: spearman_aggregate=0.2157, triplet_accuracy=0.4448 — consistent with ViT-S unaligned baseline (slightly above ViT-B unaligned 0.1891, plausible)
- aligned/train_summary.json present: n_optim_steps=625, final_loss=0.0523, wall_clock_s=434.9 — genuine training run
- All output files confirmed on disk with non-trivial values
- Verdict: PASS

### 4. Dead metric code
- Primary metric: spearman_aggregate — actively computed and reported
- Secondary metrics: triplet_accuracy, per-level spearman/triplet_accuracy — all active
- Note: spearman_fine=NaN (n_pairs_fine=0) — same limitation as main experiment; fine level has too few held-out concept pairs (33 triplets, 0 pairs for pairwise Spearman). Not a variant-specific issue.
- Verdict: PASS (spearman_aggregate is not fine-level; the primary metric is clean)

### 5. Scope overclaim
- Variant tests: "does alignment improve aggregate Spearman in a smaller student (ViT-S)?"
- Results reported at same granularity as main experiment (aggregate + per-level)
- No claim made that ViT-S outperforms ViT-B or generalizes beyond THINGS heldout
- Verdict: PASS

### 6. Architectural correctness of student loading
- eval_student_similarity_vits.py loads architecture from MODEL_DIR/dinov2-small (384-dim)
- For aligned ckpt: state dict loaded with strict=False; missing/unexpected keys logged
- ViT-S checkpoint was produced by align_student.py loading AutoModel from MODEL_DIR/dinov2-small — same architecture used in training and eval
- No dimension mismatch risk (training and eval use identical backbone)
- Verdict: PASS

## Overall Verdict

overall_verdict: pass

No critical or major evaluation methodology issues found. One minor note (fine-level NaN) pre-exists in the main experiment and does not affect the primary metric spearman_aggregate.
