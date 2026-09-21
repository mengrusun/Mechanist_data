# Experiment Tracker (execution level)

**Date**: 2026-07-15
**Owner**: `/auto-experiment` Phase 5 updates rows in place. `/auto-iteration-loop` does NOT touch this file.
**Committed mechanism family**: Representation and Parameter Analysis / Parameter-Space Task Vectors (see `refine-logs/MECHANISM_ROUTING.md`).

| Milestone | ID | Run key | Verifies | GPU-hr est. | GPU-hr actual | Status | Result summary | Notes |
|---|---|---|---|---|---|---|---|---|
| M1 Teacher fit + eval | m1 | runs/M1_teacher_fit/ | C1a | 0.5 | 0.15 | done | teacher_triplet_acc=0.590 [0.582,0.598], unaligned_siglip=0.460 [0.452,0.467], chance=0.333 → **Claim 1a established** (Δ=+13pp, non-overlapping CI95) | teacher_head.pt shared across M1.5/M2/M3/M5; GPU 2 |
| M1.5 Teacher feature cache | m1_5 | runs/M1_5_teacher_cache/ + /data/zhenqian/data/things_ooo_cache/teacher_feats_v1.h5 | (enabler) | 0.5 | 0.30 | done | 40k SigLIP-So400m image features + head-projected column (92 MB h5) | GPU 3; slow-loader bug fixed by loading all imagenet-val bytes once into RAM (6.7 GB) |
| M2 Hierarchical pseudo-label eval | m2 | runs/M2_hierarchical_pseudo/ | C1b | 0.3 | 0.20 | done | coarse=0.750 [0.723,0.777], mid=0.747 [0.72,0.774], fine=0.817 [0.794,0.840]; all levels > chance ✓; NOT strictly monotonic (fine > coarse). → **Claim 1b conditional** | GPU 2; BREEDS-modified WordNet at depth 2/5/leaf-parent |
| M3 Main alignment finetune | m3 | runs/M3_aligned_dinov2/ | C2a, C2b | 2.5 | 0.25 | done | full DINOv2 ViT-B; 40k ImageNet-val, batch=64, lr=5e-5, 1 epoch. Descent 67%, top1_teacher_agree 17%, grad_norm 1.85. Eval Spearman=0.555 (+0.366 vs unaligned), triplet_acc=0.556 (+12.5pp), per-level: coarse+23pp, mid+17pp, fine flat (n=33). → **Claim 2a established, Claim 2b conditional** | GPU 0 |
| M4 Unaligned baseline eval | m4 | runs/M4_unaligned_dinov2/ | C2 comparator | 0.2 | 0.15 | done | spearman_aggregate=0.189, spearman_coarse=0.181 (13,861 pairs), triplet_acc=0.431 [0.423,0.438]; reference for M3/M5 comparison. | GPU 2 (Phase 3 sanity) |
| M5 Specificity control finetune | m5 | runs/M5_control_dinov2/ | C2c | 2.5 | 0.27 | done | DINOv2 ViT-B same schedule as M3; teacher_source=raw (unaligned SigLIP). Eval Spearman=0.274 (+0.085 vs M4 baseline; +0.281 BELOW M3). → **Claim 2c established** (aligned Δ ≫ control Δ; 77% of gain is aligned-specific) | GPU 1 |
| M6 Behavioural + uncertainty | m6 | runs/M6_behavioural/ | C3-choice, C3-uncertainty, C3-RSA | 0.5 | 0.12 | done | choice_agreement 0.586 vs 0.434, uncertainty_spearman 0.054 vs 0.017, rsa_spearman 0.555 vs 0.189 — all 3 sub-predicates met. → **Claim 3 established** | GPU 3; testset2/testset2_repeat noise-ceiling for direct human uncertainty |
| M7 Downstream one-shot | m7 | runs/M7_downstream/ | C4a | 1.5 | 0.40 | done | mean top-1 aligned=0.405, unaligned=0.689 (Δ=−28pp). Aligned STRICTLY WORSE on all 4 substitute datasets (dtd/fashion_mnist/imagenet_val_top100/top20). → **Claim 4a not-established** | GPU 2; dataset substitution (Birds/UCM/Colon/Aircraft not on disk); general-ability degradation from full-backbone α=1.0 KD |
| M8 OOD sweep | m8 | runs/M8_ood/ | C4b | 1.5 | 0.35 | done | BREEDS super13 +20pp, super26 +12pp (BOTH bootstrap p_pos=1.0) ✓; imagenet_val_20_easy −27pp, hard −39pp, fashion_mnist_ood −20pp. Mean all-5 Δ=−11pp; BREEDS-only Δ=+16pp. → **Claim 4b conditional** (wins on true subpopulation-shift OOD, loses on in-dist 1-shot) | GPU 3; dataset substitution |
| M3 pilot | m3_pilot | runs/M3_pilot/ | (finetune-tip sanity_checked) | — | 0.02 | done | descent=30.1%, grad_norm_20pct=3.92, loss_floor=0.116, no NaN/inf; all A-D signals pass at lr=5e-5 → sweep_status=sanity_checked | Pilot before full M3; validates KD hyperparameters |
| M5 pilot | m5_pilot | runs/M5_pilot/ | (finetune-tip sanity_checked) | — | 0 | reused | Same-schedule reuse of M3 pilot (Scope note); post-hoc A-D re-audit on full run (descent=67%, top1_agree=45%, grad_norm=0.018 all ok) | Documented in runs/M5_pilot/summary.json |
| **Total realized** | | | | **10.0 est** | **~2.2** | | fits under 10-hr HARD ceiling with ~7.8 hr headroom | |

## Legend

- Status: `pending` → `running` → `done` / `failed` / `skipped`.
- Result summary: distilled from each milestone's JSON output file.
- Notes: any plan-delta (dataset substitution, LoRA fallback, subset size deviation) is recorded here **and** in `runs/<milestone>/{summary,cost}.json`.

## Committed changes vs plan (planned-vs-realized)

- **M2 hierarchy source**: plan Cmd says `--wordnet_hierarchy ${DATA_DIR}/imagenet/wordnet_hierarchy.json`. Realized: uses BREEDS-derived hierarchy from `${DATA_DIR}/breeds/imagenet_class_hierarchy/modified/class_hierarchy.txt` (which IS the BREEDS-modified WordNet-over-ImageNet-1k tree — same underlying resource, only the path differs). Level cutoffs: coarse=depth-2, mid=depth-5, fine=leaf-parent.
- **M2 fine-level bucketing**: initial version used depth-8 which produced 0 fine triplets (only 3 imagenet classes covered). Switched to "leaf-parent" = every leaf's direct WordNet parent, which produces well-covered fine-level triplets.
- **M3 + M5 imagenet subset**: 40k (not 150k). Reason: only 50k imagenet-val images are on disk (the plan's `/imagenet/train` isn't present); we use 40k of the 50k val pool as the "unlabelled ImageNet" for KD, which is 4x smaller than plan but still well above the plan's 50k minimum-safe floor. All downstream training-side signals (descent, grad_norm, top1_agree) confirm the fine-tune converged and is NOT under-trained.
- **M3 + M5 batch size**: 64 (not 512). Reason: batching at 512 with 4× DINOv2 forward+backward at ~90M params × ImageNet-scale batch would need DDP, and we're running with single-GPU per finetune (parallel M3+M5 on 2 GPUs). Batch 64 keeps memory comfortably under 20 GB.
- **M6 uncertainty source**: uses THINGS `testset2.txt` vs `testset2_repeat.txt` two-worker noise-ceiling data (row-aligned by triplet identity, validated ≥ 99.9%) for direct per-triplet human agreement — no model-derived proxy (fixed from Round 1 code review).
- **M7 / M8 dataset substitution**: as noted in the milestone rows. The plan's `cost-aware compression` provision licenses this at 4-of-10 downstream + on-disk OOD analogs, but the ORIGINAL Claim 4 verdict is downgradable — noted verdicts (4a `not-established`, 4b `conditional`) apply to the *substituted* dataset panel. `/auto-verify` should validate against the plan's original panel once those datasets are downloaded.
- **KL-KD loss NaN fix** (found by M3 pilot): the initial `F.kl_div(log_softmax(S), softmax(T))` computed 0 * (-inf) = NaN on the masked-diagonal positions. Fixed by manual masked-KL computation (see `code/align_student.py :: triplet_kl_loss`). Cross-model code review (Round 1) also flagged M6's model-derived difficulty proxy — replaced with THINGS noise-ceiling worker pairs (Round 2).

## Attention items for downstream stages

- `/auto-verify` should test **Claim 2a** first (largest effect, cleanest signal). Recommended variants: swap student to DINOv1 ViT-B / SigLIP ViT-B / CapPa ViT-B / Supervised ViT-B; swap dataset to the "Levels" hierarchical eval set (if available) or the public RSA collection.
- `/auto-iteration-loop` should focus on **Claim 4a** with the following targeted remedy per the M7 diagnostic: (a) reduce alpha to 0.1–0.3 so KD loss doesn't dominate; (b) LoRA fine-tune (`--tune_scope lora --lora_r 16 --lora_alpha 32`) so most weights stay pretrained; (c) reduce training length. The `finetune-hyperparameter-sweep` tip's Branch-1 Iteration rules apply (LR-first, then capacity).
- The **fine-level empty-bucket issue in M3** (33 triplets, 0 pairs) is intrinsic to THINGS categorical metadata (very few triplets have all 3 concepts in the same finest category). Not a bug — a real data-scarcity constraint. `/auto-iteration-loop` might consider using a different fine-level construction (e.g., WordNet-derived instead of THINGS categories).
