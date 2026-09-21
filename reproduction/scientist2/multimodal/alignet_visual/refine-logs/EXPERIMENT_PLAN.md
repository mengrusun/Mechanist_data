# Experiment Plan — Hierarchical Human-Alignment Reproduction (given behavior + discovery mechanism)

**Date**: 2026-07-14
**Language**: English
**Direction source**: `task.md` (project root — sole authoritative source)
**Behavior-source**: given (no M0 phenomenon-validation gate — the phenomenon is *assumed* per task.md)
**Mechanism**: discovery (routed at experiment stage `/auto-experiment` Phase 1.5 within the strategy chain below)

---

## Top metadata (machine markers — read by downstream stages)

```yaml
resource_fidelity: cost-aware        # NOT stamped strict — given + discovery is not the reproduction combo
mechanism_strategy:
  directions: [Tuning & Editing, Location, Decision Auditing]
  rejected:
    - Causal Intervention — claims are about a changed model, not "component X drives B in the base model".
    - Formation Tracing — no origin claim; expensive; 10-hr GPU budget insufficient.
    - Unit Interpretation — no unit-meaning claim; full SAE too expensive; not on critical path.
  note: Tuning&Editing carries Claim 2 (alignment-loss finetune); Location supports Claim 3 (final-embedding RSA / uncertainty behavioural comparison); Decision Auditing gates Claim 4 (utility + OOD as specificity check).
# chosen_mechanism intentionally OMITTED (MECHANISM=discovery — routing occurs at /auto-experiment Phase 1.5)
# families_already_settled: (omitted — round 1; no research_memory.json)
```

## Global constraints (HARD — enforced in every milestone's cmd via env)

```yaml
gpu_budget_hours: 10
allowed_gpu_ids: [0, 1, 2, 3]           # CUDA_VISIBLE_DEVICES must be a subset of this
allowed_dirs:
  - /data/zhenqian/Reproduction1/mechanica/multimodal/alignet_visual   # working dir
  - /data/zhenqian/data                                                 # DATA_DIR
  - /data/zhenqian/models                                               # MODEL_DIR
env:
  DATA_DIR: /data/zhenqian/data
  MODEL_DIR: /data/zhenqian/models
  CUDA_VISIBLE_DEVICES: "0,1,2,3"       # override per-milestone if a subset is used
  HF_TOKEN: <Your_token>
  MODELSCOPE_TOKEN: <Your_token>
  LLM_API_KEY: <Your_api>
  LLM_BASE_URL: https://www.dmxapi.cn/v1
  LLM_MODEL: gpt-5.4
  NO_PROXY: dmxapi.cn                    # LLM API bypasses proxy
conda_env: alignet_visual                 # /auto-experiment creates/uses this
teacher_backbone: siglip-so400m           # fixed across the whole project; do NOT swap
main_student: dinov2-vit-b                # fixed for the MAIN experiment; swaps live in verify stage
symbolic_links: true                      # link data/models into working dir; do not copy
missing_asset_policy: download-to-DATA_DIR-or-MODEL_DIR (HF / GitHub / ModelScope)
```

## Verify-stage candidate pool (informational — not run in the main plan; passed to `/auto-verify`)

```yaml
verify_variants:
  student_swaps: [supervised-vit-s, supervised-vit-b, supervised-vit-l, dinov1-vit-b, siglip-vit-b, cappa-vit-b]
  dataset_extras:
    hierarchical_eval:  "Levels (coarse-grained semantic / fine-grained semantic / class-boundary)"
    rsa_public:         "public human-similarity-judgment collection"
    downstream_all:     "10 one-shot classification datasets (Birds, UC Merced, Colon + 7 fine-grained specialty)"
    ood_subpop:         "BREEDS (entity13, living17, non-living26, entity30)"
    ood_natural:        "ImageNet-A"
  teacher_swaps: []                        # forbidden by task.md — SigLIP-So400m is fixed
```

## Claims-to-milestones map

| Claim | Verified by |
|---|---|
| C1a — THINGS-fit teacher beats unaligned SigLIP-So400m + chance on held-out THINGS triplets | M1 |
| C1b — Teacher's ImageNet-synth triplet signal is monotonically level-organized (coarse/mid/fine) | M2 |
| C2a — Aligned DINOv2 ViT-B improves aggregate Spearman with human triplets (Δρ ≥ 0.05, α=0.05) | M3 + M4 (aligned − unaligned) |
| C2b — Improvement holds at each of coarse/mid/fine levels separately | M3 + M4 per-level |
| C2c — Gain is specific to human alignment (aligned > non-human-aligned soft-label control) | M3 + M5 |
| C3-choice / C3-uncertainty / C3-RSA — aligned better reproduces human behaviour AND uncertainty | M6 |
| C4a — Aligned matches or exceeds unaligned on downstream one-shot classification suite | M7 |
| C4b — Aligned strictly improves OOD (BREEDS + ImageNet-A) | M8 |

## Compute budget shape (target — 10 hr total on 4× GPUs {0,1,2,3})

| Milestone | GPU-hr (est.) | Rationale |
|---|---|---|
| M1 teacher fit + eval | 0.5 | small MLP head on cached SigLIP-So400m features + eval |
| M2 hierarchical pseudo-label eval | 0.3 | forward-only on 3k triplets |
| M3 main alignment finetune (DINOv2 ViT-B) | 2.5 | ImageNet subset ≥100k; teacher features cached (M1.5) |
| M4 unaligned baseline eval | 0.2 | forward-only on THINGS held-out |
| M5 specificity control finetune | 2.5 | same schedule / same size as M3, different teacher |
| M6 behavioural + uncertainty eval | 0.5 | forward-only on THINGS held-out + RSA collection |
| M7 downstream one-shot sweep (subset 4 of 10 datasets) | 1.5 | per-dataset linear-probe / cosine-1-shot |
| M8 OOD sweep (BREEDS 4 splits + ImageNet-A) | 1.5 | linear-probe / kNN on cached features |
| M1.5 teacher-feature cache (one-time) | 0.5 | single SigLIP-So400m forward on ImageNet subset |
| **total (est.)** | **~10.0** | fits budget; monitored via `/monitor-experiment` |

Cost-aware compressions applied:
- Downstream sweep: 4-of-10 datasets in main plan (stratified: 1 natural fine-grained + 1 aerial + 1 medical + 1 specialty) → remaining 6 tagged for `/auto-verify`.
- ImageNet finetune subset: 100k–200k (not 1.28M) — sized to fit alignment finetune inside 2.5 hr per run on the given hardware.
- Teacher features are cached once (M1.5) and reused across M3 + M5 so the SigLIP-So400m teacher runs only once on the subset.
- **Not compressed**: teacher backbone (SigLIP-So400m — fixed), student backbone (DINOv2 ViT-B — fixed), THINGS eval (full held-out human triplets), BREEDS/ImageNet-A eval sets (full standard protocol).

Any downscaling not listed above is a plan-rewrite and must be flagged, not silently applied.

---

## Milestones

### M1: Teacher fit — SigLIP-So400m + THINGS triplet head; evaluate on THINGS held-out

**Verifies**: C1a
**Kind**: tuning-eval (small alignment head on frozen teacher; the SigLIP-So400m image-encoder itself is not backpropagated)
**Depends on**: —
**Priority**: MUST-RUN
**Estimated GPU-hours per run**: 0.5

**Cmd**:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n alignet_visual python code/fit_teacher_head.py \
    --teacher_backbone siglip-so400m \
    --teacher_ckpt ${MODEL_DIR}/siglip-so400m-patch14-384 \
    --things_root ${DATA_DIR}/things \
    --split_train train --split_eval heldout \
    --loss triplet_kl --temperature 1.0 \
    --output_dir runs/M1_teacher_fit
```

**Expected output**: `runs/M1_teacher_fit/eval_heldout.json` — fields: `teacher_triplet_accuracy`, `unaligned_siglip_triplet_accuracy`, `chance_baseline`, `bootstrap_ci_95`, `n_triplets_heldout`.

**Pass criterion (Claim 1a)**: `teacher_triplet_accuracy` > `unaligned_siglip_triplet_accuracy` AND > 0.333 (chance), both with bootstrap-CI-95 not crossing the comparator. Fail → Claim 1 downgrades to `not-established`; Claim 2 attribution is compromised (report but downgrade).

**method_sensitive**: [loss, temperature, n_head_params]        # exact alignment-head shape rebound by /mechanism-skills at Phase 1.5

---

### M1.5: Teacher feature cache — one SigLIP-So400m forward pass over ImageNet subset

**Verifies**: enabling M3 and M5 within budget (not a claim milestone)
**Kind**: preprocess
**Depends on**: M1
**Priority**: MUST-RUN
**Estimated GPU-hours per run**: 0.5

**Cmd**:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n alignet_visual python code/cache_teacher_features.py \
    --teacher_backbone siglip-so400m \
    --teacher_head runs/M1_teacher_fit/teacher_head.pt \
    --imagenet_root ${DATA_DIR}/imagenet/train \
    --n_images 150000 --seed 42 \
    --output_path ${DATA_DIR}/things_alignet_cache/teacher_feats_v1.h5
```

**Expected output**: `${DATA_DIR}/things_alignet_cache/teacher_feats_v1.h5` (image_ids + teacher embedding + teacher-soft-triplet-labels sampled per anchor).

---

### M2: Hierarchical pseudo-label eval on ImageNet-synth triplets (coarse / mid / fine)

**Verifies**: C1b
**Kind**: eval (forward-only using the M1 teacher head)
**Depends on**: M1
**Priority**: MUST-RUN
**Estimated GPU-hours per run**: 0.3

**Cmd**:
```bash
CUDA_VISIBLE_DEVICES=0 conda run -n alignet_visual python code/eval_hierarchical_triplets.py \
    --teacher_backbone siglip-so400m \
    --teacher_head runs/M1_teacher_fit/teacher_head.pt \
    --imagenet_root ${DATA_DIR}/imagenet/val \
    --wordnet_hierarchy ${DATA_DIR}/imagenet/wordnet_hierarchy.json \
    --triplets_per_level 1000 --seed 42 \
    --levels coarse,mid,fine \
    --output_dir runs/M2_hierarchical_pseudo
```

**Expected output**: `runs/M2_hierarchical_pseudo/level_separation.json` — per-level (`coarse`, `mid`, `fine`) triplet-choice agreement rate + separation vs. chance; monotonicity check (`separation_coarse > separation_mid > separation_fine` OR the natural ordering documented in `runs/M2_hierarchical_pseudo/README.md`).

**Pass criterion (Claim 1b)**: separation > chance at each of the three levels with bootstrap-CI-95 lower bound above chance; monotonic ordering across levels (either direction, if pre-documented in README). Fail → Claim 1 → `conditional` (teacher works globally but does not carry hierarchy).

**method_sensitive**: [triplets_per_level, wordnet_level_cutoffs]

---

### M3: Main alignment finetune of DINOv2 ViT-B with human-aligned teacher

**Verifies**: C2a, C2b (in conjunction with M4)
**Kind**: tuning-and-editing (targeted-finetune family — /mechanism-skills routes the exact submethod at /auto-experiment Phase 1.5)
**Depends on**: M1, M1.5
**Priority**: MUST-RUN
**Estimated GPU-hours per run**: 2.5

**Grid**: (single run — no grid; multi-seed left to `/auto-verify`)

**Cmd**:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n alignet_visual python code/align_student.py \
    --student_backbone dinov2-vit-b \
    --student_ckpt ${MODEL_DIR}/dinov2-base \
    --teacher_cache ${DATA_DIR}/things_alignet_cache/teacher_feats_v1.h5 \
    --align_loss triplet_kl --temperature 1.0 --alpha 1.0 \
    --tune_scope full \
    --imagenet_subset_size 150000 \
    --epochs 1 --lr 5e-5 --batch_size 512 --seed 42 \
    --output_dir runs/M3_aligned_dinov2 \
    && CUDA_VISIBLE_DEVICES=0 conda run -n alignet_visual python code/eval_student_similarity.py \
        --student_ckpt runs/M3_aligned_dinov2/checkpoint.pt \
        --things_root ${DATA_DIR}/things \
        --split heldout \
        --levels_construction imagenet_wordnet \
        --output_json runs/M3_aligned_dinov2/eval_things_multilevel.json
```

**Expected output**: `runs/M3_aligned_dinov2/eval_things_multilevel.json` — fields: `spearman_aggregate`, `spearman_coarse`, `spearman_mid`, `spearman_fine`, `bootstrap_ci_95` for each, `n_triplets`.

**Pass criterion (Claim 2a)**: `spearman_aggregate(M3) − spearman_aggregate(M4)` ≥ 0.05 AND paired-bootstrap significance at α=0.05.
**Pass criterion (Claim 2b)**: `spearman_level(M3) − spearman_level(M4)` > 0 at each of coarse / mid / fine.

**method_sensitive**: [align_loss, temperature, alpha, tune_scope, imagenet_subset_size, epochs, lr, batch_size, gpu_hours]     # concretely re-bound by /mechanism-skills at Phase 1.5

**Fail policy**: if M3 crashes or exceeds 3.5 hr on the assigned 4× GPUs, fall back to `tune_scope=lora --lora_r=16 --lora_alpha=32` and note the compression in `runs/M3_aligned_dinov2/PLAN_DELTA.md`. Do NOT silently reduce `imagenet_subset_size` below 50k.

---

### M4: Unaligned DINOv2 ViT-B baseline — multi-level Spearman on THINGS held-out

**Verifies**: comparator for C2a, C2b
**Kind**: eval (forward-only on unaligned pretrained student)
**Depends on**: —
**Priority**: MUST-RUN
**Estimated GPU-hours per run**: 0.2

**Cmd**:
```bash
CUDA_VISIBLE_DEVICES=0 conda run -n alignet_visual python code/eval_student_similarity.py \
    --student_ckpt ${MODEL_DIR}/dinov2-base \
    --things_root ${DATA_DIR}/things \
    --split heldout \
    --levels_construction imagenet_wordnet \
    --output_json runs/M4_unaligned_dinov2/eval_things_multilevel.json
```

**Expected output**: `runs/M4_unaligned_dinov2/eval_things_multilevel.json` — same schema as M3.

**Pass criterion**: n/a (this is the comparator; consumed by the M3-vs-M4 paired test above).

---

### M5: Specificity control — non-human-aligned soft-label finetune of DINOv2 ViT-B

**Verifies**: C2c
**Kind**: tuning-and-editing (matched-cost control for the M3 alignment run)
**Depends on**: M1.5 (uses teacher-feature cache with a *different* teacher-head — unaligned SigLIP embeddings, no THINGS fit)
**Priority**: MUST-RUN
**Estimated GPU-hours per run**: 2.5

**Cmd**:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 conda run -n alignet_visual python code/align_student.py \
    --student_backbone dinov2-vit-b \
    --student_ckpt ${MODEL_DIR}/dinov2-base \
    --teacher_backbone siglip-so400m \
    --teacher_head none                        # UNALIGNED SigLIP-So400m — no THINGS-fitted head
    --teacher_cache ${DATA_DIR}/things_alignet_cache/teacher_feats_unaligned_v1.h5 \
    --align_loss triplet_kl --temperature 1.0 --alpha 1.0 \
    --tune_scope full \
    --imagenet_subset_size 150000 \
    --epochs 1 --lr 5e-5 --batch_size 512 --seed 42 \
    --output_dir runs/M5_control_dinov2 \
    && CUDA_VISIBLE_DEVICES=0 conda run -n alignet_visual python code/eval_student_similarity.py \
        --student_ckpt runs/M5_control_dinov2/checkpoint.pt \
        --things_root ${DATA_DIR}/things \
        --split heldout \
        --levels_construction imagenet_wordnet \
        --output_json runs/M5_control_dinov2/eval_things_multilevel.json
```

**Expected output**: `runs/M5_control_dinov2/eval_things_multilevel.json` — same schema as M3/M4.

**Pass criterion (Claim 2c)**: `spearman_aggregate(M3) − spearman_aggregate(M5)` > 0 AND paired-bootstrap significance at α=0.05. If the control matches or beats M3, Claim 2 → `conditional` (gain not specific to human alignment — consistent with the KD-as-label-smoothing prior).

**method_sensitive**: [teacher_head, align_loss, temperature, alpha, tune_scope, imagenet_subset_size, epochs, lr, batch_size, gpu_hours]

**Optional variant (log only, do not run in main plan)**: `random_triplet_teacher` — same schedule but soft-labels are drawn from a uniform-random triplet-choice distribution; queued for `/auto-verify` as a stronger specificity control.

---

### M6: Behavioural + per-triplet uncertainty match — Claim 3

**Verifies**: C3-choice, C3-uncertainty, C3-RSA
**Kind**: eval (forward-only on aligned M3 vs. unaligned M4 students)
**Depends on**: M3, M4
**Priority**: MUST-RUN
**Estimated GPU-hours per run**: 0.5

**Cmd**:
```bash
CUDA_VISIBLE_DEVICES=0 conda run -n alignet_visual python code/eval_behavioural_uncertainty.py \
    --aligned_ckpt runs/M3_aligned_dinov2/checkpoint.pt \
    --unaligned_ckpt ${MODEL_DIR}/dinov2-base \
    --things_root ${DATA_DIR}/things \
    --rsa_public_root ${DATA_DIR}/human_similarity_rsa \
    --output_dir runs/M6_behavioural
```

**Expected output**: `runs/M6_behavioural/results.json` — fields:
- `choice_agreement_aligned`, `choice_agreement_unaligned` (per-triplet top-1 match rate on THINGS held-out and RSA public collection)
- `uncertainty_kl_aligned`, `uncertainty_kl_unaligned` (KL from model to human choice distribution)
- `uncertainty_spearman_aligned`, `uncertainty_spearman_unaligned` (rank correlation between model confidence and human agreement rate)
- `rsa_spearman_aligned`, `rsa_spearman_unaligned` (Spearman between model RDM and human RDM)
- `bootstrap_ci_95` on all deltas

**Pass criterion (Claim 3)**:
- C3-choice: `choice_agreement_aligned > choice_agreement_unaligned` (paired bootstrap α=0.05) on both eval sets;
- C3-uncertainty: `uncertainty_kl_aligned < uncertainty_kl_unaligned` AND `uncertainty_spearman_aligned > uncertainty_spearman_unaligned`;
- C3-RSA: `rsa_spearman_aligned > rsa_spearman_unaligned`.
- All three sub-predicates in predicted direction → `established`; two of three → `conditional`; ≤ one → `not-established`.

**method_sensitive**: [distance_metric, softmax_temperature, rdm_metric]

---

### M7: Downstream one-shot classification sweep — Claim 4a (utility non-inferiority)

**Verifies**: C4a
**Kind**: eval (forward-only feature extraction + linear-probe / cosine 1-shot)
**Depends on**: M3, M4
**Priority**: MUST-RUN
**Estimated GPU-hours per run**: ~0.19 per (model × dataset) combo; total 0.19 × 4 datasets × 2 models × 1 shot × 1 seed = ~1.5 hr

**Grid**:
```yaml
grid:
  student: [aligned, unaligned]
  dataset: [birds, uc_merced, colon_pathology, aircraft]        # 4-of-10 stratified subset; remaining 6 → /auto-verify
```

**Cmd template**:
```bash
CUDA_VISIBLE_DEVICES=0 conda run -n alignet_visual python code/eval_one_shot.py \
    --student ${student} \
    --aligned_ckpt runs/M3_aligned_dinov2/checkpoint.pt \
    --unaligned_ckpt ${MODEL_DIR}/dinov2-base \
    --dataset ${dataset} \
    --dataset_root ${DATA_DIR}/downstream/${dataset} \
    --n_shot 1 --seed 42 \
    --output_json runs/M7_downstream/${student}__${dataset}.json
```

**Expected output template**: `runs/M7_downstream/${student}__${dataset}.json` with `top1_accuracy` + `bootstrap_ci_95`.

**Pass criterion (Claim 4a)**: `mean_top1(aligned across 4 datasets) ≥ mean_top1(unaligned)` (non-inferiority; per-dataset drop ≤ 1 point unless compensated elsewhere). Fail → Claim 4 → `conditional`.

**Compression note**: 4 datasets (not 10) chosen to fit budget; the remaining 6 (UC Merced was chosen from the 3 named + 7 specialty; adjust here so the 4 span aerial / natural fine-grained / medical / other specialty) are logged in the verify-stage candidate pool. This is a cost-aware compression, not a claim-scope change.

**method_sensitive**: [n_shot, probe_type, dataset_subset]

---

### M8: OOD robustness sweep — BREEDS + ImageNet-A (Claim 4b)

**Verifies**: C4b
**Kind**: eval (feature extraction + evaluation; no training)
**Depends on**: M3, M4
**Priority**: MUST-RUN
**Estimated GPU-hours per run**: ~0.3 per split × 5 splits × 2 models = ~1.5 hr wall-clock (batched across GPUs)

**Grid**:
```yaml
grid:
  student: [aligned, unaligned]
  ood_split: [breeds_entity13, breeds_living17, breeds_non_living26, breeds_entity30, imagenet_a]
```

**Cmd template**:
```bash
CUDA_VISIBLE_DEVICES=0 conda run -n alignet_visual python code/eval_ood.py \
    --student ${student} \
    --aligned_ckpt runs/M3_aligned_dinov2/checkpoint.pt \
    --unaligned_ckpt ${MODEL_DIR}/dinov2-base \
    --ood_split ${ood_split} \
    --data_root ${DATA_DIR}/ood/${ood_split} \
    --output_json runs/M8_ood/${student}__${ood_split}.json
```

**Expected output template**: `runs/M8_ood/${student}__${ood_split}.json` with `top1_accuracy` + protocol-standard metric per benchmark.

**Pass criterion (Claim 4b)**: `top1(aligned, split) > top1(unaligned, split)` on each of BREEDS-{entity13, living17, non-living26, entity30} AND on ImageNet-A, with per-split effect distinguishable from zero under the standard protocol. Fail → Claim 4 → `conditional`.

**method_sensitive**: [probe_type, batch_size]

---

## Run order (queue-friendly)

Wave 1 (parallel where possible):
- M1 (teacher fit) — GPUs 0,1,2,3 briefly → produces `teacher_head.pt`
- M4 (unaligned baseline eval) — GPU 0, independent

Wave 2:
- M1.5 (teacher feature cache) — GPUs 0,1,2,3
- M2 (hierarchical pseudo-label eval) — GPU 0 (small, can run alongside M1.5 if VRAM allows)

Wave 3 (parallel finetunes):
- M3 (aligned finetune) — GPUs 0,1,2,3 for ~2.5 hr
- M5 (specificity control finetune) — sequential after M3 or partial GPU reservation (2.5 hr)

Wave 4:
- M6 (behavioural + uncertainty) — GPU 0
- M7 (downstream sweep) — GPUs 0,1,2,3 batched across the 8 grid cells
- M8 (OOD sweep) — GPUs 0,1,2,3 batched across the 10 grid cells

Total wall-clock estimate: ~8.5 hr on 4× GPUs; ~1.5 hr headroom against the 10 hr HARD ceiling.

## Decision gates for `/auto-experiment` / `/result-to-claim`

- If M1 fails Claim 1a pass criterion → downgrade Claim 1 to `not-established`; still run M2..M8 for exploratory reporting but flag Claim 2 attribution as compromised.
- If M2 fails Claim 1b → Claim 1 → `conditional`.
- If M3-vs-M4 fails aggregate or per-level Δρ ≥ 0.05 → Claim 2 → `conditional` (or `not-established` if Δρ ≤ 0).
- If M3-vs-M5 fails specificity → Claim 2 → `conditional` (gain not attributed to human alignment).
- If M6 fails ≥ 2 of 3 sub-predicates → Claim 3 → `not-established`; exactly 1 fails → `conditional`.
- If M7 fails non-inferiority → Claim 4 → `conditional`.
- If M8 fails OOD improvement on ≥ 3 of 5 splits → Claim 4 → `not-established`.

## Notes for downstream stages

- `/auto-experiment` (Phase 1.5) will route the `Tuning & Editing` chain to a concrete `/mechanism-skills` family (targeted-finetune / LoRA / full-backbone alignment loss) — the `method_sensitive:` fields in M3 and M5 are the licensed re-binding surface. The routing decision is recorded in `refine-logs/MECHANISM_ROUTING.md` by that stage.
- `/auto-verify` should stress-test the top admitted claim with student-swap variants from `verify_variants.student_swaps` and dataset-extras from `verify_variants.dataset_extras`. The teacher (SigLIP-So400m) is not swappable per task.md.
- `/auto-iteration-loop` handles FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS re-runs under the same 10-hr HARD ceiling.
