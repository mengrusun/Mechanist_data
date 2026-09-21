# EXPERIMENT_PLAN — SemanticLens Component → CLIP Semantic-Vector Verification on ResNet-50 / ImageNet

**Behavior-source**: given
**Mechanism**: discovery (experiment stage will route via `/mechanism-skills`)
**Date**: 2026-07-13
**Anchor**: `refine-logs/FINAL_PROPOSAL.md` + `idea-stage/IDEA_REPORT.md` (claims C1 & C2, FROZEN)

---

## Top metadata (machine markers — read by `/auto-experiment`, `/auto-verify`, `/auto-iteration-loop`)

```yaml
resource_fidelity: cost-aware          # NOT strict — cost-aware within 10 h GPU budget
mechanism_strategy:
  directions: ["Unit Interpretation", "Decision Auditing"]
  rejected:
    - "Causal Intervention — the given behavior asserts encoding of c, not that ablating c changes model output; out of claim scope."
    - "Tuning & Editing — no downstream capability is being tuned; SemanticLens is diagnostic, not applied."
    - "Formation Tracing — inference-time claim; training-time origin out of scope and infeasible within 10 h GPU budget."
    - "Location (headline) — used only as a lightweight specificity screen for v_c (concept text-query -> highest-cosine components + matched-control text)."
  note: "SemanticLens is a cross-modal Unit-Interpretation method; v_c is the interpretable object, and Decision-Auditing (concept search, cross-model comparison, spurious-feature detection) is the downstream operation that both validates v_c's fidelity and demonstrates its use."
# chosen_mechanism intentionally NOT stamped — MECHANISM=discovery; /auto-experiment Phase 1.5 will bind the concrete Unit-Interpretation submethod (reference-input pooling + text-similarity scoring, per FINAL_PROPOSAL.md) as CHOSEN_FAMILY.
gpu_budget_hours: 10
gpu_allowlist: [1, 2, 3, 5, 6]
workdir: /data/zhenqian/Reproduction1/mechanica/feature_description/multi_modal_feature_description
data_dir: /data/zhenqian/data
model_dir: /data/zhenqian/models
conda_env_required: true
```

**Global constraints (repeated per-milestone by `${GPU_LIST}` template placeholder — the queue picks a subset):**
- Every `cmd` sets `CUDA_VISIBLE_DEVICES=${GPU_LIST}` where `${GPU_LIST}` ⊆ `{1,2,3,5,6}`.
- No access outside `${workdir}`, `${data_dir}`, `${model_dir}`.
- Dedicated conda env required (see `M0_setup` below).

**Notice items encoded here (from `task.md`)**:
- HF token env var: `HF_TOKEN=<Your_token>`
- ModelScope token env var: `MS_TOKEN=<Your_token>`
- LLM API for downstream audit / result-to-claim:
  - `LLM_API_KEY=<Your_api>`
  - `LLM_BASE_URL=https://www.dmxapi.cn/v1`
  - `LLM_MODEL=gpt-5.4`
  - Set `no_proxy=*` or `NO_PROXY=*` — bypass proxy.

---

## Milestones

> **No M0 phenomenon-validation milestone.** `BEHAVIOR_SOURCE=given` → the behavior is taken as already validated; there is no `kind: phenomenon-validation` gate and no `depends_on: [M0]` on subsequent milestones.

### M0_setup — Environment & data preparation (infrastructure, not phenomenon-validation)

**Covers claim(s)**: shared infrastructure for C1 and C2.
**Depends on**: none.
**Purpose**: create the dedicated conda env, symlink data, download the frozen CLIP checkpoint, sanity-check ResNet-50 ImageNet-val top-1 accuracy.

```yaml
priority: MUST-RUN
gpu_hours: 0.3
kind: infrastructure                    # NOT phenomenon-validation
```

**Cmd**:
```bash
CUDA_VISIBLE_DEVICES=${GPU_LIST:-1} \
  bash scripts/00_setup_env.sh
```
The script:
1. `conda create -n semlens python=3.10 -y; conda activate semlens; pip install torch torchvision open_clip_torch h5py numpy scipy scikit-learn tqdm pandas`
2. Symlink `${data_dir}/ImageNet -> ./data/imagenet` (if not already linked).
3. Ensure `${model_dir}/clip-vit-b32/openai_clip_vit-b32.pt` exists; else download via `open_clip.create_model('ViT-B-32', pretrained='openai')` cached under `${model_dir}`.
4. Ensure `${model_dir}/resnet50-imagenet.pth` exists; else download via `torchvision.models.resnet50(weights='IMAGENET1K_V2')` and pin the weights.
5. Log SHA-256 of both CLIP + ResNet-50 checkpoints to `runs/M0_setup/checksums.json`.
6. Sanity: report ResNet-50 top-1 on 5 000-image val subsample (should be ≥ 76 %).

**Expected output**: `runs/M0_setup/checksums.json`, `runs/M0_setup/sanity_top1.json`.
**Pass criterion**: `sanity_top1.top1 ≥ 0.76`.

---

### M1_activations — Cache ResNet-50 channel activations on ImageNet-val

**Covers claim(s)**: shared prerequisite for C1 and C2.
**Depends on**: `[M0_setup]`.
**Purpose**: One ResNet-50 forward pass over ImageNet-val (50 000 images); cache spatial-mean-pooled channel activations for `layer3`, `layer4`, and the pre-`fc` penultimate features. Also store per-image class label & filename.

```yaml
priority: MUST-RUN
gpu_hours: 0.2
kind: activation-cache
```

**Cmd**:
```bash
CUDA_VISIBLE_DEVICES=${GPU_LIST:-1} \
  python scripts/01_cache_activations.py \
    --model resnet50 \
    --split val \
    --data_dir ./data/imagenet \
    --model_ckpt ${MODEL_DIR}/resnet50-imagenet.pth \
    --layers layer3 layer4 avgpool fc \
    --pool spatial_mean \
    --out runs/M1_activations/resnet50_val_activations.h5 \
    --dtype fp16 \
    --batch_size 256
```
Expected output: HDF5 file with `{layer}/activations` shape `(50000, num_channels)`, `image_index`, `class_label`, `filename` datasets.
**Pass criterion**: file exists; shape correct; NaN count = 0.

---

### M2_reference_sets — Top-k reference-input sets R_c

**Covers claim(s)**: shared prerequisite for C1 and C2.
**Depends on**: `[M1_activations]`.
**Purpose**: For each component *c* (component universe = all last-layer 1000 units + 500 stratified `layer3` channels + 500 stratified `layer4` channels = **2000 components**), rank inputs by activation and store the top-`k_max=256` image indices.

```yaml
priority: MUST-RUN
gpu_hours: 0.1                          # CPU-heavy; no GPU strictly needed
kind: index-derivation
components:
  last_layer: 1000                       # all 1000 fc units
  layer4_sampled: 500                    # stratified from 2048 channels
  layer3_sampled: 500                    # stratified from 1024 channels
  total: 2000
k_max: 256
```

**Cmd**:
```bash
python scripts/02_reference_sets.py \
  --activations runs/M1_activations/resnet50_val_activations.h5 \
  --n_last 1000 --n_layer4 500 --n_layer3 500 \
  --stratify_by channel_index_bucket \
  --k_max 256 \
  --seed 42 \
  --out runs/M2_reference_sets/refsets.h5
```
Expected output: HDF5 with `component_id → top256_image_indices`, `component_id → top256_activations`.
**Pass criterion**: 2000 component rows; per-image coverage histogram logged for R1 (concentration collapse detection).

---

### M3_clip_embeddings — CLIP image embeddings for the union of reference inputs

**Covers claim(s)**: shared prerequisite for C1 and C2.
**Depends on**: `[M2_reference_sets, M0_setup]`.
**Purpose**: Compute frozen CLIP image-tower embeddings for every unique ImageNet image that appears in any *R_c*; cache to HDF5. This is the ONE CLIP forward pass in the main experiment.

```yaml
priority: MUST-RUN
gpu_hours: 0.7                          # worst case; likely 0.3-0.5 with dedup
kind: embedding-cache
clip_ckpt: openai/CLIP ViT-B-32 (pinned SHA in runs/M0_setup/checksums.json)
```

**Cmd**:
```bash
CUDA_VISIBLE_DEVICES=${GPU_LIST:-1} \
  python scripts/03_clip_embed.py \
    --refsets runs/M2_reference_sets/refsets.h5 \
    --data_dir ./data/imagenet/val \
    --clip_arch ViT-B-32 --clip_pretrained openai \
    --clip_ckpt_cache ${MODEL_DIR}/clip-vit-b32 \
    --normalize l2 \
    --dtype fp16 \
    --batch_size 512 \
    --out runs/M3_clip_embeddings/clip_val_embeddings.h5
```
Expected output: HDF5 with `image_index → clip_image_embedding (512-dim, L2-normalized)`.
**Pass criterion**: coverage of every unique image in `refsets.h5`; NaN count = 0.

---

### M4_v_c — Per-component pooled semantic vectors v_c across k and pooling operators

**Covers claim(s)**: C1 (via *k*-sweep), C2 (via pooling ablation, all four operators).
**Depends on**: `[M3_clip_embeddings]`.
**Purpose**: For each (component, k, pooling) triple, compute v_c from the cached CLIP embeddings; store all pools in one HDF5 for downstream predicate scripts to slice.

```yaml
priority: MUST-RUN
gpu_hours: 0.05                          # CPU-friendly
kind: pooled-vector
grid:
  k: [1, 4, 16, 64, 256]
  pool: [mean, act_weighted_mean, max, medoid]
# 5 × 4 = 20 (k, pool) settings × 2000 components = 40000 v_c vectors.
method_sensitive: [pool, sites]         # /auto-experiment Phase 1.5 may rebind these to the concrete Unit-Interpretation submethod
```

**Cmd (template with grid expansion)**:
```bash
python scripts/04_pool_v_c.py \
  --refsets runs/M2_reference_sets/refsets.h5 \
  --clip_emb runs/M3_clip_embeddings/clip_val_embeddings.h5 \
  --k ${k} --pool ${pool} \
  --out runs/M4_v_c/v_c__k${k}__${pool}.h5
```

**Expected output (template)**: `runs/M4_v_c/v_c__k{k}__{pool}.h5` — 20 files, each `component_id → v_c (512-dim, L2-normalized)`.
**Pass criterion (per file)**: 2000 rows; NaN count = 0.

---

### M5_text_embeddings — CLIP text embeddings for concept vocabularies

**Covers claim(s)**: C1 hidden-layer matched-control test, C2 H2a/H2c.
**Depends on**: `[M0_setup]` (only).
**Purpose**: Embed (a) ImageNet-1k class names with the 7-prompt-template OpenAI ensemble, and (b) a broader open vocabulary (~2 000 words: nouns from Broden / a curated concept word list, cached to a file for reproducibility) — both with prompt ensembling; L2-normalized.

```yaml
priority: MUST-RUN
gpu_hours: 0.05
kind: text-embedding-cache
```

**Cmd**:
```bash
CUDA_VISIBLE_DEVICES=${GPU_LIST:-1} \
  python scripts/05_text_embed.py \
    --clip_arch ViT-B-32 --clip_pretrained openai \
    --clip_ckpt_cache ${MODEL_DIR}/clip-vit-b32 \
    --vocabs data/vocab/imagenet1k_classes.txt data/vocab/broden_concepts.txt \
    --prompt_ensemble openai_7 \
    --out runs/M5_text_embeddings/text_embeddings.h5
```
Expected output: HDF5 with `vocab_name → {word_id → text_embedding (512-dim, L2-normalized)}`.
**Pass criterion**: 1000 class-name rows + ≥ 1500 open-vocab rows.

---

### M6_C1_last_layer — Last-layer concept-purity (P1a)

**Covers claim(s)**: **C1**.
**Depends on**: `[M4_v_c, M5_text_embeddings]`.
**Purpose**: For each last-layer unit *c* (tied to an ImageNet class), compute top-1 concept among the ImageNet-1k class-name vocab by cos(v_c, text). Report concept-purity = fraction whose top-1 equals *c*'s ground-truth class. Compare against random-input baseline (recomputed once with the shuffled R_c).

```yaml
priority: MUST-RUN
gpu_hours: 0.02
kind: predicate-eval
predicate_id: P1a
grid:                                    # sweep k, fix pool=mean for main; also record mean at k=16 for headline
  k: [1, 4, 16, 64, 256]
  pool: [mean]
method_sensitive: [metric]               # /auto-experiment Phase 1.5 may bind an alternative metric-normalization
```

**Cmd (template)**:
```bash
python scripts/06_c1_last_layer.py \
  --v_c runs/M4_v_c/v_c__k${k}__${pool}.h5 \
  --text runs/M5_text_embeddings/text_embeddings.h5 --vocab imagenet1k_classes \
  --last_layer_index runs/M2_reference_sets/refsets.h5 \
  --random_input_baseline runs/M4_v_c/v_c__k${k}__${pool}__random.h5 \
  --out runs/M6_C1_last_layer/purity__k${k}__${pool}.json
```

**Expected output**: JSON with `{k, pool, top1_purity, top5_purity, random_baseline_top1_purity, delta_pure, p_value_paired}`.
**Pass criterion**: `delta_pure = top1_purity − random_baseline_top1_purity > 0` at `p < 0.05`, and `top1_purity` within the CLIP-Dissect published range for ResNet-50 last-layer (Oikarinen & Weng 2023: ~ 66-76 % depending on vocab).
**Refute if**: `delta_pure ≤ 0` OR `p ≥ 0.05` OR `top1_purity` an order of magnitude below the CLIP-Dissect range.

---

### M7_C1_hidden — Hidden-layer matched-control test (P1b + P1c)

**Covers claim(s)**: **C1**.
**Depends on**: `[M4_v_c, M5_text_embeddings]`.
**Purpose**: For each sampled hidden component *c*, take its top-1 concept `w1` from the open vocab, and its **matched-control** `w2` = second-best concept. Test paired: cos(v_c, w1) − cos(v_c, w2). Positive gap Δ_sep with `p < 0.05`. Report *k*-sensitivity (P1c: is Δ_sep monotone-nondecreasing in *k* with an early plateau?).

```yaml
priority: MUST-RUN
gpu_hours: 0.05
kind: predicate-eval
predicate_id: P1b_P1c
grid:
  k: [1, 4, 16, 64, 256]
  pool: [mean]
method_sensitive: [metric]
```

**Cmd (template)**:
```bash
python scripts/07_c1_hidden.py \
  --v_c runs/M4_v_c/v_c__k${k}__${pool}.h5 \
  --text runs/M5_text_embeddings/text_embeddings.h5 --vocab broden_concepts \
  --components_layer3 runs/M2_reference_sets/refsets.h5:layer3_sampled \
  --components_layer4 runs/M2_reference_sets/refsets.h5:layer4_sampled \
  --out runs/M7_C1_hidden/sep__k${k}__${pool}.json
```

**Expected output**: JSON with `{k, pool, layer, delta_sep_mean, delta_sep_ci, p_value_paired, plateau_k_est}`.
**Pass criterion (P1b)**: `delta_sep_mean > 0` and `p < 0.05` for at least layer4 at `k = 16`.
**Pass criterion (P1c)**: `delta_sep_mean` monotone-nondecreasing in *k*; `plateau_k_est ≤ 16` (small-*k* plateau).
**Refute if**: either fails.

---

### M8_C2_queryability — Text-query → component retrieval (P2a)

**Covers claim(s)**: **C2 (H2a)**.
**Depends on**: `[M4_v_c, M5_text_embeddings]`.
**Purpose**: For each ImageNet-1k class name text query *t*, rank all last-layer components by cos(v_c, e_t) and record the rank of the ground-truth class-tied component. Report MRR and recall@10. Compare against permutation baseline (component labels shuffled 1 000 times, bootstrap CI).

```yaml
priority: MUST-RUN
gpu_hours: 0.05
kind: predicate-eval
predicate_id: P2a
grid:
  pool: [mean, act_weighted_mean, max, medoid]  # pooling ablation for C2
  k: [16]                                        # pinned to expected plateau; P1c may revise
method_sensitive: [metric]
```

**Cmd (template)**:
```bash
python scripts/08_c2_queryability.py \
  --v_c runs/M4_v_c/v_c__k${k}__${pool}.h5 \
  --text runs/M5_text_embeddings/text_embeddings.h5 --vocab imagenet1k_classes \
  --n_permute 1000 \
  --seed 42 \
  --out runs/M8_C2_queryability/mrr__k${k}__${pool}.json
```

**Expected output**: JSON with `{k, pool, mrr, recall_at_10, permutation_mrr_ci_95, significant}`.
**Pass criterion**: `mrr` > permutation CI upper bound at `k = 16`, `pool = mean`; qualitative sign preserved across all 4 pooling operators (P2d).
**Refute if**: `mrr` ≤ permutation CI upper bound at the main setting.

---

### M9_C2_stability — Within-component stability (P2b)

**Covers claim(s)**: **C2 (H2b)**.
**Depends on**: `[M4_v_c]` (specifically needs disjoint-half re-pooling, computed on the fly here).
**Purpose**: For each component, split its top-`2k = 32` reference inputs into disjoint halves A and B, pool each into `v_c^A`, `v_c^B` (per operator), compute cosine. Median across the sampled 2000 components; bootstrap CI.

```yaml
priority: MUST-RUN
gpu_hours: 0.05
kind: predicate-eval
predicate_id: P2b
grid:
  pool: [mean, act_weighted_mean, max, medoid]
  half_k: [16]                                   # each half has k=16 → total 2k=32 top inputs used
method_sensitive: [metric]
```

**Cmd (template)**:
```bash
python scripts/09_c2_stability.py \
  --refsets runs/M2_reference_sets/refsets.h5 \
  --clip_emb runs/M3_clip_embeddings/clip_val_embeddings.h5 \
  --half_k ${half_k} --pool ${pool} \
  --out runs/M9_C2_stability/stability__hk${half_k}__${pool}.json
```

**Expected output**: JSON with `{half_k, pool, median_cosine, ci_95, tau_stable_threshold, passes}`.
**Pass criterion**: `median_cosine ≥ τ_stable = 0.5` at `pool = mean`; qualitatively consistent across all 4 pooling operators.
**Refute if**: median below threshold at pool = mean.

---

### M10_C2_separation — Within- vs between-concept cosine gap (P2c)

**Covers claim(s)**: **C2 (H2c)**.
**Depends on**: `[M4_v_c]`.
**Purpose**: On the last layer, "same-concept" pairs = the (1000 choose 2) pairs where both components are the same class (trivially: n=1000 — no same-class pairs, so we build the same-class pool from **stratified nearest-CLIP-text-neighbor class groups** — e.g. dog breeds vs. across-domain classes). "Different-concept" pairs = a random subsample. Report within-vs-between cosine gap + Cohen's *d*. On layer4, "same-concept" is defined by top-1 concept identity from M7.

```yaml
priority: MUST-RUN
gpu_hours: 0.05
kind: predicate-eval
predicate_id: P2c
grid:
  pool: [mean, act_weighted_mean, max, medoid]
  k: [16]
method_sensitive: [metric, n_pairs]              # /auto-experiment Phase 1.5 may bind n_pairs to a smaller / larger sample
```

**Cmd (template)**:
```bash
python scripts/10_c2_separation.py \
  --v_c runs/M4_v_c/v_c__k${k}__${pool}.h5 \
  --text runs/M5_text_embeddings/text_embeddings.h5 --vocab imagenet1k_classes \
  --same_concept_groups data/eval/dogbreed_groups.json \
  --n_pairs 10000 \
  --seed 42 \
  --out runs/M10_C2_separation/sep__k${k}__${pool}.json
```

**Expected output**: JSON with `{k, pool, mean_within_cos, mean_between_cos, gap, cohens_d, p_value_ttest}`.
**Pass criterion**: `gap > 0`, `cohens_d ≥ 0.5`, `p < 0.05` at pool = mean; consistent sign across all 4 pool operators (P2d).
**Refute if**: gap ≤ 0 or `d < 0.5` at pool = mean.

---

### M11_layer_granularity — Layer-granularity summary (spanning C1 + C2)

**Covers claim(s)**: **C1 + C2** (cross-layer view).
**Depends on**: `[M6_C1_last_layer, M7_C1_hidden, M8_C2_queryability, M10_C2_separation]`.
**Purpose**: Aggregate M6-M10 by layer (`fc`, `layer4`, `layer3`); produce a single per-layer results table (`P1a, P1b, P2a, P2c` × 3 layers). Flags any layer where all four fail — a finding, not a claim failure.

```yaml
priority: MUST-RUN
gpu_hours: 0.02
kind: aggregation
```

**Cmd**:
```bash
python scripts/11_layer_summary.py \
  --inputs runs/M6_C1_last_layer runs/M7_C1_hidden runs/M8_C2_queryability runs/M10_C2_separation \
  --out runs/M11_layer_granularity/layer_summary.json
```

---

### M12_cross_model_verify — Cross-model transfer (verify-stage swap)

**Covers claim(s)**: **C1 + C2** — the *shared-coordinate-system* implication of "joint image-text semantic space".
**Depends on**: `[M0_setup]` (independent of the main-experiment ResNet-50 caches).
**Purpose**: Repeat M1 → M3 → M4(mean, k=16) → M8 for each swap model in `{ViT-B/16, VGG-16, EfficientNet-B0}` (last layer only for cost). Report per-model P2a (MRR / recall@10) vs. permutation baseline.

**Note**: `/auto-verify` may re-run this as its own "model swap" variant; the milestone here is the *in-plan* cross-model check that supports P3 in `FINAL_PROPOSAL.md`.

```yaml
priority: SHOULD-RUN                     # skip only if the 10 h budget is at risk
gpu_hours: 1.5                           # ~0.5 h per swap model
kind: cross-model-transfer
grid:
  inspected_model: [vit_b_16, vgg_16, efficientnet_b0]   # ImageNet-pretrained
  pool: [mean]
  k: [16]
depends_on: [M0_setup]                   # NOT M1 — each swap model has its own activation cache
method_sensitive: [metric, sites]
```

**Cmd (template)**:
```bash
CUDA_VISIBLE_DEVICES=${GPU_LIST:-2} \
  python scripts/12_cross_model.py \
    --inspected_model ${inspected_model} \
    --data_dir ./data/imagenet \
    --model_dir ${MODEL_DIR} \
    --clip_arch ViT-B-32 --clip_pretrained openai \
    --last_layer_only true \
    --k ${k} --pool ${pool} \
    --out runs/M12_cross_model_verify/${inspected_model}__k${k}__${pool}.json
```

**Expected output (template)**: JSON per swap model with `{inspected_model, mrr, recall_at_10, permutation_mrr_ci_95, significant}`.
**Pass criterion**: MRR significantly above permutation baseline on **each** swap model.
**Refute if**: on all three swap models the MRR is at permutation floor.

---

### M13_final_report — Aggregated results table + human-readable summary

**Covers claim(s)**: **C1 + C2** (unified report).
**Depends on**: `[M6_C1_last_layer, M7_C1_hidden, M8_C2_queryability, M9_C2_stability, M10_C2_separation, M11_layer_granularity, M12_cross_model_verify]`.
**Purpose**: Produce the FINAL_PROPOSAL §6 results table + narrative, ready for `/result-to-claim` to judge whether C1 and C2 are supported.

```yaml
priority: MUST-RUN
gpu_hours: 0.02
kind: aggregation
```

**Cmd**:
```bash
python scripts/13_final_report.py \
  --results_root runs/ \
  --claims idea-stage/IDEA_REPORT.md \
  --proposal refine-logs/FINAL_PROPOSAL.md \
  --out runs/M13_final_report/results_table.json \
  --out_md runs/M13_final_report/FINAL_RESULTS.md
```

**Expected output**: `runs/M13_final_report/results_table.json` + `runs/M13_final_report/FINAL_RESULTS.md`.

---

## Run order and GPU-hour budget

| # | Milestone | Depends on | GPU-h (est.) | Cumulative |
|---|-----------|-----------|--------------|------------|
| 1 | M0_setup | — | 0.3 | 0.3 |
| 2 | M1_activations | M0 | 0.2 | 0.5 |
| 3 | M2_reference_sets | M1 | 0.1 | 0.6 |
| 4 | M3_clip_embeddings | M0,M2 | 0.7 | 1.3 |
| 5 | M4_v_c (20-cell grid) | M3 | 0.05 | 1.35 |
| 6 | M5_text_embeddings | M0 | 0.05 | 1.4 |
| 7 | M6_C1_last_layer (5-cell k-sweep) | M4,M5 | 0.02 | 1.42 |
| 8 | M7_C1_hidden (5-cell k-sweep) | M4,M5 | 0.05 | 1.47 |
| 9 | M8_C2_queryability (4-cell pool) | M4,M5 | 0.05 | 1.52 |
| 10 | M9_C2_stability (4-cell pool) | M4 | 0.05 | 1.57 |
| 11 | M10_C2_separation (4-cell pool) | M4 | 0.05 | 1.62 |
| 12 | M11_layer_granularity | M6..M10 | 0.02 | 1.64 |
| 13 | M12_cross_model_verify (3-cell swap) | M0 | 1.5 | 3.14 |
| 14 | M13_final_report | all | 0.02 | ≈ **3.2 GPU-h** |

**Total main-experiment estimate: ≈ 3-4 GPU-h**, leaving ≥ 6 GPU-h for `/auto-verify` variants and iteration.

Milestones 7-11 (M6-M10) all read the same cached `runs/M4_v_c/*` and `runs/M5_text_embeddings/*` — they are I/O-bound and parallelizable across the 5 allowed GPUs; the queue can run them concurrently.

---

## Priority tags

- **MUST-RUN** (fail-fast): M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11, M13 — every one of them is on a claim's critical path.
- **SHOULD-RUN**: M12 (cross-model verify) — skip only if the M0-M11 chain runs long enough to threaten the 10 h ceiling; `/auto-verify` will otherwise cover cross-model swap.

## Decision gates (per claim)

- **C1 supported** iff **P1a passes on last layer** AND **P1b passes on ≥ layer4** AND **P1c shows an early plateau (`k*_plateau ≤ 16`)**.
- **C2 supported** iff **P2a passes at pool=mean** AND **P2b median ≥ 0.5 at pool=mean** AND **P2c positive gap with d ≥ 0.5 at pool=mean** AND **P2d qualitative consistency across all 4 pooling operators** AND (soft) **P3 cross-model transfer passes on ≥ 2 of 3 swap models**.
- **Both refuted**: report a genuine negative finding — the claim, as stated in `task.md`, does not hold under this operationalization; the refutation is by matched-control, statistical-test, not by absence of evidence.
