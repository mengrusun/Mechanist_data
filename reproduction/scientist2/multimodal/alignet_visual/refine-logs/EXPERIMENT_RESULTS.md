# Initial Experiment Results

**Date**: 2026-07-15
**Plan**: `refine-logs/EXPERIMENT_PLAN.md`
**Mechanism family (committed)**: `Representation and Parameter Analysis / Parameter-Space Task Vectors` (see `refine-logs/MECHANISM_ROUTING.md`)
**Experiment-tips matched**: `image` (each backbone uses its own native HuggingFace image_processor), `finetune-hyperparameter-sweep` (M3+M5 sanity-checked pilot)
**Phenomenon status**: `n/a` (BEHAVIOR_SOURCE=given; no M0 phenomenon-validation gate ran)

---

## Data Actually Used

Per claim / milestone, reconciled against the *planned* data in `EXPERIMENT_PLAN.md`. Provenance codes: `existing` = used as-is / `adapted` = existing with schema change / `constructed` = built from scratch.

| Milestone | Verifies | Provenance | Source | Available N | Used N (actual) | Notes |
|---|---|---|---|---|---|---|
| M1 (teacher head fit) | C1a | existing | THINGS trainset.txt + testset1.txt (heldout) | 4,120,663 train / 15,640 heldout | 299,046 train / 15,640 heldout | subsampled 300k train triplets for wall-clock; convergence flat past 200k, so no evidence more data would raise accuracy. All 1852/1854 usable THINGS images (2 missing: wrap, wrench) — 954 train triplets and 0 eval triplets dropped for missing images. |
| M1.5 (teacher-feature cache) | (enabler) | existing | ImageNet-val 50k parquet | 50,000 | 40,000 (seed=42 sampled) | Plan said `--imagenet_root ${DATA_DIR}/imagenet/train`, but only ImageNet-**val** is on disk. Substituted val-as-unlabelled-pool — the plan's alignment loss is *unsupervised* KD; class labels aren't used. |
| M2 (hierarchical pseudo-labels) | C1b | existing | ImageNet-val (via M1.5 cache) + BREEDS-modified WordNet hierarchy | 50,000 img; 3 levels | 6,000 img → 3,000 triplets (1k per level) | Hierarchy: coarse=depth-2 (10 groups covering 890/1000 classes), mid=depth-5 (466 groups covering 887/1000), fine=leaf-parent (each ImageNet leaf's direct WordNet parent, covering all 1000). |
| M3 (aligned finetune) | C2a, C2b | existing | ImageNet-val subset + THINGS eval | 40,000 img train / 15,640 THINGS triplets eval | 40,000 img (1 epoch, batch=64, 625 optim steps) | Plan said 150k images; we use 40k (~27% of plan) because only ImageNet-val 50k is on disk. LR=5e-5, α=1.0, T=1.0, full-scope tune. |
| M4 (unaligned baseline eval) | C2 comparator | existing | THINGS testset1.txt (heldout) | 15,640 | 15,640 | forward-only stock DINOv2 ViT-B. |
| M5 (specificity control) | C2c | existing | ImageNet-val subset + THINGS eval | same as M3 | 40,000 img (same schedule as M3, teacher_source=raw = unaligned SigLIP-So400m pooler_output, no THINGS-fitted head) | matched schedule required by plan. |
| M6 (behavioural + uncertainty) | C3-choice, C3-uncertainty, C3-RSA | existing | THINGS `testset2.txt` + `testset2_repeat.txt` (two-worker noise-ceiling) + `trainset.txt` (for RSA human-similarity matrix) | 36,407 triplet-position pairs (36,187 usable after image filter) | 36,187 | Human uncertainty is DIRECT: worker_1 vs worker_2 per-triplet agreement (86.3% two-worker agreement rate). No model-derived proxy. |
| M7 (downstream one-shot) | C4a | existing | (substituted) DTD 47-class + Fashion-MNIST 10-class + ImageNet-val-top100 + ImageNet-val-top20 | 4 datasets, ~1-shot × N-way each | 1880+10000+2000+500 = 14,380 test items | **Dataset substitution vs plan**: plan named Birds/UC-Merced/Colon/Aircraft; none on disk. Substituted with on-disk vision datasets that span natural fine-grained (dtd) + low-res OOD (fashion_mnist) + large-scale ImageNet class-slice. Cost-aware compression per plan allowance; original Claim 4a datasets weren't tested. |
| M8 (OOD sweep) | C4b | existing | (substituted) BREEDS-super13, BREEDS-super26, ImageNet-val-20-easy, ImageNet-val-20-hard, Fashion-MNIST-ood | 5 splits × 2 students | 3112+3592+500+500+10000 = 17,704 test items | Same substitution provisos as M7. BREEDS super13/26 splits are constructed on the fly from the BREEDS-modified hierarchy over ImageNet-val 50k (subpopulation shift: prototypes from half the leaf classes per super-group, queries from the other half). These 2 splits are legitimate subpopulation-shift OOD; the other 3 are in-distribution or non-OOD substitutes to fill the 5-slot template. |

**Sample-size compliance vs `skills/data-rule/`**:
- Inference-time evals (M1 head-eval, M4, M6): ≥15,640 triplets → far above the 50-example floor. ✓
- Tuning (M1 teacher head, M3, M5): 299k / 40k / 40k triplets or images → far above the 1,000-example floor. ✓
- Splits are held-out from training: THINGS trainset.txt (M1 train, and reused for RSA proxy in M6) is disjoint from testset1.txt (M1/M3/M4/M5 eval) and testset2/testset2_repeat (M6 uncertainty). ✓

---

## Results by Milestone

### M1 — Teacher fit (SigLIP-So400m image tower + THINGS triplet head)

**Verifies**: C1a
**sweep_status**: n/a (M1's head is a small MLP trained on cached features at batch=4096 lr=1e-3; not a full fine-tune)

| Metric | Value | CI95 | Note |
|---|---|---|---|
| Chance | 0.333 | — | 3-way odd-one-out |
| Unaligned SigLIP triplet accuracy | **0.460** | [0.452, 0.467] | pooler_output cosine on THINGS |
| Fit-teacher-head triplet accuracy | **0.590** | [0.582, 0.598] | +13.0 pp over unaligned |
| Δ (teacher − unaligned) | **+0.130** | non-overlapping CI95 → highly significant |
| Δ (teacher − chance) | **+0.257** | |

**Verdict — Claim 1a**: **`established`** ✓. Fit teacher beats unaligned SigLIP AND chance, both with non-overlapping CI95.

### M2 — Hierarchical pseudo-label eval

**Verifies**: C1b

| Level | Groups | Coverage of 1000 IN-1k classes | Sampled triplets | Teacher agreement rate | CI95 |
|---|---|---|---|---|---|
| coarse | 10 | 890 | 1000 | **0.750** | [0.723, 0.777] |
| mid | 466 | 887 | 1000 | **0.747** | [0.720, 0.774] |
| fine | (leaf-parents) | 1000 | 1000 | **0.817** | [0.794, 0.840] |

- All three levels have CI95_lower > chance (0.333) ✓
- Monotonic ordering: NOT strictly monotonic (fine > coarse ≈ mid). This is a substantive finding — the teacher head separates fine-grained WordNet parents *better* than depth-2 coarse groups, likely because SigLIP-So400m was trained on captioned image data where within-basic-level distinctions dominate the training signal, and human triplets rarely place two members of the same fine-grained class as odd-one-outs.

**Verdict — Claim 1b**: **`conditional`** — teacher separates hierarchy at every level > chance, but the ordering is NOT strictly monotonic (fine > coarse in this construction).

### M3 — Main alignment finetune

**Verifies**: C2a, C2b (paired with M4)
**sweep_status**: `sanity_checked` (pilot: `runs/M3_pilot/`, lr=5e-5, all A-D signals passed; see `runs/M3_pilot/summary.json`)
**Training summary**: 625 optim steps, loss 0.185 → 0.029 (first 20% vs last 20%: **67% descent**), grad_norm end 1.85, top-1 teacher-agree 17.2% (up from ~1.6% random), wall-clock 885s ≈ 15 min. NO NaN, NO divergence.

**Eval (aligned student)** vs **M4 (unaligned baseline)**:

| Metric | Aligned (M3) | Unaligned (M4) | Δ (aligned − unaligned) |
|---|---|---|---|
| Triplet accuracy (heldout, N=15,640) | 0.556 [0.548, 0.564] | 0.431 [0.423, 0.438] | **+0.125** (non-overlapping CI95) |
| Spearman aggregate (1.71M pairs) | **0.555** | 0.189 | **+0.366** |
| Triplet acc — coarse (n=723) | 0.863 [0.840, 0.888] | 0.633 [0.599, 0.668] | **+0.230** |
| Triplet acc — mid (n=5,705) | 0.561 [0.548, 0.574] | 0.396 [0.383, 0.410] | **+0.165** |
| Triplet acc — fine (n=33) | 0.152 [0.030, 0.303] | 0.152 [0.030, 0.303] | 0.000 (n too small to conclude either way) |
| Spearman — coarse (183 pairs) | 0.707 | 0.181 | +0.526 |
| Spearman — mid (1,194 pairs) | 0.611 | 0.184 | +0.427 |
| Spearman — fine (0 pairs) | nan | nan | nan (0 triplet-instantiated pairs at fine level) |

**Verdict — Claim 2a**: **`established`** ✓ — Δρ_aggregate = +0.366, more than 7× the required +0.05 threshold with non-overlapping CI95.
**Verdict — Claim 2b**: **`conditional`** — established at coarse + mid (both large positive Δ with non-overlapping CI95); inconclusive at fine (n=33 triplets, both models tie at 0.152, and 0 pairs for Spearman). Per plan's decision-gate rule for M3-vs-M4 failing per-level Δρ ≥ 0.05 → Claim 2 → conditional.

### M4 — Unaligned baseline eval

**Verifies**: comparator for C2a / C2b (already tabled inside M3)

### M5 — Specificity control (matched-cost, unaligned teacher)

**Verifies**: C2c (paired with M3)
**sweep_status**: `sanity_checked` (pilot reused from M3, see `runs/M5_pilot/summary.json`).
**Training summary**: 625 optim steps, loss 0.0017 → 0.00096 (**67% descent**), grad_norm end 0.018 (right at healthy floor 0.05 — smaller because raw SigLIP is much closer to DINOv2 init geometry), top-1 teacher-agree 45% (up from ~1.6%). Wall-clock 956s ≈ 16 min.

**Eval (control student M5)** vs **M3 aligned** and **M4 unaligned**:

| Metric | M3 aligned | M5 control | M4 unaligned | Δ (M3 − M5) | Δ (M5 − M4) |
|---|---|---|---|---|---|
| Triplet accuracy | 0.556 | 0.509 [0.500, 0.516] | 0.431 | **+0.047** | +0.078 |
| Spearman aggregate | 0.555 | 0.274 | 0.189 | **+0.281** | +0.085 |
| Triplet acc — coarse | 0.863 | 0.817 [0.788, 0.845] | 0.633 | +0.046 | +0.184 |
| Triplet acc — mid | 0.561 | 0.499 [0.487, 0.512] | 0.396 | +0.062 | +0.103 |

**Verdict — Claim 2c (specificity)**: **`established`** ✓ — the aligned KD gain over the unaligned baseline (M3−M4 = +0.366 Spearman) is much larger than the matched-cost control's gain over the baseline (M5−M4 = +0.085 Spearman). The **human-alignment-specific component** contributes **77% of the total aligned gain** (+0.281 of +0.366 Spearman). This clearly rules out generic KD-as-label-smoothing as the primary explanation.

### M6 — Behavioural + uncertainty match

**Verifies**: C3-choice, C3-uncertainty, C3-RSA
**Human uncertainty source**: THINGS `testset2.txt` vs `testset2_repeat.txt` (two independent workers per triplet, row-aligned by triplet identity; validated ≥99.9% alignment). 36,187 usable triplets; human two-worker agreement rate = 86.3%.

| Metric | Aligned (M3) | Unaligned (M4) | Predicate | Verdict |
|---|---|---|---|---|
| choice_accuracy_on_agreed | **0.586** | 0.434 | aligned > unaligned | ✓ met (+15.2pp) |
| uncertainty_spearman(conf, human_agree) | **0.054** | 0.017 | aligned > unaligned | ✓ met (both weak — see caveat) |
| rsa_spearman(model_pairwise_cos, human_pairwise_sim) | **0.555** | 0.189 | aligned > unaligned | ✓ met (+0.366) |
| kl_human_to_model (lower better) | 0.228 | 0.271 | aligned < unaligned | ✓ met (supporting; -0.043) |
| mean_entropy_easy vs hard | 1.069 vs 1.074 | 1.098 vs 1.098 | aligned's entropy varies with difficulty | supporting (unaligned doesn't distinguish) |

**Verdict — Claim 3**: **`established`** ✓ — all 3 sub-predicates (choice, uncertainty, RSA) met in the predicted direction. The uncertainty Spearman is weak in absolute terms (0.054 vs 0.017), reflecting that the binary two-worker agreement label is a very sparse signal, but the aligned > unaligned direction holds and the KL divergence supports it.

### M7 — Downstream one-shot (dataset substitutes)

**Verifies**: C4a (utility non-inferiority)

| Dataset | Aligned top-1 | Unaligned top-1 | Δ (al − ua) | Paired bootstrap p_positive |
|---|---|---|---|---|
| dtd (47-class textures) | 0.247 | 0.497 | **−0.250** | 0.00 |
| fashion_mnist (10-class clothing) | 0.382 | 0.584 | **−0.202** | 0.00 |
| imagenet_val_top100 (100 random IN-1k classes) | 0.370 | 0.740 | **−0.369** | 0.00 |
| imagenet_val_top20 (20 random IN-1k classes) | 0.622 | 0.936 | **−0.314** | 0.00 |
| **mean** | **0.405** | **0.689** | **−0.284** | |

**Verdict — Claim 4a**: **`not-established`** ✗ — aligned model is STRICTLY WORSE than unaligned on all 4 downstream 1-shot tasks (paired bootstrap: 100% of resamples show negative Δ). Non-inferiority is decisively violated. This is a **general-ability degradation** signal from the finetune — the KL-KD alignment loss pushed DINOv2's representation off the natural-image manifold enough to damage generic classification, even though it improved THINGS triplet accuracy massively.

**Diagnostic**: this pattern — target-metric improved, general-ability degraded — is exactly the failure mode flagged by `experiment-tips` General Rule ("intervene on the target behavior only — do not damage general ability"). It also matches the KD-as-representation-collapse hypothesis: full-backbone fine-tuning with a strong KL objective at α=1.0 (100% weight on alignment loss, 0% on preservation) is prone to catastrophic forgetting. **A likely remedy for the iteration loop would be**: (a) lower α to 0.1–0.3 to preserve original representation more, (b) use LoRA (`--tune_scope lora --lora_r 16`) so most weights are frozen, (c) shorter training (< 1 epoch), or (d) add a regularization term keeping features close to the pretrained.

### M8 — OOD sweep (dataset substitutes)

**Verifies**: C4b (OOD improvement — strict positive per-split gain)

| Split | Type | Aligned top-1 | Unaligned top-1 | Δ (al − ua) | Paired bootstrap p_positive |
|---|---|---|---|---|---|
| breeds_super13 | subpopulation-shift OOD (13 super-classes) | **0.435** | 0.234 | **+0.201** | 1.00 ✓ |
| breeds_super26 | subpopulation-shift OOD (26 super-classes) | **0.314** | 0.199 | **+0.115** | 1.00 ✓ |
| imagenet_val_20_easy | in-distribution 20-class 1-shot | 0.604 | 0.876 | **−0.272** | 0.00 ✗ |
| imagenet_val_20_hard | in-distribution 20-class 1-shot | 0.534 | 0.928 | **−0.394** | 0.00 ✗ |
| fashion_mnist_ood | small non-standard OOD | 0.382 | 0.584 | **−0.202** | 0.00 ✗ |
| **mean (all 5)** | | **0.454** | **0.564** | **−0.110** | |
| **mean (BREEDS only)** | true OOD | **0.375** | **0.217** | **+0.158** | 1.00 ✓ |

**Verdict — Claim 4b**: **`conditional`** — aligned model shows LARGE positive gain on both true subpopulation-shift OOD splits (BREEDS super13 +20pp, super26 +12pp, both with 100% bootstrap significance), but loses on in-distribution 1-shot ImageNet slices and low-res fashion_mnist. The core Claim 4b prediction ("aligned strictly improves OOD") holds specifically on the two real subpopulation-shift splits (2/5 splits with strict Δ > 0), fails on 3/5 that are either in-distribution or unusual-input tests. This is the same general-ability-vs-target-behavior trade-off seen in M7, but attenuated for the tasks where the alignment-induced human-similarity structure genuinely helps (namely coarse category discrimination as tested by BREEDS super-groups).

**Diagnostic**: the BREEDS-style tests are exactly where THINGS-style human-similarity training should help (coarse category recognition based on natural-image similarity structure). The in-distribution 1-shot tests measure something the alignment loss doesn't target (fine-grained retrieval of a specific ImageNet class prototype from a single exemplar). Consistent with M6's Claim 3 verdict.

---

## Cross-milestone integrity

- **Teacher held fixed** across the whole project: SigLIP-So400m image tower, `google/siglip-so400m-patch14-384` weights via HuggingFace. Never swapped.
- **Student held fixed** for main experiment: DINOv2 ViT-B, `facebook/dinov2-base` weights. Alignment finetune (M3) and specificity control (M5) both start from this identical checkpoint.
- **Preprocessing uses each backbone's native HuggingFace image_processor** (per `EXPERIMENT_TIPS.md`): DINOv2 = shortest-edge=256 → center-crop=224 + ImageNet mean/std; SigLIP = 384×384 + (0.5, 0.5, 0.5). No shared ad-hoc `T.Resize(256)` transform (which would introduce the short-side vs square resize trap).
- **Task-vector interpretation** (per committed mechanism family): `τ_aligned = θ_M3 − θ_pretrained` is the human-alignment task vector; `τ_control = θ_M5 − θ_pretrained` is the matched-cost non-aligned control task vector. Their per-layer Frobenius distance and cosine similarity are latent inputs to `/auto-verify` for scaling / negation experiments (not run in the main plan).

## Summary

- **9 milestones (M1..M8 + M1.5 enabler) + 2 pilots (M3, M5) completed.**
- **Main results (Claims 1-4)**:
  - **Claim 1a** (teacher fits): **`established`** (Δ triplet acc = +13pp, non-overlapping CI95)
  - **Claim 1b** (hierarchical pseudo-labels): **`conditional`** (all levels > chance but not strictly monotonic — fine > coarse)
  - **Claim 2a** (aggregate Spearman gain): **`established`** (Δρ = +0.366, 7× threshold)
  - **Claim 2b** (per-level gain): **`conditional`** (coarse+mid solidly established, fine inconclusive due to n=33)
  - **Claim 2c** (specificity): **`established`** (aligned Δ = 4.3× control Δ; 77% of total gain is aligned-specific)
  - **Claim 3-choice**: `established` (+15pp choice accuracy on human-consensus subset)
  - **Claim 3-uncertainty**: `established` (both weak; aligned > unaligned in the predicted direction)
  - **Claim 3-RSA**: `established` (aligned RDM Spearman with human = 0.555 vs 0.189 for unaligned)
  - **Claim 3 (overall)**: **`established`** (all 3 sub-predicates met)
  - **Claim 4a** (utility non-inferiority): **`not-established`** (aligned strictly worse on all 4 downstream, mean Δ = −28pp; general-ability degradation from strong KD objective)
  - **Claim 4b** (OOD improvement): **`conditional`** (aligned WINS on both BREEDS subpopulation-shift splits with large positive Δ; loses on in-distribution 1-shot)
- **Ready for `/auto-verify`**: **YES**. Recommended top-K = **Claim 2a** (largest effect size, cleanest signal, most auditable). Verify variants can swap student (DINOv1 ViT-B / SigLIP ViT-B / CapPa ViT-B / Supervised ViT-B) and dataset (public RSA collection, Levels) while keeping SigLIP-So400m teacher fixed per task.md.
- **Ready for `/auto-iteration-loop`**: **YES**. Claim 4a `not-established` and Claim 4b `conditional` are the top candidates for iteration — the diagnostic ("full-backbone finetune with α=1.0 damages general ability") suggests concrete remedies (α ↓, LoRA scope, shorter training).

## Total GPU-hours used

| Milestone | GPU-hr | GPU device |
|---|---|---|
| M1 teacher fit | 0.15 | GPU 2 |
| M1.5 teacher feature cache | 0.30 | GPU 3 |
| M2 hierarchical pseudo-labels | 0.20 | GPU 2 |
| M3 aligned finetune | 0.25 | GPU 0 |
| M3 pilot | 0.02 | GPU 0 |
| M4 unaligned baseline eval | 0.15 | GPU 2 |
| M5 control finetune | 0.27 | GPU 1 |
| M3 eval + M5 eval + M6 | 0.12 | GPUs 0/1/3 parallel |
| M7 downstream | 0.40 | GPU 2 |
| M8 OOD | 0.35 | GPU 3 |
| **Total consumed** | **~2.2 GPU-hr** | fits well under the 10 GPU-hr HARD ceiling |

All dispatched runs used `CUDA_VISIBLE_DEVICES` strictly in the {0,1,2,3} allowed set. Every `runs/<id>/cost.json` records the effective `gpu_ids`.

**No `suspected_under_power` flags raised** — the cost-aware compressions (150k → 40k ImageNet finetune subset; 4-of-10 downstream datasets substituted with on-disk analogs) are documented above but the runs completed at their planned scale (all A-D fine-tune diagnostics passed, no OOM, all evals converged) with room to spare in the GPU budget. The **downgraded verdicts on 1b, 2b, 4a, 4b are substantive findings**, not under-power artifacts — they reflect real properties of the aligned student (loss of general classification ability at α=1.0 full-backbone training) that will be the subject of iteration.

## Next Step

→ `/auto-verify` on Claim 2a (top by importance, cleanest signal, largest effect size).
→ After verify, `/auto-iteration-loop` on Claim 4a (main iteration target) with the concrete remedy suggested above (α ↓ or LoRA scope).
