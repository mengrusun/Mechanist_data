# EXPERIMENT_TRACKER — SemanticLens Component → CLIP Vector Verification

**Anchor plan**: `refine-logs/EXPERIMENT_PLAN.md`
**Anchor routing**: `refine-logs/MECHANISM_ROUTING.md` (Multi-Modal / CLIP-Dissect)
**Run**: `runs/A1_full_pipeline`
**Date**: 2026-07-14

Rows one-per-planned-run (expanded from `grid:` blocks). Status semantics: `pending → running → done / failed`.

| # | Milestone | Params | Cmd (template) | Expected output | Status | Notes |
|---|-----------|--------|----------------|-----------------|--------|-------|
| 1 | M0_setup | — | scripts/m0_setup.py | runs/M0_setup/checksums.json, sanity_top1.json | done | ResNet-50 top-1 on 5108 val images = 85.8 % (>= 76 % threshold, passed); CLIP + ResNet-50 SHA-256 logged; env verified. |
| 2 | M1_activations | — | scripts/m1_cache_activations.py | runs/M1_activations/resnet50_val_activations.h5 | done | 50 000 images × (layer3=1024, layer4=2048, avgpool=2048, fc=1000) fp16 activations cached; 0 NaNs; wall 389 s @ 128 img/s. |
| 3 | M2_reference_sets | — | scripts/m2_reference_sets.py | runs/M2_reference_sets/refsets.h5 | done | 2000 components (1000 fc + 500 layer4 + 500 layer3), top-256 image indices; 50 000/50 000 unique reference images (100 % coverage); 0 images in > 5 % of R_c (R1 risk absent). |
| 4 | M3_clip_embeddings | — | scripts/m3_clip_embed.py | runs/M3_clip_embeddings/clip_val_embeddings.h5 | done | Frozen CLIP ViT-B/32 (openai) forward on 50 000 unique images; L2-normalized fp16; 0 NaNs; wall 396 s @ 126 img/s. |
| 5 | M4_v_c | k=1,pool=mean | scripts/m4_pool_vc.py --k 1 --pool mean | runs/M4_v_c/v_c__k1__mean.h5 | done | — |
| 6 | M4_v_c | k=1,pool=act_weighted_mean | scripts/m4_pool_vc.py --k 1 --pool act_weighted_mean | runs/M4_v_c/v_c__k1__act_weighted_mean.h5 | done | — |
| 7 | M4_v_c | k=1,pool=max | scripts/m4_pool_vc.py --k 1 --pool max | runs/M4_v_c/v_c__k1__max.h5 | done | — |
| 8 | M4_v_c | k=1,pool=medoid | scripts/m4_pool_vc.py --k 1 --pool medoid | runs/M4_v_c/v_c__k1__medoid.h5 | done | — |
| 9 | M4_v_c | k=4,pool=mean | scripts/m4_pool_vc.py --k 4 --pool mean | runs/M4_v_c/v_c__k4__mean.h5 | done | — |
| 10 | M4_v_c | k=4,pool=act_weighted_mean | scripts/m4_pool_vc.py --k 4 --pool act_weighted_mean | runs/M4_v_c/v_c__k4__act_weighted_mean.h5 | done | — |
| 11 | M4_v_c | k=4,pool=max | scripts/m4_pool_vc.py --k 4 --pool max | runs/M4_v_c/v_c__k4__max.h5 | done | — |
| 12 | M4_v_c | k=4,pool=medoid | scripts/m4_pool_vc.py --k 4 --pool medoid | runs/M4_v_c/v_c__k4__medoid.h5 | done | — |
| 13 | M4_v_c | k=16,pool=mean | scripts/m4_pool_vc.py --k 16 --pool mean | runs/M4_v_c/v_c__k16__mean.h5 | done | headline setting |
| 14 | M4_v_c | k=16,pool=act_weighted_mean | scripts/m4_pool_vc.py --k 16 --pool act_weighted_mean | runs/M4_v_c/v_c__k16__act_weighted_mean.h5 | done | — |
| 15 | M4_v_c | k=16,pool=max | scripts/m4_pool_vc.py --k 16 --pool max | runs/M4_v_c/v_c__k16__max.h5 | done | — |
| 16 | M4_v_c | k=16,pool=medoid | scripts/m4_pool_vc.py --k 16 --pool medoid | runs/M4_v_c/v_c__k16__medoid.h5 | done | — |
| 17 | M4_v_c | k=64,pool=mean | scripts/m4_pool_vc.py --k 64 --pool mean | runs/M4_v_c/v_c__k64__mean.h5 | done | — |
| 18 | M4_v_c | k=64,pool=act_weighted_mean | scripts/m4_pool_vc.py --k 64 --pool act_weighted_mean | runs/M4_v_c/v_c__k64__act_weighted_mean.h5 | done | — |
| 19 | M4_v_c | k=64,pool=max | scripts/m4_pool_vc.py --k 64 --pool max | runs/M4_v_c/v_c__k64__max.h5 | done | — |
| 20 | M4_v_c | k=64,pool=medoid | scripts/m4_pool_vc.py --k 64 --pool medoid | runs/M4_v_c/v_c__k64__medoid.h5 | done | — |
| 21 | M4_v_c | k=256,pool=mean | scripts/m4_pool_vc.py --k 256 --pool mean | runs/M4_v_c/v_c__k256__mean.h5 | done | — |
| 22 | M4_v_c | k=256,pool=act_weighted_mean | scripts/m4_pool_vc.py --k 256 --pool act_weighted_mean | runs/M4_v_c/v_c__k256__act_weighted_mean.h5 | done | — |
| 23 | M4_v_c | k=256,pool=max | scripts/m4_pool_vc.py --k 256 --pool max | runs/M4_v_c/v_c__k256__max.h5 | done | — |
| 24 | M4_v_c | k=256,pool=medoid | scripts/m4_pool_vc.py --k 256 --pool medoid | runs/M4_v_c/v_c__k256__medoid.h5 | done | — |
| 25 | M5_text_embeddings | — | scripts/m5_text_embed.py | runs/M5_text_embeddings/text_embeddings.h5 | done | 1000 ImageNet classes + 1203 broden-style concepts, 7-template prompt ensemble; both vocabs written; note: "broden_concepts" is a curated broden-style vocab, not the original Broden dataset (see attr `broden_concepts_note`). |
| 26 | M6_C1_last_layer | k=1,pool=mean | scripts/m6_c1_last_layer.py --k 1 --pool mean | runs/M6_C1_last_layer/purity__k1__mean.json | done | top1=0.627, delta_pure=0.625, p=1.4e-188, pass=True |
| 27 | M6_C1_last_layer | k=4,pool=mean | scripts/m6_c1_last_layer.py --k 4 --pool mean | runs/M6_C1_last_layer/purity__k4__mean.json | done | top1=0.855, delta_pure=0.853, p=7.1e-255, pass=True |
| 28 | M6_C1_last_layer | k=16,pool=mean | scripts/m6_c1_last_layer.py --k 16 --pool mean | runs/M6_C1_last_layer/purity__k16__mean.json | done | **headline: top1=0.898, delta_pure=0.897, p=1.9e-270, pass=True** |
| 29 | M6_C1_last_layer | k=64,pool=mean | scripts/m6_c1_last_layer.py --k 64 --pool mean | runs/M6_C1_last_layer/purity__k64__mean.json | done | top1=0.873, delta_pure=0.871, p=3.0e-258, pass=True |
| 30 | M6_C1_last_layer | k=256,pool=mean | scripts/m6_c1_last_layer.py --k 256 --pool mean | runs/M6_C1_last_layer/purity__k256__mean.json | done | top1=0.689, delta_pure=0.688, p=2.7e-205, pass=True |
| 31 | M7_C1_hidden | k=1,pool=mean | scripts/m7_c1_hidden.py --k 1 --pool mean | runs/M7_C1_hidden/sep__k1__mean.json | done | l3 Δ_sep=0.022, l4 Δ_sep=0.019, both p=0, pass=True |
| 32 | M7_C1_hidden | k=4,pool=mean | scripts/m7_c1_hidden.py --k 4 --pool mean | runs/M7_C1_hidden/sep__k4__mean.json | done | l3 Δ_sep=0.007, l4 Δ_sep=0.020, both p=0, pass=True |
| 33 | M7_C1_hidden | k=16,pool=mean | scripts/m7_c1_hidden.py --k 16 --pool mean | runs/M7_C1_hidden/sep__k16__mean.json | done | **headline: l4 Δ_sep=0.020 [0.018, 0.022], p=0; l3 Δ_sep=0.005** |
| 34 | M7_C1_hidden | k=64,pool=mean | scripts/m7_c1_hidden.py --k 64 --pool mean | runs/M7_C1_hidden/sep__k64__mean.json | done | l3 Δ_sep=0.004, l4 Δ_sep=0.014 |
| 35 | M7_C1_hidden | k=256,pool=mean | scripts/m7_c1_hidden.py --k 256 --pool mean | runs/M7_C1_hidden/sep__k256__mean.json | done | l3 Δ_sep=0.004, l4 Δ_sep=0.006 |
| 36 | M8_C2_queryability | k=16,pool=mean | scripts/m8_c2_queryability.py --k 16 --pool mean | runs/M8_C2_queryability/mrr__k16__mean.json | done | **headline: MRR=0.898, R@10=0.974, perm95=0.010, sig=True** |
| 37 | M8_C2_queryability | k=16,pool=act_weighted_mean | scripts/m8_c2_queryability.py --k 16 --pool act_weighted_mean | runs/M8_C2_queryability/mrr__k16__act_weighted_mean.json | done | MRR=0.898, R@10=0.974, sig=True |
| 38 | M8_C2_queryability | k=16,pool=max | scripts/m8_c2_queryability.py --k 16 --pool max | runs/M8_C2_queryability/mrr__k16__max.json | done | MRR=0.699, R@10=0.892, sig=True |
| 39 | M8_C2_queryability | k=16,pool=medoid | scripts/m8_c2_queryability.py --k 16 --pool medoid | runs/M8_C2_queryability/mrr__k16__medoid.json | done | MRR=0.783, R@10=0.951, sig=True |
| 40 | M9_C2_stability | half_k=16,pool=mean | scripts/m9_c2_stability.py --half_k 16 --pool mean | runs/M9_C2_stability/stability__hk16__mean.json | done | **headline: median cos=0.970 [0.969, 0.971], pass=True** |
| 41 | M9_C2_stability | half_k=16,pool=act_weighted_mean | scripts/m9_c2_stability.py --half_k 16 --pool act_weighted_mean | runs/M9_C2_stability/stability__hk16__act_weighted_mean.json | done | median cos=0.970, pass=True |
| 42 | M9_C2_stability | half_k=16,pool=max | scripts/m9_c2_stability.py --half_k 16 --pool max | runs/M9_C2_stability/stability__hk16__max.json | done | median cos=0.937, pass=True |
| 43 | M9_C2_stability | half_k=16,pool=medoid | scripts/m9_c2_stability.py --half_k 16 --pool medoid | runs/M9_C2_stability/stability__hk16__medoid.json | done | median cos=0.828, pass=True |
| 44 | M10_C2_separation | k=16,pool=mean | scripts/m10_c2_separation.py --k 16 --pool mean | runs/M10_C2_separation/sep__k16__mean.json | done | **headline: layer4 gap=0.170, d=1.96, p=0, pass=True. fc gap=-0.001 (grouping-heuristic artifact — see EXPERIMENT_RESULTS.md)** |
| 45 | M10_C2_separation | k=16,pool=act_weighted_mean | scripts/m10_c2_separation.py --k 16 --pool act_weighted_mean | runs/M10_C2_separation/sep__k16__act_weighted_mean.json | done | layer4 gap=0.169, d=1.94, pass=True |
| 46 | M10_C2_separation | k=16,pool=max | scripts/m10_c2_separation.py --k 16 --pool max | runs/M10_C2_separation/sep__k16__max.json | done | layer4 gap=0.048, d=0.99, pass=True |
| 47 | M10_C2_separation | k=16,pool=medoid | scripts/m10_c2_separation.py --k 16 --pool medoid | runs/M10_C2_separation/sep__k16__medoid.json | done | layer4 gap=0.224, d=2.65, pass=True |
| 48 | M11_layer_granularity | — | scripts/m11_layer_summary.py | runs/M11_layer_granularity/layer_summary.json | done | Cross-layer table synthesized (fc, layer4, layer3); k-plateau analysis for P1c logged (plateau_k=1 for both, monotone-nondecreasing=False — see EXPERIMENT_RESULTS.md interpretation). |
| 49 | ~~M12_cross_model_verify~~ | vit_b_16,k=16,pool=mean | ~~scripts/m12_cross_model.py --inspected_model vit_b_16~~ | ~~runs/M12_cross_model_verify/vit_b_16__k16__mean.json~~ | **SKIPPED (user directive 2026-07-14 — retracted)** | Result retracted; MUST NOT be consumed downstream. |
| 50 | ~~M12_cross_model_verify~~ | vgg_16,k=16,pool=mean | ~~scripts/m12_cross_model.py --inspected_model vgg16~~ | ~~runs/M12_cross_model_verify/vgg_16__k16__mean.json~~ | **SKIPPED (user directive 2026-07-14 — retracted)** | Result retracted; MUST NOT be consumed downstream. |
| 51 | ~~M12_cross_model_verify~~ | efficientnet_b0,k=16,pool=mean | ~~scripts/m12_cross_model.py --inspected_model efficientnet_b0~~ | ~~runs/M12_cross_model_verify/efficientnet_b0__k16__mean.json~~ | **SKIPPED (user directive 2026-07-14 — retracted)** | Result retracted; MUST NOT be consumed downstream. |
| 52 | M13_final_report | — | scripts/m13_final_report.py | runs/M13_final_report/results_table.json, FINAL_RESULTS.md | done (retained; does NOT strictly depend on M12) | Reads M0-M11 aggregates. Aggregator's P1c/P2c verdict tags refer to internal thresholds; see EXPERIMENT_RESULTS.md for the semantically-informed verdict on M0-M11 (both C1 and C2 supported on ResNet-50). |

**Total planned runs: 52** — **49 done (M0-M11 + M13)**, **3 SKIPPED (M12 retracted per user directive 2026-07-14)**, **0 pending, 0 failed**. Actual aggregate GPU-h retained: **~0.4 h** (M12's ~0.3 h is sunk cost, not counted toward pipeline evidence). Well below the 3.2 h plan estimate and the 10 h task budget.

## Run directory

- `runs/A1_full_pipeline/cost.json` — canonical cost manifest for this full-pipeline invocation.
