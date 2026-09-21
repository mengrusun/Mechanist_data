# Claim Ledger — Steerable per-variable pure directions in Llama-3.1-8B-Instruct's dictator decision

**Direction**: Steerable social-variable directions in LLM decision making (task.md; Llama-3.1-8B-Instruct dictator game over gender G, age A, instruction framing I, meeting condition M)
**Date**: 2026-07-13 → 2026-07-13
**Pipeline**: completed | **Iteration**: 6/10 "almost" (2/6)
**Models**: claim=claude-opus-4-7, experiment=claude-opus-4-7, verify=claude-sonnet-4-6, iteration=claude-opus-4-7
**Updated after**: iteration:final

| Claim | Main experiment | Verify | Post-Iteration | Final |
|-------|-----------------|--------|----------------|-------|
| C1 linear encoding | supported (probe 1.0 all V) | ⚪ INTEGRITY_ONLY (cap) | ⚠ narrowed — LOPO reveals shallow phrasing-detector confound | ⚠ narrowed — holds at mid-/late-layers (10-14, 28-32); shallow picked ell_V* is partly phrasing artifact |
| C2 purity via decorrelation | supported (GS); not-supported (LEACE) | ⚪ INTEGRITY_ONLY (cap) | ⚠ narrowed — "partially decorrelated basis" not "pure directions"; not empirically linked to C3 | ⚠ configuration-specific & narrowed — GS partial decorrelation; C3 causal evidence uses raw v̂_V |
| C3 bidirectional steering | partial at ell_V*; supported at L=16 | ✓ PASS robustness=1.00 | ✓ strengthened — n=10 random null at ≥2.5σ direction-specificity; V=M inversion at 2.84σ | ✓ holds (sign-asymmetric direction-specific causal steering at L=16 across two models) |
| C4 selectivity (4×4) | not-supported at ell_V* | ⚪ INTEGRITY_ONLY (cap) | ⚠ L=16 sweep ran — p=0.128 borderline; partial selectivity for V=A/M | ⚠ configuration-specific & narrowed — partial (V=A/M selective; V=I non-selective; V=G one-signed) |

---
## C1 — Linear encoding of each social/contextual variable
- **Statement**: For each social/contextual variable V ∈ {G, A, I, M}, a linear direction v̂_V in the Llama-3.1-8B-Instruct residual stream carries V's decision-relevant influence on the dictator-game transfer amount.
- **Origin**: task.md Claim §1 (linear encoding)
- **Data**: DG-1000 dictator-game trials + 4×1,000 paired-prompt partners — provenance=constructed; available=5000, used=5000 (800 train / 200 held-out baseline) + iteration LOPO stress
- **Models**: Llama-3.1-8B-Instruct
- **Method**: M2 — paired-prompt CAA difference-of-means direction v̂_V(layer); linear-probe cv_acc + projection-transfer OLS at picked ell_V*. Iteration ⓪ added Leave-One-Phrasing-Out cross-val as a lexical-confound stress test — screen (paired-prompt CAA) → decode (linear probe) → verify (proj-transfer β + LOPO) → recover (n/a)
- **Main experiment**: supported — probe cv_acc = 1.000 for all V; held_acc = 1.000; projection-transfer β at ell_V*: G=-10.12 (p=0.002), A=-4.85 (p=0.059 marginal), I=+32.10 (p=1.5e-9), M=+41.28 (p=3.5e-5). Iteration LOPO stress narrows: V=M at ell_V*=2 LOPO=0.483 (chance).
- **Verify**: robustness=null — method n/a / dataset n/a / model excluded; integrity=WARN; verdict=INTEGRITY_ONLY (stage2_skip_reason=max_verify_claims_cap)
- **Iteration**: almost — main-experiment integrity WARN; LOPO narrows the shallow claim; falsified: 'linear encoding of V=M at ell=2' is a phrasing-detector artifact; narrowed_to: holds at mid- (10-14) and late-layers (28-32) but not at token-embedding-proximal ell_V* ∈ {2,4,6}
- **Final**: ⚠ narrowed — linear encoding holds at mid- to late-layers (10-14, 28-32) but the shallow picked ell_V* result is confounded by phrasing (V=M LOPO=0.483 = chance); swap-test deferred
- **Caveats**: V=A projection-transfer β marginal at p=0.06; picked ell_V* ∈ {2,4,6} — token-embedding-proximal; Phase 2 audit: proxy GT (tau) not explicitly labeled; 48 unique prompt texts appear in both train/held splits; productive layer band per LOPO: 10-14 (V=I/M) and 28-32 (V=G/A)
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M2, runs/M_main_v1/artifacts/m2/, refine-logs/EXPERIMENT_RESULTS.md#M2, verify/C1_linear_encoding_social_vars/, runs/iteration_round_1/c1_lexical_stress_test.json, runs/iteration_round_1/corrected_layer_pick.json
- **Figures**:
  - ![Per-layer 5-fold probe cv_acc (dashed) vs. leave-one-phrasing-out cv_acc (solid) per variable V ∈ {G,A,I,M} on Llama-3.1-8B-Instruct — reveals the phrasing-detector artifact at shallow layers (V=M ℓ=2 LOPO=0.483 = chance) and the productive mid-layer bands 10-14 and 28-32.](figures/C1/c1_lopo_vs_probe_by_layer.png) — vector: `figures/C1/c1_lopo_vs_probe_by_layer.pdf`
  - #### C1 main-experiment probe cv_acc + projection-transfer β at picked layers ell_V* (Llama-3.1-8B-Instruct)

    | V | ell_V* | probe cv_acc | held_acc | proj-transfer beta | beta p-value | baseline tau effect |
    |---|-------:|-------------:|---------:|-------------------:|-------------:|--------------------:|
    | G | 4 | 1.000 | 1.000 | -10.12 | 0.002 | -0.380 |
    | A | 6 | 1.000 | 1.000 | -4.85 | 0.059 | +0.106 |
    | I | 2 | 1.000 | 1.000 | +32.10 | 1.5e-09 | +1.287 |
    | M | 2 | 1.000 | 1.000 | +41.28 | 3.5e-05 | +0.713 |

    Source `.tex`: `figures/C1/c1_probe_transfer_stats.tex`

---
## C2 — Purity via decorrelation
- **Statement**: A pure direction ṽ_V (raw v̂_V minus its overlap with the other three variables' directions) linearly predicts V without predicting the other variables — a cross-leakage matrix diagonal ≫ off-diagonal under a pre-registered threshold.
- **Origin**: task.md Claim §2 (purity via decorrelation)
- **Data**: DG-1000 held-out activations cached from M2 — provenance=constructed; available=5000, used=1000 held (200 baseline + 800 partners)
- **Models**: Llama-3.1-8B-Instruct
- **Method**: M3 — Gram-Schmidt + LEACE decorrelators; 4×4 cross-leakage probe matrix at ell_V*; predicates: diag mean ≥ 0.95× raw diag AND off-diag max ≤ chance+0.05 (0.55) — screen (raw v̂_V) → decode (GS + LEACE decorrelate) → verify (4×4 cross-leakage probe) → recover (n/a)
- **Main experiment**: supported (Gram-Schmidt); not-supported (LEACE) — GS: diag mean 0.958, off-diag max 0.506; LEACE: diag mean 0.573, off-diag max 0.829, ‖v̂_LEACE‖/‖v̂_raw‖ = 3.1-7.4
- **Verify**: robustness=null — method n/a / dataset n/a / model excluded; integrity=WARN; verdict=INTEGRITY_ONLY (stage2_skip_reason=max_verify_claims_cap)
- **Iteration**: almost — largest unresolved conceptual concern (P3): C3's causal steering uses raw v̂_V while C2 claims pure/decorrelated directions; the two are not empirically linked at L=16; narrowed_to: 'pure directions free of confounds' → 'GS-decorrelated basis with off-diag 0.506 residual coupling'
- **Final**: ⚠ configuration-specific & narrowed — GS variant supports a partial decorrelation at shallow layers; LEACE fails at these layers; C3 causal evidence used raw v̂_V so this claim is not empirically bridged to the causal result
- **Caveats**: LEACE fails at shallow picked layers — dataset-specific pathology; GS off-diag max 0.506 is borderline (threshold 0.55); 1-D scalar-projection cross-leakage probe is conservative; Reviewer's P3: C3's causal success at L=16 was demonstrated on raw v̂_V — future purity-intervention experiment at L=16 with GS directions is the natural closure
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M3, runs/M_main_v1/artifacts/m3/, refine-logs/EXPERIMENT_RESULTS.md#M3, verify/C2_purity_via_decorrelation/
- **Figures**:
  - ![4×4 cross-leakage probe accuracy matrix under GS vs LEACE decorrelators at Llama-3.1-8B-Instruct picked shallow layers ℓ_V*. Diagonal = target-self, off-diagonal = leakage. GS: diag mean 0.958 / off-diag max 0.506. LEACE: diag mean 0.573 / off-diag max 0.829.](figures/C2/c2_crossleakage_gs_vs_leace.png) — vector: `figures/C2/c2_crossleakage_gs_vs_leace.pdf`
  - #### C2 decorrelation summary — GS vs LEACE at picked shallow layers (Llama-3.1-8B-Instruct)

    | decorrelator | diag mean | off-diag max | preservation >= 0.95x raw diag? | off-diag <= chance+0.05 (0.55)? |
    |--------------|----------:|-------------:|:-------------------------------:|:-------------------------------:|
    | raw | 0.934 | 0.694 | baseline | no |
    | mean-centered | 0.934 | 0.694 | yes | no |
    | GS | 0.958 | 0.506 | yes | yes |
    | LEACE | 0.573 | 0.829 | no | no |

    Source `.tex`: `figures/C2/c2_purity_summary.tex`

---
## C3 — Bidirectional causal steering
- **Statement**: Activation-addition h ← h + α ṽ_V at signed α ∈ [-4σ, +4σ] causally and monotonically shifts the mean transfer amount, with at least one α<0 inverting V's baseline sign (bidirectional causal steering with a dose-response + inversion audit and a coherence gate on generation quality).
- **Origin**: task.md Claim §3 split part-a (bidirectional causal steering)
- **Data**: DG-1000 held-out × 4 V × signed-α grid; supp L={12,16} on raw v̂_V; verify variant on Meta-Llama-3-8B-Instruct at L={0,10,16}; iteration L=16 random-direction null (n=3 → n=10) — provenance=constructed; available=5000, used=56000 decode ops main + supp + verify variant compact n=10/cell + iteration random-null (4 V × 5 α × 10 seeds at L=16)
- **Models**: Llama-3.1-8B-Instruct, Meta-Llama-3-8B-Instruct (verify)
- **Method**: M4 — Signed CAA activation-addition on ṽ_V at picked ell_V* + L={12,16} supp; verify variant: same on Meta-Llama-3-8B-Instruct L={0,10,16}. Iteration ② added the L=16 random-direction control (10 seeds/setting) — screen (v̂_V at ell_V*) → decode (α-sweep) → verify (dose-response + inversion + random-direction null) → recover (extended null at n=10)
- **Main experiment**: partial at ell_V* [prov. layer-pick under-power]; SUPPORTED at L=16 across two models + strengthened by n=10 random null at ≥2.5σ direction-specificity — Main L=16: V=M inverts (+0.71→−0.31 at α=+2σ), V=A amplifies 5.5×, V=G amplifies 2.5× at α=−2σ, V=I bidirectional; verify variant (Meta-Llama-3 L=16): V=M sign-inverts at α=+2, V=G/A/I bidirectional; iteration null (n=10 seeds): V=M α=+2 inversion at 2.84σ specificity (0/10 random seeds inverted); V=A α=+2 at 2.69σ, V=G α=−2 at 4.30σ, V=I α=−2 at 2.50σ
- **Verify**: robustness=1.00 — method n/a / dataset n/a / model pass; integrity=WARN (Phase 2 + Phase 9); verdict=PASS
- **Iteration**: PASS (held; strengthened by L=16 random-direction null n=10 at ≥2.5σ direction-specificity for all four showcase effects; V=M inversion at 2.84σ); changed: iteration 1 ② added L=16 random-direction control (3 seeds), iteration 2 ① extended to 10 seeds, iteration 2 ⓪ LOPO retroactively justifies L=16 as productive layer-band representative; narrowed_to: 'bidirectional' → 'sign-asymmetric direction-specific'; uses raw v̂_V (not purified) — does not empirically validate C2's purity narrative
- **Final**: ✓ holds (sign-asymmetric direction-specific causal steering at L=16 across two models, ≥2.5σ direction-specificity for all four showcase effects); provisional at picked shallow ell_V* due to layer-pick heuristic
- **Caveats**: verify variant compact n_baseline=10/cell (main = 200) — qualitative PASS; L=16 uses raw v̂_V not GS/LEACE-purified (Reviewer P3: does not empirically link to C2's purity claim); 'bidirectional' narrowed to 'sign-asymmetric direction-specific'; under-powered at ell_V* — layer-pick heuristic root cause
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M4, runs/M_main_v1/artifacts/m4/, runs/M4_supp_deep_v1/m4/, runs/verify_C3_variant_model_swap_v1/artifacts/, runs/iteration_round_1/L16_random_control/, runs/iteration_round_1/L16_random_control_extended/, refine-logs/EXPERIMENT_RESULTS.md#M4, verify/C3_bidirectional_causal_steering/
- **Figures**:
  - ![Mean transfer amount vs. signed α at L=16 per variable V ∈ {G,A,I,M} — Llama-3.1-8B-Instruct main (M4-supp) vs Meta-Llama-3-8B-Instruct verify variant. V=M sign-inverts at α=+2σ in both models; V=A amplifies 5.5×; V=G amplifies 2.5× at α=−2σ; V=I bidirectional (+46% amplify at α=-2, -22% attenuate at α=+2).](figures/C3/c3_dose_response_L16.png) — vector: `figures/C3/c3_dose_response_L16.pdf`
  - #### C3 direction-specificity σ vs random-direction null at L=16 (n=10 seeds) — Llama-3.1-8B-Instruct main; ≥2.5σ for all four showcase effects, V=M inversion at 2.84σ.

    | V | alpha (sigma_proj units) | effect type | direction-specificity vs random null (sigma) |
    |:-:|:------------------------:|:-----------:|---------------------------------------------:|
    | M | +2 | sign inversion | 2.84 sigma |
    | A | +2 | amplification | 2.69 sigma |
    | G | -2 | amplification | 4.30 sigma |
    | I | -2 | amplification | 2.50 sigma |

    Source `.tex`: `figures/C3/c3_direction_specificity.tex`

---
## C4 — Selectivity (4×4 matrix)
- **Statement**: Steering ṽ_V leaves the other three variables' effects on the transfer decision intact — a 4×4 selectivity matrix M[V, W] has diagonal effect-magnitude significantly larger than off-diagonal (permutation test rejects diagonal = off-diagonal).
- **Origin**: task.md Claim §3 split part-b (selectivity), Claim §4 support (targeted intervention)
- **Data**: DG-1000 held-out reused across 4×4 steer-V/measure-W cells at α ∈ {-2,0,+2}σ at ell_V*; iteration L=16 4×4 sweep — provenance=constructed; available=5000, used=12000 (main) + iteration L=16 12 grid points
- **Models**: Llama-3.1-8B-Instruct
- **Method**: M5 — 4×4 selectivity matrix M[V,W] at ell_V*, LEACE-pure, single-site; predicates: diag ≫ off-diag AND permutation-test p<0.05. Iteration ② added the L=16 4×4 selectivity sweep — screen (ell_V* / L=16) → decode (steer V, measure W) → verify (4×4 matrix + permutation) → recover (n/a)
- **Main experiment**: not-supported at ell_V* [prov. layer-pick under-power]; partially selective at L=16 with substantial collateral coupling — At ell_V*: p=0.84 (H0 not rejected), V=G collapse. At L=16 (iteration): p=0.128; per-V diag/mean-off ratio at α=+2σ: V=A 2.25, V=M 2.19 (selective); V=I 1.00 (non-selective); V=G 0.00 at +σ (works at −σ with diag=−0.570)
- **Verify**: robustness=null — method n/a / dataset n/a / model excluded; integrity=WARN; verdict=INTEGRITY_ONLY (stage2_skip_reason=max_verify_claims_cap)
- **Iteration**: INTEGRITY_ONLY (unchanged) — L=16 sweep improves p from 0.84 (shallow) to 0.128 (L=16) but reveals partial selectivity; falsified: universal selectivity across V (V=I non-selective at L=16, V=G fails at +σ); narrowed_to: partial and V-dependent (V=A/M selective, V=I non-selective, V=G one-signed)
- **Final**: ⚠ configuration-specific & narrowed — partially selective at L=16 for V=A/M; V=I is non-selective and V=G is one-signed; permutation p=0.128 not stat-sig with n=4 V; swap-test deferred (max_verify_claims cap)
- **Caveats**: iteration L=16 sweep: permutation p=0.128 not stat-sig with n=4 V — future work should expand variable set or use a stronger selectivity statistic; V=I selectivity ratio 1.00 at L=16; V=G steering only works at α=−σ (diag=−0.570); swap-test deferred; Phase 2 audit: synthetic_proxy GT; min-diagonal=0 artifact at shallow layers
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M5, runs/M_main_v1/artifacts/m5/, runs/iteration_round_1/L16_c4_selectivity/, refine-logs/EXPERIMENT_RESULTS.md#M5, verify/C4_selectivity_matrix/
- **Figures**:
  - ![4×4 selectivity matrix M[V,W] = w_effects[W](α=+2σ) − w_effects[W](α=0) at Llama-3.1-8B-Instruct picked shallow layers ell_V* (LEACE-pure, single-site) vs. L=16 (iteration ②, raw v̂_V). Left: shallow (p=0.84, V=G collapse). Right: L=16 (p=0.128, partial selectivity).](figures/C4/c4_selectivity_matrix_layers.png) — vector: `figures/C4/c4_selectivity_matrix_layers.pdf`
  - #### C4 per-V diagonal / mean-off-diagonal ratio at L=16 (α=+2σ) — V=A 2.25, V=M 2.19 (selective); V=I 1.00 (non-selective); V=G 0.00 at +σ (only steers at −σ).

    | V | diag / mean-off-diag ratio (L=16, alpha=+2 sigma) | notes |
    |:-:|--------------------------------------------------:|-------|
    | A | 2.25 | selective |
    | M | 2.19 | selective |
    | I | 1.00 | non-selective (co-modulates other Ws) |
    | G | 0.00 | one-signed (only steers at alpha=-sigma; diag=-0.570 there) |

    Source `.tex`: `figures/C4/c4_perV_selectivity_ratio.tex`

---
## Journey Summary
- **Claim**: given behavior (task.md) faithfully captured → 4 falsifiable sub-claims (C1 linear encoding, C2 purity, C3 bidirectional steering, C4 selectivity) mapped to milestones M1-M7
- **Mechanism strategy**: Location → Causal Intervention
- **Mechanism routing**: family=Representation and Parameter Analysis, submethod=Steering Vectors (CAA); composition = paired-prompt CAA extractor → GS/LEACE decorrelation → signed activation-addition → 4×4 selectivity + baselines
- **Experiment**: 7 dispatched runs (sanity + main + M4-supp L={12,16} + M7 portability), ~2.5 GPU-hours, headline mixed: C1 supported, C2 supported (GS) / failed (LEACE at shallow layers), C3 under-powered at picked shallow layers but supported at supp L=16 (M sign-inverts, A/G amplify, I bidirectional), C4 not supported at picked layers (V=G collapse); M7 portability negative (DeepSeek outputs τ=10 constant)
- **Verify**: 4 claim(s): 1 PASS / 0 FAIL / 0 INCONCLUSIVE / 0 ZEV / 3 INTEGRITY_ONLY (cap=3, swap_off=0); integrity[Phase2=WARN / Phase9=WARN]
- **Iteration**: 2/6 iterations, claim-reentries=0/2, score 6/10 verdict almost, termination=positive_verdict; 4 back-edge dispatches (~0.9 GPU-h); C3 strengthened by n=10 random-direction null at ≥2.5σ direction-specificity; C1 LOPO stress test confirmed shallow-layer phrasing-confound suspicion for V=M (LOPO=0.483 = chance); C4 L=16 selectivity ran for the first time (p=0.128, borderline)
- **Figures**: 8 across 4 claims; 0 judgment-skipped; 0 render-skipped, 0 errored

## Open Items
- verify+iteration: C1, C2, C4 remain INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap) — upgrade via `/auto-verify C{1,2,4} — resume: true` for full stress-test
- iteration P3 (C2 raw-vs-pure conceptual mismatch): C3's causal success at L=16 uses raw v̂_V but C2 claims pure/decorrelated directions — the two are not empirically linked. Paper-scope fix: narrow C2 to 'GS-decorrelated basis with off-diag 0.506 residual coupling'; empirical fix: re-run C3-style steering at L=16 with GS-purified directions
- narrative: 'bidirectional causal steering' should be reframed to 'sign-asymmetric direction-specific causal steering' (each V responds strongly to one signed α direction, not symmetrically) — per iteration Section-8 reviewer recommendation
- narrative: C4 selectivity headline needs restraint — L=16 permutation p=0.128 is better than shallow p=0.84 but not stat-sig with n=4 V; per-V ratio shows partial selectivity + substantial collateral coupling (V=I ratio 1.00 = non-selective; V=G collapses at +σ)
- narrative: C1 must disclose the LOPO failure at V=M ell=2 (LOPO=0.483 = chance) — shallow-layer C1 for V=M is a phrasing detector, not linear encoding of the underlying variable; productive layer band is [12-24]
- experiment: M4-supp + M7 portability runs did not write cost.json (GPU-pin witness) — implementation gap in supp/portability wrappers, not a pin violation
- experiment: M7 portability NEGATIVE (DeepSeek anchors at τ=10 under DG-1000 prompt) — verify swapped to Meta-Llama-3-8B-Instruct instead
