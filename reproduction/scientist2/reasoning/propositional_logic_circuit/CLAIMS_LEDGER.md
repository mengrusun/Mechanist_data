# Claim Ledger — Sparse Modular Circuit for Propositional-Logic Reasoning

**Direction**: A sparse, modular circuit implements propositional-logic reasoning in LLMs (Mistral-7B lead); identify components, prove modularity, verify necessity + sufficiency via activation patching.
**Date**: 2026-07-14 → 2026-07-15
**Pipeline**: **completed** | **Iteration**: 7.0/10 "ready" (0/6)
**Models**: claim=claude-opus-4-7, experiment=claude-opus-4-7, verify=claude-sonnet-4-6, iteration=claude-opus-4-7
**Updated after**: iteration:final

| Claim | Main experiment | Verify | Post-Iteration | Final |
|-------|-----------------|--------|----------------|-------|
| C1 sparse component set | partial — completeness ✓ (0.955) / minimality ✗ (0.0014) | INTEGRITY_ONLY (cap) | ⓪ narrative caveats added | ⚪ integrity_only (audit passed with WARN; swap-test deferred) |
| C2 modular decomposition | fail — median_dominance 1.20 (target ≥2); only 'fact' role borderline | INTEGRITY_ONLY (cap) | ⓪ narrative caveats added; refuted stated | ⚪ integrity_only (audit passed with WARN; swap-test deferred; refuted) |
| C3 necessity + sufficiency | partial — necessity ✓ (LD 0.955) / sufficiency ✗ (LD 0.113) | PASS (robustness 1.0, model swap Gemma-2-9B) | ⓪ narrative caveats + narrowed-to added | ✓ verify PASS — necessary-but-not-sufficient asymmetry is cross-family robust; sufficiency half refuted, necessity half stands |

---
## C1 — sparse component set (Mistral-7B propositional logic)
- **Statement**: A sparse subset of specific attention heads and MLP components jointly implements the minimal propositional-logic reasoning task — the circuit is small relative to the full model (target |shortlist|/|total| ≤ ~15% with completeness ≥ 0.9 and single-removal minimality drop ≥ 0.05 on Mistral-7B).
- **Origin**: given (task.md)
- **Data**: synthetic propositional-logic template — anchor cell (k=3, chain=2, natural), corrupt_fact — provenance=constructed; available=~4000 clean + ~16000 matched-corrupt prompts, used=500 anchor pairs (M1)
- **Models**: Mistral-7B-v0.1
- **Method**: Attribution-patching screen (Attribution Patching) — rank attention heads + MLPs by causal contribution to logit-diff on clean vs corrupt_fact, shortlist by cumulative-effect + 15% size cap; completeness = restore-with-shortlist LD-recovery; minimality = avg single-component-removal LD-drop (M1)
- **Main experiment**: **partial** — shortlist_size=158, sparsity_fraction=0.150, cumulative_effect=0.758, completeness_LD=0.955 ✓, completeness_PD=0.905, minimality_avg=0.0014 ✗ (target ≥0.05), minimality_median=0.0027
- **Verify**: robustness=—; axes: method excluded / dataset excluded / model excluded; integrity=WARN; verdict=**INTEGRITY_ONLY** (stage2_skip_reason: max_verify_claims_cap)
- **Iteration**: ready (score 7/10, round 1 STOP); narrative caveats: disclosed 20/158 minimality sampling; hard-cap sparsity acknowledged
- **Final**: ⚪ integrity_only (audit passed with WARN — undisclosed minimality sampling 20/158; swap-test deferred — max_verify_claims cap; narrative caveats added by iteration ⓪)
- **Caveats**: High-completeness but very-low-minimality suggests the "sparse" shortlist is over-inclusive — a tighter top-K may satisfy minimality without losing completeness; Phase 2 audit flagged undisclosed minimality sampling (20/158 components); soft methodology disclosure, not a reversal
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M1, results/M1_attribution.json, refine-logs/EXPERIMENT_RESULTS.md#C1, verify/C1_sparse_component_set/main_experiment_audit/EXPERIMENT_AUDIT.md, verify/C1_sparse_component_set/ROBUSTNESS.md
- **Figures**:
  - ![C1 — attribution-patched 158-component shortlist (15.0% of components) restores 0.955 clean-vs-corrupt logit-diff (completeness ✓, target ≥ 0.9) but single-component removal drops recovery by only 0.0014 on average (minimality ✗, target ≥ 0.05).](figures/C1/c1_completeness_minimality.png) — vector: `figures/C1/c1_completeness_minimality.pdf`
  - ![C1 — per-layer per-head attention attribution magnitudes (32 layers × 32 heads on Mistral-7B). |Direct-effect scores| on the anchor cell (k=3, chain=2, natural), corrupt_fact corruption.](figures/C1/c1_attribution_heatmap.png) — vector: `figures/C1/c1_attribution_heatmap.pdf`

---
## C2 — modular decomposition (fact-id / rule-app / answer-proj)
- **Statement**: The circuit decomposes into a small number of modular sub-circuits with distinct functional roles (fact-identification, rule-application, answer-projection) rather than presenting as an entangled mixture — target role-assignment matrix S with per-component dominance ratio ≥ 2×, dissociation ≥ 0.1, Jaccard cross-cell stability ≥ 0.6, null-shuffle p ≤ 0.01.
- **Origin**: given (task.md)
- **Data**: synthetic propositional-logic template — role-specific corruptions at anchor cell + stability cells (k=5,chain=2) and (k=3,chain=3) — provenance=constructed; available=~4000 clean + ~16000 matched-corrupt prompts, used=300 pairs × 3 roles (M4) + 200 pairs × 3 roles × 2 cells (M4.stab); top-40 shortlist (17 attn + 23 mlp)
- **Models**: Mistral-7B-v0.1
- **Method**: Role-dissociation via type-specific counterfactual corruption + null-shuffle p-value test; per-component dominance ratio + role-partition dissociation; cross-cell Jaccard stability across (k=5,chain=2) and (k=3,chain=3) (M4 + M4.stab)
- **Main experiment**: **fail** — median_dominance=1.20 ✗ (target ≥2.0); dissociation {fact:0.089, rule:~0, answer:~0}; null-shuffle p {fact:≈0.05, rule:1.0, answer:1.0}; Jaccard cross-cell stability {fact:0.75 ✓, answer:0.77 ✓, rule:0.36 ✗}. **Headline**: components do NOT partition into three distinct role specialists — only "fact" role has borderline signal; "rule" and "answer" partitions are statistically indistinguishable from random.
- **Verify**: robustness=—; axes: method excluded / dataset excluded / model excluded; integrity=WARN; verdict=**INTEGRITY_ONLY** (stage2_skip_reason: max_verify_claims_cap)
- **Iteration**: ready (score 7/10, round 1 STOP); narrative caveats: top-40/158 shortlist scope explicitly disclosed as a scope condition; falsified: modular decomposition on anchor cell (main experiment refuted; only 'fact' role borderline)
- **Final**: ⚪ integrity_only (audit passed with WARN — shortlist scope reduction top-40/158 disclosed; swap-test deferred — max_verify_claims cap; underlying main-experiment: modular decomposition refuted; narrative caveats added by iteration ⓪)
- **Caveats**: Shortlist reduced from 158 to top-40 (17 attn + 23 mlp) for tractability; M2 dose-response justifies (top-31 recovers 0.946); Phase 2 audit flagged the shortlist scope reduction — disclosed in Notes, no reversal
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M4, refine-logs/EXPERIMENT_PLAN.md#M4stab, results/M4_roles.json, results/M4stab.json, refine-logs/EXPERIMENT_RESULTS.md#C2, verify/C2_modular_decomposition/main_experiment_audit/EXPERIMENT_AUDIT.md, verify/C2_modular_decomposition/ROBUSTNESS.md
- **Figures**:
  #### C2 — role-dissociation statistics on the anchor cell and cross-cell stability

  | role | dissociation (target ≥ 0.1) | null-shuffle p (target ≤ 0.01) | Jaccard k5c2↔k3c3 (target ≥ 0.6) | signal |
  |---|---|---|---|---|
  | fact | 0.089 | 0.05 | 0.75 | borderline ✓ |
  | rule | 0.027 | 1.00 | 0.36 | random-like ✗ |
  | answer | 0.026 | 1.00 | 0.77 | random-like ✗ |

  Median per-component dominance ratio on the anchor cell: **1.20** (target ≥ 2.0) — refuted.

  Source `.tex`: `figures/C2/c2_role_dissociation_table.tex`

---
## C3 — necessity + sufficiency (path patching / reinsertion)
- **Statement**: Activation-patching / causal-mediation experiments show that the identified components are both necessary (clean→corrupt patching restores correct behaviour) and sufficient (their outputs alone drive the answer) — target recovery ≥ 0.8 on {logit_diff, prob_diff, KL} with matched-control specificity gap ≥ 0.6.
- **Origin**: given (task.md)
- **Data**: synthetic propositional-logic template — anchor cell (k=3, chain=2, natural), 500 clean + 500 matched-corrupt pairs per corruption type — provenance=constructed; used=500 anchor pairs (M2 + M3); 5 seeds for sufficiency
- **Models**: Mistral-7B-v0.1, Gemma-2-9B
- **Method**: Path-patching necessity (clean→corrupt patch shortlist; measure LD/PD/KL recovery + matched-control specificity, M2) + Sufficiency reinsertion (corrupt+identified-component-clean-patch; resample-ablate everything else; 5 seeds, M3) + Cross-family verify on Gemma-2-9B (M5)
- **Main experiment**: **partial (necessity ✓ / sufficiency ✗)** — necessity: LD_recovery=0.955 ✓, PD_recovery=0.905, specificity_gap=0.836 ✓; sufficiency: LD_recovery=0.113 ✗ (target ≥0.8), PD_recovery=0.116, KL_recovery=0.076, matched-control ≈0, specificity_gap=0.113, per-seed std=0.036; cross-family Gemma-2-9B: necessity LD=1.018 ✓, sufficiency LD=0.019 ✗. **Headline**: circuit is *necessary but not sufficient*; the necessity-yes/sufficiency-no pattern recurs on Gemma-2-9B — a cross-family robust negative on the sufficiency half.
- **Verify**: robustness=1.0; axes: method n/a / dataset n/a / model **pass** (Gemma-2-9B swap, reused M5); integrity=WARN; verdict=**PASS**
- **Iteration**: ready (score 7/10, round 1 STOP); narrative caveats: KL scaling artifact explicit; one-variant verify acknowledged; anchor-cell scope highlighted; kept interventional phrasing modest; narrowed_to: the identified circuit is necessary but not sufficient (a partial refutation of the strong claim); the necessary half is cross-family robust (Mistral-7B & Gemma-2-9B)
- **Final**: ✓ verify PASS (robustness=1.0) — the necessary-but-not-sufficient asymmetry is cross-family robust; the sufficiency half of the given claim is refuted, the necessity half stands (narrative caveats added by iteration ⓪)
- **Caveats**: KL recovery denominator collapses (tiny baseline KL(clean||corrupt)=0.043) — LD and PD are the definitive metrics; Phase 2 audit WARN: KL scaling artifact documented
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M2, refine-logs/EXPERIMENT_PLAN.md#M3, results/M2_necessity.json, results/M3_sufficiency.json, results/M5_gemma9b.json, refine-logs/EXPERIMENT_RESULTS.md#C3, verify/C3_necessity_sufficiency/main_experiment_audit/EXPERIMENT_AUDIT.md, verify/C3_necessity_sufficiency/variant_audit/EXPERIMENT_AUDIT.md, verify/C3_necessity_sufficiency/ROBUSTNESS.md
- **Figures**:
  - ![C3 — the necessary-but-not-sufficient asymmetry, cross-family robust. On Mistral-7B and Gemma-2-9B, path-patching the identified shortlist restores logit-diff (necessity ✓, 0.955 and 1.018); reinserting only the shortlist while resample-ablating the rest recovers almost nothing (sufficiency ✗, 0.113 and 0.019). The verify PASS (robustness = 1.0) is on this asymmetry.](figures/C3/c3_necessity_vs_sufficiency.png) — vector: `figures/C3/c3_necessity_vs_sufficiency.pdf`

  #### C3 — full recovery-metric comparison across necessity + sufficiency + models

  | model | LD nec (≥0.8) | PD nec | specificity nec (≥0.6) | LD suf (≥0.8) | PD suf | KL suf | specificity suf |
  |---|---|---|---|---|---|---|---|
  | Mistral-7B-v0.1 | 0.955 | 0.905 | 0.836 | 0.113 | 0.116 | 0.076 | 0.113 |
  | Gemma-2-9B | 1.018 | 1.012 | — | 0.019 | 0.005 | -0.050 | — |

  KL recovery on the anchor cell is de-emphasised because the baseline KL(clean‖corrupt) is 0.043, so the recovery ratio blows up; LD and PD are the definitive metrics and tell the same story.

  Source `.tex`: `figures/C3/c3_metric_comparison_table.tex`

---
## Journey Summary
- **Claim**: given behavior faithfully captured — three sub-claims (sparse component set, modular decomposition, necessity + sufficiency) on Mistral-7B propositional logic
- **Mechanism strategy**: Location → Causal Intervention
- **Mechanism routing**: family=Causal Attribution / Attribution Patching + Patching (Attribution Patching + Path Patching + Resample-Ablation Reinsertion)
- **Experiment**: 9 milestones (M0.dataset + M0.setup + M1–M5 + M4.stab + M6), ~2.03 GPU-h productive (+1.33 GPU-h discarded on scope-reduced M4 rerun), headline: mixed — C1 partial (redundant), C2 refuted, C3 necessity ✓ / sufficiency ✗ (recurs on Gemma-2-9B)
- **Verify**: 3 claims Stage-1 audited (all WARN → admitted); Stage 2 pick=C3; 1 model-swap variant reused (Gemma-2-9B via M5) — C3 robustness=1.0 → PASS; C1 & C2 INTEGRITY_ONLY (cap); 1 PASS / 0 FAIL / 0 INCONCLUSIVE / 0 ZEV / 2 INTEGRITY_ONLY (cap=2, swap_off=0); integrity[Phase2:WARN/Phase9:PASS]; 0.00 GPU-h
- **Iteration**: 0/6 iterations (positive verdict, STOP fired round 1), claim-reentries=0/2, score 7/10 verdict ready, termination=positive_verdict
- **Figures**: 5 figures across 3 claims (2 image + 1 table for C1+C2, 1 image + 1 table for C3); 0 judgment-skipped; 0 render-skipped, 0 errored

## Open Items
- C1 INTEGRITY_ONLY — swap-test deferred by max_verify_claims cap (K=1); to run: `/auto-verify C1 --resume true` (Phase 2 audit already cached)
- C2 INTEGRITY_ONLY — swap-test deferred by max_verify_claims cap (K=1); to run: `/auto-verify C2 --resume true` (Phase 2 audit already cached)
- C1 minimality drop 0.0014 far below 0.05 target — 158-comp shortlist has heavy pairwise redundancy
- C2 modularity refuted — median_dominance 1.20 vs ≥2.0; only "fact" role has borderline signal (dissociation 0.089, p≈0.05)
- C3 sufficiency fails — reinsertion recovers only 0.113 LD (target ≥0.8) stably across 5 seeds; identifies necessary-but-not-sufficient regime
- M5.contingent (Gemma-2-27B) skipped — model not locally available; plan permits skip
- M4 / M4.stab used top-40 shortlist (not full 158) — documented in EXPERIMENT_RESULTS.md; per M2 dose-response top-31 already recovers 0.946 causal effect, so scope reduction is justified
- KL recovery denominator collapses on the anchor cell (tiny KL(clean||corrupt)=0.043 baseline) — LD/PD are the definitive metrics; documented
