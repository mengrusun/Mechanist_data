# Claim Ledger — Faithful SAGE reproduction — iterative, activation-grounded SAE-feature explanation vs. Neuronpedia

**Direction**: SAGE (multi-agent propose→test→revise pipeline over an SAE dictionary) produces natural-language explanations that outperform Neuronpedia on generative and predictive accuracy across multiple LLM+SAE pairs and early-to-late layers.
**Date**: 2026-07-14 → 2026-07-14
**Pipeline**: completed | **Iteration**: 6/10 "almost" (1/6)
**Models**: claim=claude-opus-4-7, experiment=claude-opus-4-7, verify=claude-sonnet-4-6, iteration=claude-opus-4-7
**Updated after**: iteration:final

| Claim | Main experiment | Verify | Post-Iteration | Final |
|-------|-----------------|--------|----------------|-------|
| C1 gen-acc main pair | partial (Δ=+0.018 CI[+0.005,+0.036]) — attribution-confounded | integrity WARN, swap deferred (cap) | narrowed: Neuronpedia-beat is backbone artifact, loop-only Δ≈0 | ⚪ integrity_only [WARN]; narrowed at iteration |
| C2 pred-acc main pair | not-supported (Δpearson=-0.018 CI[-0.094,+0.058]) | integrity PASS, swap deferred (cap) | reframed as failed replication | ⚪ integrity_only [PASS]; failed replication |
| C3 layer-depth generalization | not-supported (no per-layer CI excludes 0) | integrity WARN, swap deferred (cap) | reframed as unresolved-due-to-under-power | ⚪ integrity_only [WARN]; unresolved-under-power |
| C4 cross-pair generalization | inconclusive-positive-trend (Δpearson=+0.153 CI[-0.031,+0.341]) | robustness 1.00 model-swap PASS integrity PASS | reframed as verified negative transfer | ✓ verify PASS but scientifically NEGATIVE (verified negative transfer) |

---
## C1 — generative-accuracy on main pair
- **Statement**: On Gemma-2-2B + gemmascope-res-16k, SAGE explanations trigger the target feature more reliably than Neuronpedia's on paired same-feature-id probe texts (paired 95% CI lower bound > 0).
- **Origin**: task.md (given behavior) — generative-accuracy half of the SAGE claim
- **Data**: Neuronpedia activation corpus for gemmascope-res-16k — provenance=existing; available=16k features × 26 layers; used=44 features × 5 probe texts (planned 300×5); subset: under-powered due to DMXAPI throughput variance
- **Models**: Gemma-2-2B; gemmascope-res-16k; GPT-5 via DMXAPI
- **Method**: M1 — screen→decode via SAGE loop (Explainer→Designer→Analyzer→Reviewer)→paired generative-accuracy assay. Paired bootstrap 95% CI on (SAGE − Neuronpedia) AND (SAGE − matched-backbone GPT-5-1shot) for attribution.
- **Main experiment**: partial-support [provisional under-power] [attribution-confounded] — Δgen_acc (SAGE − Neuronpedia) = +0.018 CI[+0.005,+0.036] p=0.045; but Δgen_acc (SAGE − GPT-5-1shot) = -0.005 n.s.
- **Verify**: integrity WARN — Stage 2 swap-test deferred by max_verify_claims cap
- **Iteration**: score 6 'almost' — reviewer endorses honest reframing; no back-edge fired (type-⓪ only). Narrowed: SAGE's +1.8pp beat is over the specific public Neuronpedia catalog (older-backbone historical baseline), NOT over a matched-backbone control — the pipeline itself does not produce a significant loop-only gain at n=44.
- **Final**: ⚪ integrity_only [WARN — audit passed with warnings, swap-test deferred (max_verify_claims cap)]; iteration narrowed the claim: the Neuronpedia-beat is a backbone-upgrade artifact, matched-backbone loop-only Δ ≈ 0
- **Caveats**: suspected under-power (used_n 44/300); attribution-confound (SAGE vs Neuronpedia beat is a backbone artifact)
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M1, refine-logs/EXPERIMENT_RESULTS.md, results/m1/features_L{4,12,20}.jsonl, verify/C1_gen_acc_main_pair/main_experiment_audit/, review-stage/AUTO_ITERATION_FINAL_REPORT.md#C1
- **Figures**:

#### C1 headline metric — SAGE vs. Neuronpedia public catalog vs. matched-backbone GPT-5-1shot control on generative accuracy. The +1.8pp SAGE-vs-Neuronpedia gain (CI excludes 0) is not attributable to the SAGE loop: SAGE − GPT-5-1shot Δ ≈ 0.

**Table C1.** SAGE vs. Neuronpedia public catalog vs. matched-backbone GPT-5-1shot control on generative accuracy. The +1.8pp SAGE-vs-Neuronpedia gain (CI excludes 0) is not attributable to the SAGE loop: SAGE - GPT-5-1shot Δ ≈ 0.

| Comparison | Δ gen_acc | 95% CI | Wilcoxon p | n_features | Verdict |
| --- | --- | --- | --- | --- | --- |
| SAGE vs Neuronpedia (public GPT-4o-mini) | +0.018 | [+0.005, +0.036] | 0.045 | 44 | partial-support (backbone-attributable) |
| SAGE vs matched-backbone GPT-5-1shot | -0.005 | n.s. | n.s. | 44 | no advantage (loop-only null) |

Source `.tex`: `figures/C1/c1_comparison_table.tex`

## C2 — predictive-accuracy on main pair
- **Statement**: On Gemma-2-2B + gemmascope-res-16k, SAGE explanations yield higher per-feature Pearson correlation between explanation-conditioned activation predictions and ground-truth activations on Neuronpedia's held-out text than Neuronpedia's (paired 95% CI lower bound > 0).
- **Origin**: task.md (given behavior) — predictive-accuracy half of the SAGE claim
- **Data**: Neuronpedia corpus — used=44 features × 20 held-out texts (planned 300×20); ~15% of planned; same feature set as C1
- **Models**: Gemma-2-2B; gemmascope-res-16k; GPT-5 via DMXAPI
- **Method**: M1 — per-feature Pearson r; paired bootstrap 95% CI
- **Main experiment**: not-supported [provisional under-power] — Δpearson (SAGE − Neuronpedia) = -0.018, CI [-0.094, +0.058], p=0.43
- **Verify**: integrity PASS — Stage 2 swap-test deferred by max_verify_claims cap
- **Iteration**: score 6 'almost' — reviewer endorses reframing as failed replication / null result (paper-side type-⓪)
- **Final**: ⚪ integrity_only [PASS — audit clean, swap-test deferred (max_verify_claims cap)]; iteration reframes as failed-replication / null result
- **Caveats**: suspected under-power (used_n 44/300)
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M1, refine-logs/EXPERIMENT_RESULTS.md, results/m1/features_L{4,12,20}.jsonl, verify/C2_pred_acc_main_pair/main_experiment_audit/, review-stage/AUTO_ITERATION_FINAL_REPORT.md#C2
- **Figures**:

#### C2 predictive-accuracy result — no significant SAGE advantage in per-feature Pearson r over Neuronpedia (provisional at 15% of planned n).

**Table C2.** No significant SAGE advantage in per-feature Pearson r over Neuronpedia.

| Metric | SAGE | Neuronpedia | Δ (SAGE − Ref) | 95% CI | p-value | n |
| --- | --- | --- | --- | --- | --- | --- |
| per-feature Pearson r | n/a | n/a | -0.018 | [-0.094, +0.058] | 0.43 | 44 |

Source `.tex`: `figures/C2/c2_result_table.tex`

## C3 — layer-depth generalization
- **Statement**: The C1 and C2 improvements hold individually at early / mid / late residual-stream depths (L4, L12, L20) after Bonferroni correction over 3 depths × 2 metrics.
- **Origin**: task.md (given behavior) — layer-depth generalization
- **Data**: Neuronpedia corpus stratified by layer — used=14-15 per depth (planned 100/depth); ~15% of planned per-depth n
- **Models**: Gemma-2-2B; gemmascope-res-16k; GPT-5 via DMXAPI
- **Method**: M1 read-off — per-depth paired bootstrap 95% CI with Bonferroni α=0.05/6
- **Main experiment**: not-supported [provisional under-power] — no per-layer CI excludes zero at L4/L12/L20 for either metric at n=14-15/layer
- **Verify**: integrity WARN — Stage 2 swap-test deferred by max_verify_claims cap
- **Iteration**: score 6 'almost' — reviewer endorses reframing as unresolved-due-to-under-power (NOT evidence for absence of depth dependence)
- **Final**: ⚪ integrity_only [WARN — audit passed with warnings, swap-test deferred (max_verify_claims cap)]; iteration reframes as unresolved-due-to-under-power
- **Caveats**: suspected under-power (used_n 14-15 per depth vs 100 planned); L20 SAE l0 alignment looser (l0_71 variant available but not swapped in)
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M1, refine-logs/EXPERIMENT_RESULTS.md, verify/C3_layer_depth_generalization/main_experiment_audit/, review-stage/AUTO_ITERATION_FINAL_REPORT.md#C3
- **Figures**:

#### C3 layer-stratified read-off — per-layer n=14-15 features is too small to detect layer-specific effects at any of L4/L12/L20. All per-layer nulls provisional (Bonferroni α=0.05/6).

**Table C3.** Layer-stratified read-off (M1). All per-layer nulls provisional.

| Layer | n_features | Δ gen_acc (SAGE − Neuronpedia) | Δ Pearson r (SAGE − Neuronpedia) | Per-layer verdict |
| --- | --- | --- | --- | --- |
| L4 (early) | ~15 | not-significant | not-significant | null (under-powered) |
| L12 (mid) | ~15 | not-significant | not-significant | null (under-powered) |
| L20 (late) | ~14 | not-significant | not-significant | null (under-powered; l0_71 SAE variant untried) |

Source `.tex`: `figures/C3/c3_layer_table.tex`

## C4 — cross-LLM+SAE-pair generalization
- **Statement**: The C1 / C2 improvements hold on at least one cross-LLM+SAE pair from {Qwen3-4B + transcoder-hp, GPT-OSS-20B + resid-post-aa}; the second pair is deferred to /auto-verify.
- **Origin**: task.md (given behavior) — cross-LLM+SAE-pair generalization
- **Data**: Qwen3-4B + transcoder-hp (main M2) + GPT-OSS-20B + resid-post-aa (variant); used=34+45 features; scope-narrowed to SAGE-lite predictive-only on both cross-pairs
- **Models**: Qwen3-4B + transcoder-hp; GPT-OSS-20B + resid-post-aa; GPT-5 via DMXAPI
- **Method**: M2 — same M1 protocol scaled to 34 features on Qwen3-4B; verify model-swap ran same protocol on 45 features on GPT-OSS-20B at layers L3/L11/L19
- **Main experiment**: inconclusive-positive-trend [provisional under-power] [scope-narrowed] — Δpearson (SAGE-lite − Neuronpedia-proxy) = +0.153 CI[-0.031,+0.341] p=0.18 on Qwen3-4B
- **Verify**: robustness=1.00 — model swap PASS: GPT-OSS-20B variant Δ=-0.045 n.s.; integrity=PASS; verdict=PASS
- **Iteration**: score 6 'almost' — reviewer emphasizes this is a *verified negative transfer* result, NOT a SAGE-succeeds-cross-pair result. Falsified: any positive SAGE-lite advantage on cross-LLM+SAE-pair (both pairs non-significant in opposite directions).
- **Final**: ✓ verify PASS but scientifically NEGATIVE — verified negative transfer across two pairs (Qwen3-4B Δ=+0.153 n.s.; GPT-OSS-20B Δ=-0.045 n.s.); SAGE-lite fails to outperform Neuronpedia-proxy on either cross-pair. Do NOT frame as SAGE succeeding cross-pair.
- **Caveats**: suspected under-power (used_n 34/150 on main, 45 features on variant); scope-narrowed to SAGE-lite + predictive-only; full-SAGE + generative-accuracy on cross-pairs not yet tested
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M2, refine-logs/EXPERIMENT_RESULTS.md, results/m2/features.jsonl, verify/C4_cross_pair_generalization/main_experiment_audit/, verify/C4_cross_pair_generalization/variant_audit/, verify/C4_cross_pair_generalization/variants/model-swap-gpt-oss-20b/, verify/C4_cross_pair_generalization/ROBUSTNESS.md, review-stage/AUTO_ITERATION_FINAL_REPORT.md#C4
- **Figures**:

#### C4 cross-pair generalization — SAGE-lite predictive-accuracy on two architecturally distinct LLM+SAE pairs. Effect directions differ (+0.153 vs -0.045), only the non-significance is robust; no positive transfer observed.

**Table C4.** Cross-pair generalization — SAGE-lite predictive-accuracy on two architecturally distinct LLM+SAE pairs.

| Cross-pair | n_features | Δ Pearson r (SAGE-lite − Ref) | 95% CI | p-value | Verdict |
| --- | --- | --- | --- | --- | --- |
| Qwen3-4B + transcoder-hp (main M2) | 34 | +0.153 | [-0.031, +0.341] | 0.18 | not-significant (positive trend) |
| GPT-OSS-20B + resid-post-aa (variant) | 45 | -0.045 | [-0.210, +0.115] | n.s. | not-significant (negative direction) |

Source `.tex`: `figures/C4/c4_crosspair_table.tex`

---
## Journey Summary
- **Claim**: 1 faithful-capture idea (behavior_source=given) → 4 claims (C1-C4) targeting generative + predictive accuracy on the main pair, layer-depth generalization, and cross-LLM+SAE-pair generalization.
- **Mechanism strategy**: directions: [Unit Interpretation]
- **Mechanism routing**: family=Feature Dictionary Learning / SAE, submethod=SAE-feature auto-interpretation (Explainer→Designer→Analyzer→Reviewer loop) — Screen: none (SAE feature id IS the located unit) · Decode: SAGE loop · Verify: paired generative + predictive assays · Recover: layer-strat + cross-pair
- **Experiment**: 5 runs, 1.66 GPU-hours (of 10-hour budget), headline: mixed — C1 partial (attribution-confounded), C2/C3 null (under-powered), C4 positive-trend (under-powered)
- **Verify**: 4 claims: 1 PASS (C4) / 0 FAIL / 0 INCONCLUSIVE / 0 ZEV / 3 INTEGRITY_ONLY (cap=3, swap_off=0); integrity[Phase2/Phase9]: WARN/PASS
- **Iteration**: 1/6 iteration (0/2 claim-reentries consumed — only type-⓪ narrative-only), score 6/10 verdict almost, termination=positive_verdict (three-dimensional STOP rule fired on entry)
- **Figures**: 4 tables across 4 claims; 0 judgment-skipped; 0 render-skipped, 0 errored

## Open Items
- Attribution confound: single-pass GPT-5 already outscores Neuronpedia's GPT-4o-mini baseline, so C1's beat over Neuronpedia is NOT attributable to the SAGE loop itself. Reviewer type-⓪ paper-side recommendation: split into C1a (vs public Neuronpedia older-backbone catalog) + C1b (vs matched-backbone GPT-5-1shot, promote in abstract).
- DMXAPI throughput variance (occasional 10-20 min stalls per call) was the wall-clock bottleneck — realized n at ~15% of planned for M1, ~23% for M2. All four claims tagged suspected_under_power (UNDERPOWER=tag).
- C1 verify swap-test deferred (max_verify_claims cap): upgrade via `/auto-verify C1 — resume: true, swap-variants: true`. **Highest reviewer priority** (~1 GPU-hr) — closes the attribution-confound argument.
- C2 verify swap-test deferred (max_verify_claims cap): upgrade via `/auto-verify C2 — resume: true, swap-variants: true`. Second priority.
- C3 verify swap-test deferred (max_verify_claims cap): upgrade via `/auto-verify C3 — resume: true, swap-variants: true`. Lowest priority (underlying issue is under-power); combine with larger n_features/layer M1 rerun + L20 l0_71 SAE swap.
- L20 SAE l0 alignment looser than L4/L12; l0_71 variant available but not swapped in — potential noise source on C3-L20.
- M2 scope-narrowed at experiment stage to predictive-only + SAGE-lite (no Designer+Analyzer) on Qwen3-4B; full-SAGE + generative-accuracy on Qwen3-4B never tested.
- Paper-side wording checklist (all type ⓪, execution is paper author's responsibility) — see `review-stage/AUTO_ITERATION_FINAL_REPORT.md` §Paper-side wording checklist.
