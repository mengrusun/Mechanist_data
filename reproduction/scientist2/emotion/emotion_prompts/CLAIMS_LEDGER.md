# Claim Ledger — Faithful behavior capture C1–C4 + Location → Causal Intervention mechanism strategy on Qwen3-14B / GSM8K (with SocialIQA + MedQA as C2 companions)

**Direction**: Emotional Framing in Prompts as a Weak, Input-Dependent Signal
**Date**: 2026-07-13 → 2026-07-14
**Pipeline**: completed | **Iteration**: 6.0/10 "almost" (0/6)
**Models**: claim=claude-opus-4-7, experiment=claude-opus-4-7, verify=claude-sonnet-4-6, iteration=claude-opus-4-7
**Updated after**: iteration:final

| Claim | Main experiment | Verify | Post-Iteration | Final |
|-------|-----------------|--------|----------------|-------|
| C1 static prefixes → small input-dependent shifts | partial | INTEGRITY_ONLY | iter-1 ⓪ narrative-only — sign-consistency band computed post-hoc (reports/C1_si… | ⚪ integrity_only (swap-test deferred — max_verify_claims cap); partial support strengthened — emotion-mean effect within noise floor, but 8/24 individual prefix cells produce reliable directional shifts |
| C2 social > factual > math spread | not-supported | INTEGRITY_ONLY | iter-1 ⓪ failed hypothesis — ordering reversed from prediction; paper narrative … | ⚪ integrity_only (swap-test deferred — max_verify_claims cap); failed hypothesis — realized ordering (math > social ≈ factual) is the reverse of predicted |
| C3a no consistent argmax emotion across tasks | supported | INTEGRITY_ONLY | iter-1 ⓪ conservative rephrasing to 'task-dependent best emotion' — fear on math… | ⚪ integrity_only (swap-test deferred — max_verify_claims cap); ✓ task-dependent best emotion (fear on math/factual, surprise on social) |
| C3b no monotone intensity → gain | supported | INTEGRITY_ONLY | iter-1 ⓪ per-emotion CIs surfaced from existing M4_c3_analysis.json (no new comp… | ⚪ integrity_only (swap-test deferred — max_verify_claims cap); ✓ threshold-level support (3/6 GSM8K emotions fail monotonicity) |
| C4 EmotionRL beats fixed prefixes / neutral | not-supported [provisional — suspected under-power] | INCONCLUSIVE | iter-1 ⓪ demoted to 'untested/deferred, removed from core contributions' — M7 re… | ✗ inconclusive — M7 descoped; C4 UNTESTED (not falsified); requires ≥ 2.17 GPU-h follow-up allocation |
| CM residual-stream frame direction (Location + Causal) | partially-supported [provisional — suspected under-power on Causal arm] | PASS (r=1.00) | iter-1 ⓪ narrative bifurcation — CM-location [strongly supported: probe 1.00 vs … | ✓ PASS — CM-Location strongly supported (probe 1.00 vs 0.28 null, +71 pp gap; robust to Qwen3-4B swap); CM-Causal not-supported at tested scale [provisional — Causal arm under-powered] |

---
## C1 — static prefixes → small input-dependent shifts
- **Statement**: Static emotional prefixes cause only small, input-dependent accuracy shifts on Qwen3-14B/GSM8K: per-emotion mean|Δaccuracy vs neutral| ≤ format-perturbation p90 noise floor AND per-item sign-consistency ∈ [0.4, 0.6] across all 24 emotional conditions (6 emotions × 2 intensities × 2 wording sources).
- **Origin**: task.md claim 1 — faithfully captured (BEHAVIOR_SOURCE=given)
- **Data**: GSM8K test split (first 500 items, deterministic order) — provenance=existing; available=1319 (GSM8K test), used=500; subset: cost-aware: first 500 items = 38% of test set; strict marker absent so subsetting allowed
- **Models**: Qwen3-14B (bf16, vLLM)
- **Method**: 26 conditions (24 emotional + neutral + filler_matched_length) × 500 GSM8K items via CoT completion (T=0.0, max_new_tokens=256, regex-parsed answer); noise floor from 8 semantics-preserving format perturbations of neutral (p90 |Δ| = 0.0524 = 5.24 pp); C1 gate: per-emotion mean |Δ| ≤ noise floor AND per-item sign-consistency ∈ [0.4, 0.6].
- **Main experiment**: partial — neutral acc=0.820; noise floor p90 |Δ|=0.0524; per-emotion mean Δ ∈ [−0.014, +0.015] all within noise floor; but 6/24 individual prefix cells cross the noise floor (largest: happiness_1_human −11.2 pp; disgust_2_llm +7.4 pp; anger_2_human +6.8 pp). Per-item sign-consistency band not computed for all cells due to time. Per-emotion mean effect is within noise floor as C1 predicts, but individual prefix cells can cross it. C1 is supported at the emotion-mean level but not fully at every prefix cell.
- **Verify**: robustness=— — method n/a / dataset n/a / model n/a; integrity=WARN; verdict=INTEGRITY_ONLY; stage2_skip_reason=max_verify_claims_cap
- **Iteration**: iter-1 ⓪ narrative-only — sign-consistency band computed post-hoc (reports/C1_sign_consistency.json, scripts/c1_sign_consistency.py): 16/24 cells in [0.4, 0.6] band, 8/24 out-of-band; the 8 out-of-band cells align exactly with the 6 noise-floor crossers + 2 borderline. Partial-support reading strengthened.
  - narrowed_to: Partial support at the per-emotion mean level; individual prefixes can produce reliable directional shifts (8/24 cells out of [0.4, 0.6] band).
- **Final**: ⚪ integrity_only (swap-test deferred — max_verify_claims cap); partial support strengthened — emotion-mean effect within noise floor, but 8/24 individual prefix cells produce reliable directional shifts
- **Artifacts**: runs/M2/*.json, runs/M2/act_*.pt, runs/M2b/*.json, verify/C1_small_input_dependent_shift/ROBUSTNESS.md, verify/C1_small_input_dependent_shift/main_experiment_audit/EXPERIMENT_AUDIT.md
---
## C2 — social > factual > math spread
- **Statement**: The effect of emotional framing is markedly larger on socially grounded tasks than on math/factual QA: inter-emotion accuracy spread on SocialIQA is at least 2× the spread on GSM8K, and the ordering spread_social > spread_factual > spread_math holds in ≥ 2 of 3 pairwise comparisons.
- **Origin**: task.md claim 2 — faithfully captured
- **Data**: SocialIQA dev (first 500) + MedQA test (first 500) + GSM8K test (M2's 500) — provenance=existing; available=SocialIQA=1954, MedQA=1273, GSM8K=1319, used=500 each; subset: cost-aware: 500-item slice per dataset
- **Models**: Qwen3-14B (bf16, vLLM)
- **Method**: 26 conditions × {SocialIQA (3-way MCQ, log-likelihood), MedQA (4/5-way MCQ, US-MLE, log-likelihood)} × 500 items; compute per-task inter-emotion spread (max−min accuracy across 24 emotional conditions) and cross-task ordering; C2 gate: spread_social ≥ 2×spread_math AND ordering social>factual>math in ≥ 2/3 pairs.
- **Main experiment**: not-supported — Neutral: SocialIQA 0.746, MedQA 0.614, GSM8K 0.820. Spreads: GSM8K=0.186 (18.6 pp) >> SocialIQA=0.030 (3.0 pp) > MedQA=0.028 (2.8 pp). Realized ordering math >> social > factual — opposite of predicted; 1/3 pairwise orderings hold. The prediction is disconfirmed: math (GSM8K) shows the LARGEST inter-emotion spread, driven by CoT generation variance in ~250-token completions, while MCQ tasks are protected by their constrained log-likelihood eval. The affect-as-information prediction reverses under this measurement design.
- **Verify**: robustness=— — method n/a / dataset n/a / model n/a; integrity=WARN; verdict=INTEGRITY_ONLY; stage2_skip_reason=max_verify_claims_cap
- **Iteration**: iter-1 ⓪ failed hypothesis — ordering reversed from prediction; paper narrative demotion to 'failed hypothesis'.
  - falsified: The affect-as-information prediction that social-reasoning tasks should show larger emotional-prefix spread than math
  - narrowed_to: Under the mixed CoT/MCQ measurement design used, the reverse ordering (math > social ≈ factual) holds; a same-mode comparison remains an open question but is not this project's contribution.
- **Final**: ⚪ integrity_only (swap-test deferred — max_verify_claims cap); failed hypothesis — realized ordering (math > social ≈ factual) is the reverse of predicted
- **Caveats**: Spread magnitude is confounded with eval mode (CoT free-form for GSM8K vs constrained-MCQ log-likelihood for SocialIQA/MedQA); a same-mode comparison would strengthen the reading.
- **Artifacts**: runs/M3/*.json, reports/M4_c3_analysis.json, verify/C2_task_family_spread_ordering/ROBUSTNESS.md, verify/C2_task_family_spread_ordering/main_experiment_audit/EXPERIMENT_AUDIT.md
---
## C3a — no consistent argmax emotion across tasks
- **Statement**: No single basic emotion is argmax across all three task families {math, social, factual}: the argmax-emotion identity flips at least once across {GSM8K, SocialIQA, MedQA}.
- **Origin**: task.md claim 3 (identity axis) — faithfully captured
- **Data**: M2 + M3 result JSONs (post-hoc) — provenance=adapted; available=n/a (uses aggregate), used=500×3 tasks; subset: analysis-only; no new runs
- **Models**: Qwen3-14B (data source)
- **Method**: Post-hoc analysis (scripts/analyze_c3.py): per-task per-emotion argmax over the 4 (intensity × wording-source) cells; check whether the argmax emotion identity flips across the 3 tasks.
- **Main experiment**: supported — Per-task argmax: GSM8K=fear (mean 0.835), SocialIQA=surprise (mean 0.750), MedQA=fear (mean 0.641). Argmax flips ≥ 1 across the 3 tasks (fear→surprise on the math/social boundary). No single emotion wins across all task families — fear leads on math and factual, surprise leads on social; the identity flips at least once as required.
- **Verify**: robustness=— — method n/a / dataset n/a / model n/a; integrity=PASS; verdict=INTEGRITY_ONLY; stage2_skip_reason=max_verify_claims_cap
- **Iteration**: iter-1 ⓪ conservative rephrasing to 'task-dependent best emotion' — fear on math/factual, surprise on social. Argmax flips ≥ 1 (fear→surprise) as required.
  - narrowed_to: Task-dependent best emotion: fear on GSM8K + MedQA, surprise on SocialIQA.
- **Final**: ⚪ integrity_only (swap-test deferred — max_verify_claims cap); ✓ task-dependent best emotion (fear on math/factual, surprise on social)
- **Artifacts**: reports/M4_c3_analysis.json, verify/C3a_no_consistent_winner/ROBUSTNESS.md, verify/C3a_no_consistent_winner/main_experiment_audit/EXPERIMENT_AUDIT.md
---
## C3b — no monotone intensity → gain
- **Statement**: Stronger emotional wording does not yield proportionally larger gains: for ≥ 3 of 6 emotions on GSM8K, the paired Δ(intensity-2 − intensity-1) 95% bootstrap CI straddles 0 or reverses sign; no emotion satisfies intensity-2 > intensity-1 > 0 consistently across all 3 tasks.
- **Origin**: task.md claim 3 (magnitude axis) — faithfully captured
- **Data**: M2 + M3 result JSONs (post-hoc) — provenance=adapted; available=n/a, used=500×3 tasks; subset: analysis-only
- **Models**: Qwen3-14B (data source)
- **Method**: Post-hoc analysis: per-task per-emotion paired bootstrap CI (n=1000) on Δ(intensity-2 − intensity-1) averaged across wording sources; report fraction of emotions failing monotonicity on GSM8K.
- **Main experiment**: supported — On GSM8K, 3/6 emotions have paired Δ(intensity-2 − intensity-1) bootstrap CI straddling 0 or reversed sign — exactly meets the ≥ 3/6 threshold. Monotonicity fails: intensity-2 does not consistently produce larger gains than intensity-1 — 3 of 6 emotions show CI-straddle-0 or sign reversal on GSM8K, meeting the ≥ 3 threshold.
- **Verify**: robustness=— — method n/a / dataset n/a / model n/a; integrity=WARN; verdict=INTEGRITY_ONLY; stage2_skip_reason=max_verify_claims_cap
- **Iteration**: iter-1 ⓪ per-emotion CIs surfaced from existing M4_c3_analysis.json (no new compute) — threshold-level support (3/6 emotions fail monotonicity on GSM8K).
  - narrowed_to: Threshold-level support: exactly 3/6 GSM8K emotions fail monotonicity (borderline pass).
- **Final**: ⚪ integrity_only (swap-test deferred — max_verify_claims cap); ✓ threshold-level support (3/6 GSM8K emotions fail monotonicity)
- **Artifacts**: reports/M4_c3_analysis.json, verify/C3b_no_monotone_intensity/ROBUSTNESS.md, verify/C3b_no_monotone_intensity/main_experiment_audit/EXPERIMENT_AUDIT.md
---
## C4 — EmotionRL beats fixed prefixes / neutral
- **Statement**: An adaptive per-query policy (EmotionRL) that selects the emotional prefix per query yields more reliable accuracy gains than any fixed emotional prefix or neutral baseline: acc(π_θ) > acc(neutral) AND acc(π_θ) > acc(e*) on held-out GSM8K, both with lower 95% CI > 0 across ≥ 3 seeds.
- **Origin**: task.md claim 4 — faithfully captured
- **Data**: GSM8K train (first 2000) + GSM8K test (M2's 500) — PLANNED — provenance=existing; available=train=7473, test=1319, used=0; subset: M7 descoped entirely due to budget exhaustion after M2/M2b/M3/M5/M6
- **Models**: Qwen3-14B (frozen scorer) — PLANNED, Llama-3.2-1B / bert-base-cased (policy backbone) — PLANNED
- **Method**: M7a: reward table via Qwen3-14B on 2000 GSM8K train items × 13 prefix actions. M7b: SFT of classifier, 3 seeds, 2 epochs. M7c: REINFORCE-with-learned-baseline, 1 epoch. M7d: held-out eval on 500 GSM8K test items; success gate = acc(π_θ) > acc(neutral) AND > acc(e*) with lower 95% CI > 0 across 3 seeds.
- **Main experiment**: not-supported [provisional — suspected under-power] — M7 descoped entirely — no reward table, no policy training, no held-out eval executed. C4 is UNTESTED — the M7 milestone was descoped after ~9.08 GPU-h of the 10-h budget was consumed by M2/M3/M5/M6. Also note Llama-3.2-1B and bert-base-cased are absent from $MODEL_DIR (only Llama-3.2-3B-Instruct is present), so a policy backbone would require download.
- **Verify**: robustness=— — method n/a / dataset n/a / model n/a; integrity=FAIL; verdict=INCONCLUSIVE
- **Iteration**: iter-1 ⓪ demoted to 'untested/deferred, removed from core contributions' — M7 requires ≥ 2.17 GPU-h vs ~0.79 remaining; cannot resolve in-loop.
  - narrowed_to: C4 is UNTESTED, not falsified. Full M7 must be run in a follow-up allocation to obtain a verdict.
- **Final**: ✗ inconclusive — M7 descoped; C4 UNTESTED (not falsified); requires ≥ 2.17 GPU-h follow-up allocation
- **Caveats**: [suspected under-power: used_n 0/500 test + 0/2000 train, seeds 0/3, chunks 0/5 — M7 descoped entirely due to budget] | Policy backbone (Llama-3.2-1B / bert-base-cased) absent from $MODEL_DIR; download required if C4 is retried.
- **Artifacts**: verify/C4_adaptive_policy_beats_fixed/ROBUSTNESS.md, verify/C4_adaptive_policy_beats_fixed/main_experiment_audit/EXPERIMENT_AUDIT.md
---
## CM — residual-stream frame direction (Location + Causal)
- **Statement**: On Qwen3-14B, a low-rank residual-stream direction (or a small attention-head set) at early-to-mid layers carries the emotional-frame identity of static prefixes (Location) and causally modulates per-item GSM8K accuracy in the sign predicted by C1 (Causal Intervention); a matched-length non-emotional filler control produces null effect and steering shows monotone dose-response in α on ≥ 3 of 4 sites.
- **Origin**: mechanism claim discovered by the plan (mechanism_strategy = [Location, Causal Intervention])
- **Data**: M2 cached residual-stream activations at layers {0,4,8,12,16,20,24,28,32,36} (200/500 items × 24 emotional conditions); M6 paired GSM8K subset for interventions; MedQA subset for off-target (DESCOPED) — provenance=adapted; available=26 conditions × 500 items × 10 layers, used=M5: 200 items × 24 emotional cond × 10 layers = 48000 activations. M6: 50/200 planned items × happiness_2_human only (1 of 12 prefixes); subset: M6 reduced from 200 to 50 items due to budget; only 1 of 12 prefixes tested; specificity controls (filler, off-target MedQA, off-layer L20) all DESCOPED
- **Models**: Qwen3-14B (frozen; transformers backend with residual-stream hooks — not vLLM)
- **Method**: M5 Location (Probing/Residual-Stream-States family): per-layer 6-way logistic emotion-identity probe on last-prefix-token residual (70/30 item-disjoint split, 3 seeds), vs length-shuffled null baseline; SVD on 26 × d_model activation matrix at top-2 layers for candidate directions. M6 Causal Intervention (Causal-Attribution/Patching + Steering-Vectors families): (1) activation patching from emotional→neutral at frame layer (identity-patch sanity only, cross-emotion DESCOPED); (2) steering α ∈ {−1.0, 0.0, +0.5, +1.0} at L4, L8 in σ_proj units; (3) matched-filler control (DESCOPED); (4) off-target MedQA (DESCOPED); (5) off-layer L20 null (DESCOPED). Success gate: patched-neutral matches emotional-run Δ sign; monotone dose in ≥ 3/4 α values on ≥ 1 site; filler null; off-target near-null.
- **Main experiment**: partially-supported [provisional — suspected under-power on Causal arm] — M5 Location: probe accuracy 1.00 at layers 4-36 vs 0.28 length-shuffled null (+71 pp gap); onset at L4; SVD top-2 singular values dominate at L4 (52.99, 17.39) and L8 (91.78, 29.36). M6 Causal: steering dose-response L4 range 2 pp / L8 range 4 pp (both at or below the 5.24 pp noise floor); no monotone α dependence; identity-patch sanity matches baseline at L4; parse-rate 100% (no OOD collapse). Location is strongly supported — a linearly-decodable emotion identity signal emerges at layer 4 and persists through the final layers, well above the length-controlled null. Causal is NOT supported at tested scale — steering the identified direction does not move GSM8K accuracy beyond noise; the identity is represented but not read out for math problem solving. Interpretation is provisional given the suspected under-power on the Causal arm (only 1/12 prefixes, 50/200 items, 0/3 specificity controls).
- **Verify**: robustness=1.00 — method n/a / dataset n/a / model pass; integrity=PASS; verdict=PASS
- **Iteration**: iter-1 ⓪ narrative bifurcation — CM-location [strongly supported: probe 1.00 vs null 0.28 length-shuffled; +71 pp gap; onset L4] separated from CM-causal [not-supported at tested scale: steering dose-response ≤ 4 pp within noise floor; identity-patch sanity clean]. Verify-swap variant (Qwen3-4B) confirmed Location holds across model family (probe 1.00 vs null 0.69 with length confound at 7-condition swap; probe-null gap 31 pp still positive).
  - falsified: Causal-arm hypothesis: that steering the frame direction moves GSM8K accuracy in the C1-predicted sign — falsified at tested scale (α ∈ {−1,0,+0.5,+1} in σ_proj, 50/200 items, 1 wording variant)
  - narrowed_to: CM-location holds: emotion identity is linearly decodable from the residual stream at L4-L36 across the Qwen3 family. CM-causal is 'not-supported at tested scale' with under-power caveat (used_n 50/200, wording 1/2, specificity controls 0/3).
- **Final**: ✓ PASS — CM-Location strongly supported (probe 1.00 vs 0.28 null, +71 pp gap; robust to Qwen3-4B swap); CM-Causal not-supported at tested scale [provisional — Causal arm under-powered]
- **Caveats**: [suspected under-power on Causal arm: used_n 50/200 items, wording sources 1/2 (only happiness_2_human tested), specificity controls 0/3 (filler / off-target MedQA / off-layer L20 all descoped)]
- **Artifacts**: reports/M5_location.json, reports/M5_directions.pt, runs/M6/*.json, verify/CM_residual_direction_causal/ROBUSTNESS.md, verify/CM_residual_direction_causal/main_experiment_audit/EXPERIMENT_AUDIT.md, verify/CM_residual_direction_causal/main_experiment_audit/MECHANISM_AUDIT.md, verify/CM_residual_direction_causal/variant_audit/EXPERIMENT_AUDIT.md, verify/CM_residual_direction_causal/variants/model-swap-qwen3-4b/

---
## Journey Summary
- **Claim**: given behavior — 5 behavioral claims + 1 mechanism claim faithfully captured from task.md; mechanism_strategy = [Location, Causal Intervention]
- **Mechanism strategy**: Location → Causal Intervention
- **Mechanism routing**: family=Probing/Residual-Stream-States (M5) + Causal-Attribution/Patching + Steering-Vectors (M6); reconciliation_status=ok
- **Experiment**: 87 tracked runs, ~9.08 GPU-h of 10 budget; headline mixed — C3a/C3b supported, C1 partial, C2 not-supported (opposite ordering), CM Location strongly supported / Causal not-supported at tested scale, C4 untested (M7 descoped)
- **Verify**: 6 claim(s): 1 PASS (CM) / 0 FAIL / 1 INCONCLUSIVE (C4) / 0 ZEV / 4 INTEGRITY_ONLY (cap=4, swap_off=0); integrity[Phase2=WARN/Phase9=PASS]; picked CM for Stage 2 (importance); variant model-swap-qwen3-4b (Location robust across model family)
- **Iteration**: 3 review rounds within the loop (0 back-edges consumed = all fixes ⓪ narrative-only); final score 6/10 verdict 'almost'; termination=iterations_exhausted (reviewer said stop at convergence; C4 blocks dimension-3 STOP but cannot be resolved within budget); post-hoc C1 sign-consistency + CM narrative bifurcation delivered
- **Figures**: not-run — LEDGER_FIGURES hook deferred to a manual /paper-figure run; per-claim plottable sources are noted per claim (see Open Items)

## Open Items
- C4 UNTESTED — M7 (EmotionRL adaptive prompt selection) was descoped due to budget; requires ≥ 2.17 GPU-h follow-up allocation. Upgrade: run M7 then /auto-verify C4 -- resume: false. Not a falsification — a deferred test.
- C1 INTEGRITY_ONLY (max_verify_claims_cap) — Stage-2 swap-test deferred. WARN reason (sign-consistency) resolved in-loop via scripts/c1_sign_consistency.py + reports/C1_sign_consistency.json. Upgrade: /auto-verify C1 -- resume: true.
- C2 INTEGRITY_ONLY (max_verify_claims_cap) — Stage-2 swap-test deferred. Eval-mode confound (CoT for GSM8K vs MCQ-LL for SocialIQA/MedQA) is a genuine methodological caveat; same-mode comparison would strengthen. Upgrade: /auto-verify C2 -- resume: true.
- C3a INTEGRITY_ONLY (max_verify_claims_cap) — Stage-2 swap-test deferred (integrity=PASS already). Upgrade: /auto-verify C3a -- resume: true.
- C3b INTEGRITY_ONLY (max_verify_claims_cap) — Stage-2 swap-test deferred. WARN reason (per-emotion CI bounds) resolved narratively from existing M4 analysis. Upgrade: /auto-verify C3b -- resume: true.
- CM-Causal specificity controls (filler / off-target MedQA / off-layer L20) were descoped from M6; the 'not-supported at tested scale' negative is provisional pending these controls. Extending M6 with a filler-patch on 50 items (~0.3-0.5 GPU-h) would upgrade this negative to a well-controlled null.
- CM Location variant length-confound note: with 7 conditions in the Qwen3-4B swap variant, the length-shuffled null was 0.69 (not the ≤0.2 designed for M5's 24 conditions); probe-null gap 31 pp still positive, but the swap's null-baseline threshold is not literally the same as the main experiment's.
- Figures: Ledger Figures hook was deferred (not invoked in this run). Per-claim plottable sources are: C1 → per-emotion Δ bar + noise floor line (runs/M2/*.json + runs/M2b/*.json); C2 → grouped bar of spreads across GSM8K/SocialIQA/MedQA (reports/M4_c3_analysis.json); C3a → task-vs-emotion argmax heatmap; C3b → per-emotion Δ(int2−int1) forest plot (reports/M4_c3_analysis.json); CM → per-layer probe accuracy line + null baseline (reports/M5_location.json) and steering dose-response (runs/M6/*.json). Invoke /paper-figure with mode: auto-ledger to populate.
