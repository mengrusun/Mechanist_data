# Claim Ledger — Emotion-Specific Global Circuits in Llama-3.2-3B-Instruct (SEV)

**Direction**: Localisable emotion circuits in LLMs (from task.md): identify per-emotion component sets, prove they are causal + stable, and use them for reliable emotion control that beats prompting and single-direction steering.
**Date**: 2026-07-13 → 2026-07-14
**Pipeline**: completed | **Iteration**: 6/10 "ready" (2/6)
**Models**: claim=claude-opus-4-7, experiment=claude-opus-4-7, verify=claude-sonnet-4-6, iteration=claude-opus-4-7
**Updated after**: iteration:final

| Claim | Main experiment | Verify | Post-Iteration | Final |
|-------|-----------------|--------|----------------|-------|
| C1 Localizability | supported | ⚪ INTEGRITY_ONLY (WARN) | ⓪ narrowed to coarse-locatability scope | ✓ supported at coarse-locatability scope |
| C2 Causal + Stability | not-supported | ⚪ INTEGRITY_ONLY (HIGH WARN) | ② operator fix confirmed negative is REAL | ✗ not-supported (real negative) |
| C3 Applied beats prompting + steering | not-supported | ✗ FAIL (robustness 1.0) | ② α+components fix + ③ rewrite → C3_v2 | ✗ falsified — superseded by C3_v2 |
| C3_v2 Prefix modulation + open-generation-control negative (NEW from ③) | — | n/a | ⓪ tightening + reframing to falsification-study | ✓ supported at narrowed scope (not stress-tested) |

---
## C1 — Localizability of per-emotion component set C_e
- **Statement**: The framework produces stable, sparse, per-emotion component sets C_e on Llama-3.2-3B-Instruct across SEV: |C_e| meets sparsity floor (k_h ≤ 96, k_n ≤ 8000); per-emotion Jaccard exceeds size-matched permutation-null 95% CI on ≥ 5/6 emotions. [Narrowed at iteration 4 (⓪) to coarse-locatability scope: 'discriminative, emotion-relevant regions/layers with stable component-set structure', NOT 'precise identification of the causal subset'.]
- **Origin**: task.md claim 1 (Localizability) — captured given; refined into Block B1 (M1 Location); scope narrowed at iteration 4 (⓪).
- **Data**: SEV — /data/zhenqian/data/SEV/sev.json — provenance=existing; available=2880 pairs, used=2880 pairs (Stage A train 1440 fit + Stage B val 720 + 3 folds × 80% train for Jaccard)
- **Models**: Llama-3.2-3B-Instruct
- **Method**: Stage-A per-layer per-emotion linear probe (ITI-style) + MLP-neuron cosine alignment; Stage-B per-shortlisted-component single-component-enhancement target-prefix log-prob gain — Stage-A screen → Stage-B causal-verify composition; sparsity grid k_h ∈ {24,48,96}, k_n ∈ {2000,4000,8000}, 3-fold event-subsample Jaccard vs. 200-draw permutation null, random top-k control.
- **Main experiment**: supported — (k_h*, k_n*) = (24, 2000); Stage-B macro gain 2.136 nats; head Jaccard 0.947-1.000 vs. null ~0.27; neuron Jaccard 0.972-0.983 vs. null ~0.046; 6/6 pass both axes.
- **Verify**: robustness=n/a — method n/a / dataset n/a / model excluded; integrity=WARN; verdict=INTEGRITY_ONLY (stage2_skip_reason=max_verify_claims_cap)
- **Iteration**: supported at coarse-locatability scope (⓪ caveats only); scope narrowed to "discriminative emotion-relevant regions/layers with stable component-set structure (NOT precise causal subset identification)"; no ①/②/③ actions used
- **Final**: ✓ supported at coarse-locatability scope (Jaccard 6/6 both axes; audit passed with 2 WARNs — flat kstar grid + head Stage-B degenerate layer-level; swap-test deferred by max_verify_claims cap)
- **Caveats**: flat kstar grid: k* landed at smallest cell — grid may not have exercised sparsity/floor tradeoff sufficiently; head Stage-B degenerate at layer level; Localization here should be read as 'discriminative, emotion-relevant regions/layers with stable component-set structure', NOT as 'precise identification of the causal subset'
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#block-1, refine-logs/EXPERIMENT_RESULTS.md#m1, runs/A1_location/, verify/C1_localizability_per_emotion_component_sets_Ce/, review-stage/AUTO_ITERATION_FINAL_REPORT.md#4b1

---
## C2 — Causal + Stability of C_e
- **Statement**: C_e causally carries emotion generation and is specific + scenario-stable on Llama-3.2-3B-Instruct: ablation weakens target at α2, enhancement strengthens it with monotonic dose-response (Spearman ≥ 0.7 over 3 α); specificity holds against three complementary controls (random-set null, targeted-C_{e'}, off-target scoring); scenario Jaccard(S1,S2) exceeds permutation null.
- **Origin**: task.md claim 2 (Mechanistic evidence — traceable, stable circuits) — captured given; refined into Block B2; operator bug fix + re-run in iteration 1 (②).
- **Data**: SEV eval fold — /data/zhenqian/data/SEV/sev.json — provenance=existing; available=720 eval pairs, used=720 eval pairs (full) — original + full re-run under fixed per-stem operator
- **Models**: Llama-3.2-3B-Instruct
- **Method**: Ablation (per-stem mean-substitute — FIXED in iteration 1 from buggy global-mean formulation) + Enhancement (α ∈ {0.5,1.0,2.0}) + 3 specificity controls (100-draw random-set null — enlarged pool to 72 heads / 4915 neurons in iter-1, targeted-C_{e'}, off-target) + Scenario stability. Rubric: {full, partial, causal-only, not-supported}.
- **Main experiment**: not-supported — Rubric 0 full / 0 partial / 0 causal-only / 6 not-supported. Ablation Δ target = +0.510 to +0.928 (WRONG SIGN); enhancement Δ target at α=1.0 = +1.34 to +5.83 (correct sign); Spearman(α, Δ_enh) < 0.7 on 4/6.
- **Verify**: robustness=n/a — method n/a / dataset n/a / model excluded; integrity=WARN (HIGH — operator scope bug flagged); verdict=INTEGRITY_ONLY (stage2_skip_reason=max_verify_claims_cap)
- **Iteration**: not-supported (real negative confirmed by iteration-1 ② operator fix + enlarged random-null pool). Post-fix integrity now PASS. Iteration-1 re-run: ablation Δ REMAINS POSITIVE on all 6 emotions with per-stem operator (0.481-0.927 nats, essentially unchanged from buggy 0.510-0.928). Falsified: operator-bug hypothesis. Narrowed: for 3/6 emotions (joy, sadness, anger) random top-K sets outperform C_e in random-null — weakens unique-causal-privilege at subset level.
- **Final**: ✗ not-supported (real negative confirmed by ② fix; positive Δ under ablation is real, not operator artifact; the located components are emotion-relevant/discriminative but not strongly verified causal necessities under this intervention)
- **Caveats**: Positive Δ under ablation admits two non-exclusive interpretations — not directionally causal, OR mean-substitution removes suppressive/non-directional signal that net-raises target; for 3/6 emotions random top-K sets outperform C_e; swap-test deferred (post-fix integrity now PASS)
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#block-2, refine-logs/EXPERIMENT_RESULTS.md#m2, runs/A2_causal/, runs/iteration_round_1/A2_causal_fixed/, verify/C2_causal_stability_of_Ce/, experiments/m2_causal.py, review-stage/AUTO_ITERATION_FINAL_REPORT.md#4b2

---
## C3 — Applied circuit control beats prompting + single-direction steering  [SUPERSEDED by C3_v2]
- **Statement**: [SUPERSEDED at iteration 2 by C3_v2 via ③] Original: Arm A > Arm B on ≥ 5/6 emotions AND Arm A > Arm C on ≥ 5/6 emotions under hidden-target 6-way judged accuracy.
- **Origin**: task.md claim 3 (Applied control) — captured given; refined into Block B3 + B4; superseded by C3_v2 in iteration 2.
- **Data**: SEV eval fold — 2160 continuations + 19,440 val forwards; used=2160/2160 judged (main experiment) + 2160/2160 judged (iter-1 fix)
- **Models**: Llama-3.2-3B-Instruct, Qwen2.5-7B-Instruct (M4 verify-swap)
- **Method**: Three matched-budget arms. Iter-1 (②): α {0.5,1.0,2.0}→{0.05,0.1,0.3}; components (k_h, k_n) neighborhood shifted down to {5,12,24}×{200,500,2000}.
- **Main experiment**: not-supported — Original: Macro A=0.053 (BELOW 1/6 chance); B=0.669; C=0.357. A>B: 0/6, A>C: 0/6. Qwen swap: A=0.076, B=0.969, C=0.125. After iter-1 fix: Arm A macro 0.139 (2.6× improvement, still catastrophically below B=0.663).
- **Verify**: robustness=1.0 — method n/a / dataset n/a / model pass; integrity=PASS; verdict=FAIL (robust across Qwen model-swap)
- **Iteration**: falsified as originally stated; rewritten to C3_v2 (narrower scope). Iter-1 (②) α+components fix; iter-2 (③) claim-stage re-entry lightweight in-loop.
- **Final**: ✗ falsified — Arm A > Arm B: 0/6 both original (macro 0.053) and after iter-1 fix (macro 0.139); rewritten to C3_v2 at iteration 2
- **Caveats**: Arm A macro below random chance (original) and still below prompting after iter-1 fix — additive-injection at any tested α cannot beat prompting for open-generation control; superseded by C3_v2
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#block-3, refine-logs/EXPERIMENT_PLAN.md#block-4, refine-logs/EXPERIMENT_RESULTS.md#m3, runs/A3_applied/, runs/A4_verify_qwen/, runs/iteration_round_1/A3_applied_fixed/, verify/C3_applied_circuit_control_beats_prompting_steering/, review-stage/AUTO_ITERATION_FINAL_REPORT.md#21

---
## C3_v2 — Prefix modulation exists + open-generation control is a negative (NEW claim from ③)
- **Statement**: Located emotion-relevant components C_e are associated with reliable prefix-level target-emotion score increases under additive activation injection (positive enhancement Δ_target on all 6 emotions at α ∈ {0.05,0.1,0.3,0.5,1.0}), with limited/partial cross-emotion specificity (the targeted-C_{e'} predicate passes for 4/6 emotions). However, this prefix-level modulation does not translate into effective open-generation control: additive-injection steering is catastrophically worse than prompting (macro 0.139 vs 0.663) and substantially worse than CAA/RepE (0.354). We therefore treat this as a negative result for the practical generation-time control application of C_e.
- **Origin**: Produced by iteration 2 (③ lightweight in-loop claim-stage re-entry) as the narrowed defensible scope for C3.
- **Data**: Reuses SEV eval-fold data + M4 Qwen swap + iteration_round_1 fixed re-run — no new experiments needed for the rewrite; empirical support pre-exists.
- **Models**: Llama-3.2-3B-Instruct, Qwen2.5-7B-Instruct
- **Method**: Reuses (a) M2 enhancement Δ measurements (positive part) + (b) M3/M4 3-arm judge-scored eval accuracy (negative part).
- **Main experiment**: supported (at narrowed scope) — Positive: enhancement Δ_target > 0 on all 6 emotions (from M2 iter-1 fixed metrics, α=1.0 range 1.34-5.83 nats); specificity targeted-C_{e'} 4/6. Negative: Arm A macro 0.139 vs Arm B 0.663 vs Arm C 0.354 after iter-1 fix.
- **Verify**: — (not stress-tested; produced by ③ re-entry)
- **Iteration**: supported (at narrowed scope); READY per iteration-5 reviewer. Iter-2 (③): claim created. Iter-3 (⓪): wording tightening. Iter-4 (⓪): manuscript-scaffold verb sweep. Iter-5 (⓪): title updated to falsification-study framing.
- **Final**: ✓ supported at narrowed scope (positive prefix modulation on all 6 emotions + limited specificity 4/6 + explicit negative for open-generation control); not stress-tested
- **Caveats**: New claim from ③ re-entry — has not been through independent Stage 2 stress test; upgrade via `/auto-verify C3_v2 --resume=true --dimensions=model --gpu_id=1,2,3,5,6`; the positive + negative parts must be reported together (positive alone would over-claim, negative alone would under-recognise the located components)
- **Artifacts**: review-stage/AUTO_REVIEW.md, review-stage/AUTO_ITERATION_FINAL_REPORT.md#21, review-stage/AUTO_ITERATION_FINAL_REPORT.md#4b3, runs/A3_applied/, runs/A4_verify_qwen/, runs/iteration_round_1/A3_applied_fixed/, runs/A2_causal/, runs/iteration_round_1/A2_causal_fixed/

---
## Journey Summary
- **Claim**: given (3 claims C1/C2/C3 fixed by task.md); refined into 5-milestone plan (M0.5 data-prep + M1 Location + M2 Causal + M3 Applied + M4 Qwen swap)
- **Mechanism strategy**: Location → Causal Intervention → Tuning & Editing
- **Mechanism routing**: family=Causal Attribution / Ablation, submethod=paired-contrast direction extraction + Stage-A probe/cosine + Stage-B single-component enhancement prefix-logprob gain; ablation=mean-substitute; enhancement=additive activation injection
- **Experiment**: 5 milestones, ~4.3 GPU-hours realized, headline mixed: C1 supported / C2 not-supported (ablation Δ wrong sign — later confirmed by iteration ② as real, not operator artifact) / C3 not-supported (circuit Arm A macro 0.053 below chance)
- **Verify**: 3 claim(s): 0 PASS / 1 FAIL / 0 INCONCLUSIVE / 0 ZEV / 2 INTEGRITY_ONLY (cap=2, swap_off=0); integrity[Phase2 WARN/Phase9 PASS]; Stage 2 pick = C3 (robustness 1.00 via Qwen M4 model-swap, FAIL confirmed)
- **Iteration**: 2/6 iterations, claim-reentries=1/2, score 6/10 verdict ready, termination=positive_verdict; iter-1 ② fix confirmed C2 negative is REAL (not operator bug) + moved C3 Arm A macro 0.053→0.139 (still below prompting 0.663); iter-2 ③ narrowed C3 → C3_v2 (positive prefix modulation + explicit negative for open-generation control); iter 3-5 ⓪ narrative-only reframed manuscript as falsification-study
- **Figures**: disabled — LEDGER_FIGURES judgment-skipped for all 4 claims (verdict evidence already summarised in per-claim key_stats prose)

## Open Items
- C1 [INTEGRITY_ONLY, stage2_skip_reason: max_verify_claims_cap] — swap-test deferred; upgrade via `/auto-verify C1 --resume=true --dimensions=model --gpu_id=1,2,3,5,6` (Phase 2 audits reused; only Stages 2-3 run). Two Phase-2 WARNs (flat kstar grid + head Stage-B degenerate at layer level) already documented in caveats.
- C2 [INTEGRITY_ONLY, stage2_skip_reason: max_verify_claims_cap; post-fix integrity now PASS] — swap-test deferred; upgrade via `/auto-verify C2 --resume=true --dimensions=model --gpu_id=1,2,3,5,6` (Phase 2 audits reused).
- C3_v2 [new claim from ③ rewrite; not stress-tested] — swap-test deferred; upgrade via `/auto-verify C3_v2 --resume=true --dimensions=model --gpu_id=1,2,3,5,6`. Needs its own Stage 1 audit + Stage 2 stress test to formally confirm the narrower scope.
- Recurring unresolved patterns: (a) val-eval metric misalignment for C3-family — target-prefix logprob unreliable as proxy for judge-scored generation accuracy; (b) why 3/6 emotions have C_e Δ below random-null mean under M2 (mechanistic explanation not pursued in loop).
- Beyond-loop-scope items (would move reviewer score from 6 to 7-8): stronger null-model methodology (layer/size/rank-matched perturbations, multiple ablation operators), sign-sensitive or projection-based causal interventions, multi-model transfer study, theoretical account of scoring-time vs trajectory-time intervention decoupling.
- Multi-GPU device_map bug observed in iteration-1 M2/M3 re-runs (both re-runs fell back to single-GPU); should be traced in a future round.
