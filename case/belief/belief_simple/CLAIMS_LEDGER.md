# Claim Ledger — Belief Localization in Pretrained Language Models (Pythia)

**Direction**: Reproduction of belief-circuit localization in Pythia via Fisher-information heads, checkpoint-window analysis, and probe-and-amplify controller — per task.md (`behavior_source=given`, `mechanism=given`, `resource_fidelity=strict`).
**Date**: 2026-07-22 → 2026-07-22
**Pipeline**: completed | **Iteration**: 8.8/10 "almost" (5/6 rounds, 0/6 back-edge budget consumed, termination=stalled)
**Models**: claim=claude-opus-4-7, experiment=claude-opus-4-7, verify=claude-sonnet-4-6, iteration=claude-opus-4-7
**Updated after**: iteration:final

| Claim | Main experiment | Verify | Post-Iteration | Final |
|-------|-----------------|--------|----------------|-------|
| C1 Scale-Dependent Emergence | ✅ SUPPORTED — 39pp dissociation at 410m | ⚪ INTEGRITY_ONLY (cap) | ⓪ narrative — "3-scales-only, not a scaling law" | SUPPORTED-across-3-Pythia-scales (awaiting swap-test) |
| C2 Belief Heads Localization | ✅ SUPPORTED — 5/5 LOCALIZED, \|H*\|∈{1,2,4} | ❌ FAIL — robustness 0.00 (OLMo-1B: 0/2 localized) | ⓪ narrative — Pythia-family boundary caveat | SUPPORTED-within-Pythia (verify FAIL is out-of-scope cross-family probe) |
| C3 Formation Window | ✅ SUPPORTED — personal[13k,13k] vs attributed[0,33k] | ⚪ INTEGRITY_ONLY (cap) | ⓪ narrative — interval-censored coarse bounds | SUPPORTED-coarse-window (awaiting swap-test) |
| C4 Dynamic Controllability | ✅ SUPPORTED — net_impr +151 (1b) / +27 (2.8b), beats prompt-hint | ⚪ INTEGRITY_ONLY (cap) | ⓪ narrative — PPL disclosure + scale-nonuniform | SUPPORTED-on-tested-Pythia-scales (awaiting swap-test) |

> **Top-level bookkeeping note (iteration ⓪ i2):** Automated verify marks C2 as FAIL because the cross-family model-swap robustness probe (OLMo-1B) failed. This does NOT contradict the paper claim, which is intentionally restricted to the Pythia family — all 5/5 admissible (Pythia model, target) pairs LOCALIZED under all four verbatim criteria. The mechanical FAIL label reflects a stress-test *beyond* the claim's scope, not a defect in the claim itself. See C2 row's Iteration + Final fields and `refine-logs/FINAL_PROPOSAL.md` §C2 Boundary caveat for the scientific interpretation.

---
## C1 — Scale-Dependent Emergence
- **Statement**: Personal belief and attributed belief exhibit distinct emergence patterns across pythia-{410m, 1b, 2.8b}; the two belief-ability accuracy curves differ in slope, in the presence/absence of monotonic scaling, or in where they cross (chance vs above-chance).
- **Origin**: task.md — Claim 1 (given)
- **Data**: `belief_core/{reality,believe_truth,follow_belief}.jsonl` — provenance=existing; available=227+681+681=1589, used=227+681+681=1589 (FULL, per task.md)
- **Models**: pythia-410m, pythia-1b, pythia-2.8b
- **Method**: M1 — log-prob-comparison behavioral evaluation (gold > distractor); Wilson 95% CI per (model, task) cell; above-chance gate = `acc>0.5 AND wilson_ci_low>0.5` — screen → decode → verify (no mechanism intervention; behavioural-only)
- **Main experiment**: SUPPORTED — pythia-410m: WK 0.881 / personal 0.852 / attributed 0.457 (below chance, CI [0.420, 0.494]); pythia-1b: 0.925 / 0.786 / 0.833; pythia-2.8b: 0.960 / 0.994 / 0.796. Above-chance gate cleared in 5 of 6 belief cells (pythia-410m×attributed fails). Personal-attributed gap: 410m=+39pp (personal above, attributed below chance); 1b=−4pp; 2.8b=+20pp. Personal curve is non-monotonic 410m→1b→2.8b (0.85→0.79→0.99).
- **Verify**: INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap) — baseline integrity PASS. Swap-test deferred; upgrade: `/auto-verify C1 -- resume: true`
- **Iteration**: i1 ⓪ narrative-only — added "3-scales-only, not a scaling-law" caveat to FINAL_PROPOSAL.md §C1. INTEGRITY_ONLY carried forward as an Open Item (no back-edge action; upgrade command dispatched).
- **Final**: SUPPORTED-across-3-Pythia-scales (differential behavior observed; not a scaling law) — awaiting swap-test
- **Caveats**: Only 3 model sizes observed. "Scale-dependent emergence" should be read as *differential behavior across the available Pythia scales* rather than as a scaling-law characterization; the personal-belief non-monotonicity (0.85 → 0.79 → 0.99) precludes any smooth monotonic reading.
- **Artifacts**: `refine-logs/artifacts/behavioral/{pythia-410m,pythia-1b,pythia-2.8b}/{world_knowledge,personal_belief,attributed_belief}.json`, `refine-logs/artifacts/behavioral/above_chance_gate.json`
- **Figures**:
  - ![Behavioral scaling of world_knowledge / personal_belief / attributed_belief across pythia-{410m, 1b, 2.8b} — Wilson 95% CI. The smallest model handles personal belief above chance (0.85) but fails attributed belief entirely (0.46, below chance). Personal-belief accuracy is non-monotonic across scales.](figures/C1/c1_scale_matrix.png) — vector: `figures/C1/c1_scale_matrix.pdf`

---
## C2 — Belief Heads Localization
- **Statement**: For pythia models that behaviorally clear the above-chance gate, Fisher-information masks (top-0.1% target AND-NOT top-1% F_knowledge) identify a smallest attention-head set H* whose zero-ablation satisfies all four causal-localization criteria (target-acc drop ≥0.30 AND >mean+2σ of 20 random-head controls; off-target belief AND world-knowledge drop ≤0.10; post-ablation PPL ≤1.05× clean), separately for personal_belief and attributed_belief.
- **Origin**: task.md — Claim 2 (given)
- **Data**: belief_core (person∈{james,mary} for Fisher; full for eval) + Pile PPL sample (1,048,576 tokens) — provenance=existing; available=F_personal/F_attributed 454 each (James+Mary subset), F_knowledge 227 full, eval 227+681+681; used=F_personal=454, F_attributed=454, F_knowledge=227, eval on FULL datasets, PPL on 1,048,576 tokens (subset: Fisher signals restricted to person∈{james,mary} per task.md; evaluation uses full datasets)
- **Models**: pythia-410m, pythia-1b, pythia-2.8b
- **Method**: M2 — Fisher-information matrix + zero-ablation: (M2.1) empirical Fisher per signal, fp32 accumulation, per-head aggregation over fused QKV + dense; (M2.2) top-0.1% AND-NOT top-1% masks + per-head candidate scoring + jackknife stability (ρ≥0.6 flag); (M2.3) deterministic greedy-add + greedy-remove smallest-H* search, |S|≤30; (M2.4) 20 random-head controls (seeds 100..119) + 20 random-mask controls (seeds 200..219) + 4-criteria acceptance — screen (Fisher) → decode (mask+rank) → verify (zero-ablation) → recover (controls)
- **Main experiment**: SUPPORTED — 5 of 5 admissible (model, target) pairs LOCALIZED under all four verbatim criteria; |H*| ∈ {1, 2, 4}, median=2, smallest is a single head (pythia-2.8b × attributed: L5H22; pythia-1b × attributed: L4H1). Jackknife Spearman ρ ∈ [0.885, 0.992] all above 0.6. Details: pythia-410m×personal |H*|=2 (L13H1,L8H8) Δ_target=0.372 PPL=1.010×; pythia-1b×personal |H*|=2 (L12H1,L9H1) Δ=0.471 PPL=1.019×; pythia-1b×attributed |H*|=1 (L4H1) Δ=0.463 PPL=1.011×; pythia-2.8b×personal |H*|=4 Δ=0.307 PPL=1.007×; pythia-2.8b×attributed |H*|=1 (L5H22) Δ=0.355 PPL=1.003×. Negative off-belief drops (e.g. −0.31 on pythia-410m×personal) hint personal- and attributed-belief circuits are opposing in some scales.
- **Verify**: FAIL — model-swap-olmo-1b: 0/2 localized (personal_belief: C2c/C2d both fail; attributed_belief: C2a never met — drops are negative, heads act as suppressors). robustness=0.00 (0/1 eligible). integrity_status=warn (cross-vocab PPL). Full trace: `verify/C2_belief_heads_localization/ROBUSTNESS.md`
- **Iteration**: i1 ⓪ narrative-only — added Pythia-family boundary caveat to FINAL_PROPOSAL.md Must-Prove Claims §C2 + this ledger row. Reviewer score 8/10 verdict `almost`. C2 verify-bookkeeping remains FAIL because (a) the OLMo-1B variant cannot be un-run, (b) task.md hard constraint restricts on-disk Pythia weights to {410m, 1b, 2.8b} — no in-family neighbor available for a same-family model-swap without an explicit download of pythia-160m or pythia-6.9b (blocked). The OLMo-1B cross-family swap is definitionally out-of-scope for a Pythia-scoped claim; verify's mechanical robustness=0.00 must be interpreted with this scope-mismatch context, not as evidence against the claim's literal wording.
- **Final**: SUPPORTED-within-Pythia (fragile under cross-family swap; interpretation: Pythia-family localization result, not universal semantic head identity)
- **Caveats**:
  - pythia-410m×attributed excluded from M2.3 per R1 protocol (M1 above-chance gate failed at 45.7%, CI upper 0.494).
  - **Cross-family fragility (verify-derived, iteration i1)**: on OLMo-1B (16L/16H, 2048D) the same Fisher-mask+zero-ablation pipeline produces heads that either entangle with general LM computation (personal: PPL up to 46× clean) or act as **suppressors** (attributed: all 30 greedy steps show *negative* drops, inverted causal direction). Jackknife ρ=0.954 on OLMo-1B — Fisher signal itself stable, so the failure is architectural, not statistical.
  - **Terminology**: "belief heads" is an operational designator for the Pythia-family Fisher-selected set, not a universal semantic head identity.
- **Artifacts**: `refine-logs/artifacts/fisher/{model}/{signal}.pt`, `refine-logs/artifacts/masks/{model}/{Mask,RankedHeads,Jackknife}_{target}.json`, `refine-logs/artifacts/hstar/{model}/H_{target}{,_acceptance}.json`, `refine-logs/artifacts/ablation/{model}/{target}/{main,random_head,random_mask}/*.json`
- **Figures**:
  - #### M2 final acceptance for the 5 admissible (model, target) pairs. Every pair LOCALIZED under all four verbatim criteria; \|H*\| ∈ {1, 2, 4}; the attributed circuit is often a single head (pythia-2.8b × attributed: L5H22; pythia-1b × attributed: L4H1).

    | Model | Target | \|H*\| | Heads (layer, head) | Δ target | Δ off-belief | Δ WK | PPL ratio | mean_rh + 2σ | C2a | C2b | C2c | C2d | Status |
    |---|---|---|---|---|---|---|---|---|---|---|---|---|---|
    | pythia-410m | personal | 2 | (13,1), (8,8) | 0.372 | -0.314 | +0.000 | 1.010× | 0.081 | PASS | PASS | PASS | PASS | **LOCALIZED** |
    | pythia-1b | personal | 2 | (12,1), (9,1) | 0.471 | -0.123 | +0.004 | 1.019× | 0.151 | PASS | PASS | PASS | PASS | **LOCALIZED** |
    | pythia-1b | attributed | 1 | (4,1) | 0.463 | +0.090 | +0.004 | 1.011× | 0.257 | PASS | PASS | PASS | PASS | **LOCALIZED** |
    | pythia-2.8b | personal | 4 | (14,16), (15,3), (13,1), (12,4) | 0.307 | -0.150 | +0.004 | 1.007× | 0.035 | PASS | PASS | PASS | PASS | **LOCALIZED** |
    | pythia-2.8b | attributed | 1 | (5,22) | 0.355 | +0.001 | +0.004 | 1.003× | 0.033 | PASS | PASS | PASS | PASS | **LOCALIZED** |

    Source `.tex`: `figures/C2/c2_localization_table.tex`
  - ![Smallest localized attention-head-set size |H*| for each (Pythia model, belief target) — median 2, minimum 1. Sparsity of the belief circuit is robust across scale.](figures/C2/c2_hstar_size_bar.png) — vector: `figures/C2/c2_hstar_size_bar.pdf`

---
## C3 — Formation Window
- **Statement**: Personal belief and attributed belief exhibit distinct formation windows during pythia-1b pretraining, as measured by both a behavioral-emergence step (first checkpoint with acc≥0.60 that persists 2-of-next-3) and a causal-emergence step (first checkpoint with Δ:=acc(target)−acc_ablated(H*_target,target)≥0.20 that persists 2-of-next-3).
- **Origin**: task.md — Claim 3 (given)
- **Data**: pythia-1b intermediate checkpoints (native schedule; 24 of 154 planned checkpoints stored on disk) + belief_core datasets — provenance=existing; available=24 checkpoints × 1589 examples × 9 measurements/checkpoint; used=all 24 available checkpoints (step0..step143000 log-spaced); FULL datasets per task-eval; 9 measurements per checkpoint (subset: 154 checkpoints planned, only 24 native log-spaced ones on disk — environmental constraint, not a cost-saving downscale; strict-harness HALT rule inapplicable)
- **Models**: pythia-1b (intermediate checkpoints)
- **Method**: M3 — checkpoint-analysis with zero-ablation: at each of 24 available pythia-1b checkpoints run 9 measurements = 3 behavioral × {no-intervention, ablate H*_personal (L12H1+L9H1), ablate H*_attributed (L4H1)}. Post-hoc emergence detection with pre-registered thresholds (behavioral acc≥0.60 with 2/3 persistence; causal Δ≥0.20 with 2/3 persistence). Distinctness = non-identical windows.
- **Main experiment**: SUPPORTED — Distinct formation windows: personal [step 13000, step 13000] vs attributed [step 0, step 33000]. Distinct=TRUE. Notable early-training reorganization at step 2000: personal=0.043 (dips near zero) while attributed=0.971 (spike near ceiling). Attributed-belief behaviour is above 60% at initialization (step 0: acc 0.670) — likely a prompt-structure bias; its causal circuit only emerges at step 33000.
- **Verify**: INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap) — baseline integrity WARN (scope: 24/154 checkpoints). Swap-test deferred; upgrade: `/auto-verify C3 -- resume: true`
- **Iteration**: i1 ⓪ narrative-only — sharpened "interval-censored coarse bounds" wording in FINAL_PROPOSAL.md §C3 (personal window [13000, 13000] reflects sampling granularity between recorded steps 8000 and 23000, not a sharp transition; attributed early-window at step 0 reflects operational-definition + prompt-structure bias, not a causal-circuit claim at step 0 — its causal emergence is step 33000). INTEGRITY_ONLY carried forward as an Open Item.
- **Final**: SUPPORTED-coarse-window (distinct behavioral+causal windows on the observed native-log-spaced checkpoint subset) — awaiting swap-test
- **Caveats**:
  - Coarse-log-spaced resolution — 24 of 154 planned pythia-1b intermediate checkpoints on disk (environmental gap, not downscale). Emergence-step localization is at native-log-schedule resolution rather than every-1000-steps.
  - Personal window [13000, 13000] is a *singleton* reflecting the adjacent recorded checkpoints (step 8000 next-below, step 23000 next-above); do NOT interpret as a sharp behavioral transition at exactly step 13000.
  - Attributed early behavioral window (from step 0, acc 0.670) reflects the operational definition (acc ≥ 0.60 + 2/3 persistence) applied to a prompt structure that biases toward follow-belief continuations at initialization; the causal circuit for attributed_belief only emerges at step 33000 (17k steps after personal's causal emergence at 13000). The two facts are consistent, not contradictory.
- **Artifacts**: `refine-logs/artifacts/formation/step{0..143000}.json (×24)`, `refine-logs/artifacts/formation/summary.{json,md}`
- **Figures**:
  - ![Formation windows for personal vs attributed belief on pythia-1b. Attributed behaviour is above 60% at initialization (prompt-structure bias) but its causal circuit only emerges at step 33000; personal-belief behaviour and its causal circuit both crystallize together at step 13000. Distinct=TRUE. Coarse-log-spaced resolution reflects the 24 checkpoints available on disk (of 154 planned).](figures/C3/c3_formation_trajectories.png) — vector: `figures/C3/c3_formation_trajectories.pdf`

---
## C4 — Dynamic Controllability
- **Statement**: A lightweight probe-and-amplify controller — a frame-classifier (MLP on 3 pre-belief-head layers) that infers the current frame at inference time and amplifies the matched Claim-2 belief-head set with a tuned magnitude — yields net-positive OOD improvement on belief_holdout for the two belief tasks without destroying world-knowledge accuracy or general LM ability (PPL), and its improvement is comparable to (or better than) an explicit prompt-hint baseline.
- **Origin**: task.md — Claim 4 (given)
- **Data**: `belief_core/` (training, 227+681+681=1589, 80/20 stratified split seed=0) + `belief_holdout/` (OOD eval: reality n=367, believe_truth n=1101, follow_belief n=1101) + Pile PPL sample (1,048,576 tokens) — provenance=existing; available=belief_core=1589, belief_holdout=367+1101+1101=2569; used=belief_core=1271 train / 318 val, belief_holdout=FULL, PPL=1,048,576 tokens (subset: controller trained on belief_core train split; α tuned on belief_core val split; ALL evaluation is on belief_holdout OOD)
- **Models**: pythia-1b (H*_personal + H*_attributed both LOCALIZED in M2.4), pythia-2.8b (both LOCALIZED)
- **Method**: M4 — probe-and-amplify controller: (M4.1) train MLP frame classifier hidden=256 dropout=0.1 on concatenated hidden states from 3 layers before earliest-H*-head layer (pythia-1b L_ctrl=4 probing layers [1,2,3]; pythia-2.8b L_ctrl=5 probing layers [2,3,4]); (M4.2) grid-search α_personal × α_attributed ∈ {1.0,1.5,2.0,3.0,4.0,6.0}² on belief_core val, select by max net_improvement s.t. Δ_wk≤0.05 strict guardrail; (M4.3) OOD evaluation on belief_holdout — 3 arms (baseline_no_control, controller, prompt_hint_baseline) + PPL preservation check — screen (probe train) → decode (α grid) → verify (OOD 3-arm) → recover (WK preservation + PPL)
- **Main experiment**: SUPPORTED — Frame classifier OOD acc: pythia-1b 0.9977 / pythia-2.8b 0.9957. pythia-1b (α_p*=3.0, α_a*=1.5): OOD personal_belief 0.672→0.850 (+17.8pp), attributed_belief 0.788→0.909 (+12.1pp), WK exactly preserved 0.850→0.850, controller recovered=165 degraded=14 net_impr=+151, prompt-hint net_impr=−1, PPL 1.047×. pythia-2.8b (α_p*=1.5, α_a*=4.0): OOD personal 0.948→0.973 (+2.5pp), attributed 0.825→0.880 (+5.5pp), WK exactly preserved 0.886→0.886, controller net_impr=+27, prompt-hint net_impr=−87 (actively harmful on larger model), PPL 1.005×.
- **Verify**: INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap) — baseline integrity PASS. Swap-test deferred; upgrade: `/auto-verify C4 -- resume: true`
- **Iteration**: i1 ⓪ narrative-only — clarified in FINAL_PROPOSAL.md §C4 that "WK exactly preserved" is task-level (belief_holdout WK subset) and that general LM ability is tracked separately via Pile PPL: 1.047× on pythia-1b and 1.005× on pythia-2.8b (both under C2 1.05× bar, non-zero). Also flagged: gain magnitude is model-dependent (+151 on 1b vs +27 on 2.8b), so no claim of uniform controllability across scale. INTEGRITY_ONLY carried forward as an Open Item.
- **Final**: SUPPORTED-on-tested-Pythia-scales (net-positive OOD gain on both pythia-1b and pythia-2.8b; strictly beats prompt-hint on both; PPL preserved within the C2 bar; effect size differs by model) — awaiting swap-test
- **Caveats**:
  - pythia-410m excluded from M4 per plan (only H*_personal localized in M2, needs both H* to run controller).
  - Effect size is not uniform: +151 net_impr on pythia-1b vs +27 on pythia-2.8b. Do not present controllability as scale-invariant.
  - PPL delta is small but non-zero (1.047× on 1b, 1.005× on 2.8b); report explicitly rather than folding into "WK exactly preserved" which only covers task-level WK accuracy.
- **Artifacts**: `refine-logs/artifacts/controller/{pythia-1b,pythia-2.8b}/{probe.pt,probe_report.json,probe_split.json,alpha_grid/*.json,alpha_selected.json,ood_baseline_no_control.json,ood_controller.json,ood_prompt_hint.json,M4_report.{json,md}}`
- **Figures**:
  - ![Controller vs baseline vs prompt-hint on belief_holdout OOD (n=367+1101+1101=2569). Controller delivers strictly positive net improvement on both models; the prompt-hint baseline is net-negative on both (actively harmful on pythia-2.8b: −87). World-knowledge accuracy exactly preserved across arms.](figures/C4/c4_ood_by_arm_and_model.png) — vector: `figures/C4/c4_ood_by_arm_and_model.pdf`
  - #### M4.3 OOD summary: controller wins over prompt-hint on both Pythia scales; effect size differs (+151 on 1b vs +27 on 2.8b — see caveat C4.2 about non-uniform gain across scale).

    | Model | α_p* | α_a* | OOD personal (baseline → controller) | OOD attributed (baseline → controller) | OOD WK (baseline → controller) | Controller net_impr | Prompt-hint net_impr | Frame classifier OOD acc | PPL ratio (controller / clean) |
    |---|---|---|---|---|---|---|---|---|---|
    | pythia-1b | 3.0 | 1.5 | 0.672 → 0.850 | 0.788 → 0.909 | 0.850 → 0.850 | **+151** | -1 | 0.9977 | 1.047× |
    | pythia-2.8b | 1.5 | 4.0 | 0.948 → 0.973 | 0.825 → 0.880 | 0.886 → 0.886 | **+27** | -87 | 0.9957 | 1.005× |

    Source `.tex`: `figures/C4/c4_summary_table.tex`

---
## Journey Summary
- **Claim**: 4 given claims (no ideation) → unified plan across M1→M2→M3&M4; `resource_fidelity=strict`, per-claim `chosen_mechanism` map stamped
- **Mechanism strategy**: n/a (`MECHANISM=given` — no `/mechanism-explore` invocation)
- **Mechanism routing**: per-claim map — C1:not-applicable, C2:fisher-information-matrix-zero-ablation, C3:checkpoint-analysis-with-zero-ablation, C4:probe-and-amplify-controller
- **Experiment**: ~148 runs, ~5.5 wall-clock hours (4-6 concurrent GPUs), all 4 claims SUPPORTED with headline: distinct scale-emergence + sparse (|H*|∈{1,2,4}) **Pythia-family** Fisher-localized (operationally-defined) belief heads + distinct formation windows on pythia-1b + probe-and-amplify controller beats prompt-hint baseline
- **Verify**: C2=FAIL (robustness 0.00, OLMo-1B 0/2 localized — cross-family, out-of-scope for Pythia-scoped claim); C1/C3/C4=INTEGRITY_ONLY (max_verify_claims_cap, swap-test deferred). Overall: 0 PASS, 1 FAIL (definitionally out-of-scope), 3 INTEGRITY_ONLY.
- **Iteration**: 5/6 rounds, 0/6 back-edge budget consumed (all ⓪ narrative-only or no-action; claim-reentries=0/2). Score trajectory 8.0 → 8.5 → 8.8 → 8.8 → 8.8, verdict `almost` throughout. i1: added Pythia-family boundary caveat to C2 + scaling-law/coarse-window/PPL caveats to C1/C3/C4. i2: added top-level bookkeeping-vs-scope-support one-liner, synced C2 Final to "SUPPORTED-within-Pythia", scope-qualified all 4 summary-table verdicts. i3–i5: no-action reviewer polls confirming stability; consecutive_noop_count=2 → stall guard fired at i5. Reviewer's iteration-3 declaration ("this is the natural terminal state ... further looping is likely to be churn") aligns with the stall termination. Positive verdict never fired only because verify's C2 remains mechanically in `verify_failed` (out-of-scope cross-family probe) and no in-family Pythia neighbor is on disk to swap-test.
- **Figures**: 8 across 4 claims (6 image figures + 2 tables); 0 judgment-skipped; 0 render-skipped, 0 errored. Global index: `figures/INDEX.md`.

## Open Items
- C3 environmental gap: only 24 of 154 planned pythia-1b intermediate checkpoints stored on disk (native log-spaced subset). Ran M3 on the 24 available — resolution is coarse-log-spaced rather than every-1000-steps. Not a downscale (missing files, not cost-saving), so strict-harness HALT rule inapplicable. Documented in `MECHANISM_ROUTING.md → ## Plan reconciliation` + `EXPERIMENT_RESULTS.md`.
- Cross-model code review substitution: llm-chat MCP unavailable during experiment stage — Phase 3 replaced with self-review + a hook-mechanism sanity test (`scripts/_test_hooks.py`: α=0 knockout non-trivial, α=1 identity exactly zero, α>1 amplifies) + log-prob-metric round-trip test (`scripts/_test_logprob.py`). One bug caught during self-review: batched `continuation_logprob` shared `past_key_values` across continuations, which `transformers 4.57` stateful `Cache` mutates in-place — fixed by dropping the KV-cache optimization.
- C1 verify: INTEGRITY_ONLY, `stage2_skip_reason=max_verify_claims_cap`. Swap-test deferred; upgrade with `/auto-verify C1 -- resume: true`.
- C3 verify: INTEGRITY_ONLY, `stage2_skip_reason=max_verify_claims_cap`. Also Phase 2 integrity WARN (scope: 24/154 checkpoints). Swap-test deferred; upgrade with `/auto-verify C3 -- resume: true`.
- C4 verify: INTEGRITY_ONLY, `stage2_skip_reason=max_verify_claims_cap`. Swap-test deferred; upgrade with `/auto-verify C4 -- resume: true`.
- C2 cross-family fragility (verify FAIL): on OLMo-1B (16L/16H, 2048D) the same Fisher-mask+zero-ablation pipeline produces heads that either entangle with general LM computation (personal: PPL up to 46× clean) or act as **suppressors** (attributed: all 30 greedy steps show negative drops — inverted causal direction). Jackknife ρ=0.954 on OLMo-1B confirms Fisher signal itself is stable → failure is architectural, not statistical. Optional follow-up: download pythia-160m or pythia-6.9b to distinguish cross-family from cross-scale fragility (blocked by task.md hard-constraint allowlist unless explicitly authorized).
