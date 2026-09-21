# Auto Iteration Final Report — Verifying Locatability, Causality, and Applied Control of Emotion Circuits in Llama-3.2-3B

- **Generated**: 2026-07-14T01:05:00
- **Iterations consumed**: 2 / 6
- **Claim-reentries consumed**: 1 / 2
- **Final reviewer score**: 6 / 10
- **Final canonical verdict**: ready
- **Termination reason**: positive_verdict (three-dimensional STOP rule satisfied: score >= 6 AND verdict ∈ {ready, almost} AND no claim FAIL/INCONCLUSIVE/ZERO_ELIGIBLE_VARIANTS)
- **Cumulative cost**: runs_total = 2, gpu_hours_total ≈ 0.6 (in addition to the 4.3h original main experiment)
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md)

---

## Executive Summary

The loop entered with C3 in FAIL (main-experiment Arm A macro accuracy 0.053 below the 1/6 chance floor, robust under Qwen model-swap), C1 and C2 in INTEGRITY_ONLY (cap-cut before Stage 2), and a reviewer-flagged HIGH WARN on C2's ablation operator (global mean where the plan specified per-stem mean over 5 other emotions). Iteration 1 dispatched a Type ② main-experiment-script fix for both C2 (corrected per-stem operator + enlarged random-null pool from 14 heads to 72 heads / 4915 neurons) and C3 (α reduced 10x from {0.5,1.0,2.0} to {0.05,0.1,0.3}, component neighborhood shifted downward). The C2 fix demonstrated the wrong-sign ablation was REAL, not artifactual (per-stem Δ +0.48 to +0.93 nats, essentially unchanged); the C3 fix improved Arm A macro 2.6x (0.053→0.139) but still catastrophically lost to Arm B prompting (0.663) on all 6 emotions. Iteration 2's reviewer accepted this as scientific evidence that additive-injection generation control cannot beat prompting at any reasonable α, and recommended a Type ③ claim rewrite — which was executed as a lightweight in-loop rewrite producing C3_v2 (narrowed scope: prefix-level score modulation + limited specificity + explicit negative for open-generation control). Iterations 3-5 applied three ⓪ narrative refinements (C3_v2 wording tightening, C2 two-way interpretation of positive-Δ anomaly, C1 coarse-locatability scope-control, falsification-study title) that took the reviewer's score from 3/10 → 4 → 5 → 5.5 → 6/10 and verdict from "not ready" → "not ready" → "almost" → "almost" → "READY". The paper is submission-ready under the reviewer's explicit condition: as a falsification / stress-test / negative-results paper, not as a triumphant circuit-discovery paper.

### Claim Disposition Overview

| Original state | # Claims | Final status after iteration |
|---|---|---|
| PASS | 0 | — |
| FAIL | 1 (C3) | 1 rewritten via ③ → C3_v2 supported at narrowed scope (positive prefix modulation + negative for generation control) |
| INCONCLUSIVE | 0 | — |
| ZERO_ELIGIBLE_VARIANTS | 0 | — |
| INTEGRITY_ONLY | 2 (C1, C2) | 1 (C1) held with ⓪ scope-control caveats; 1 (C2) confirmed as real negative via ② operator fix (not artifactual) with ⓪ reframing |
| Legacy DEFERRED | 0 | — |

---

## Section 1 — PASS Claims (brief audit)

*None. Stage 2 verify only picked C3 (which came back FAIL); C1 and C2 were INTEGRITY_ONLY, so no PASS claim entered the loop.*

---

## Section 2 — FAIL Claims (full journey)

### 2.1 `C3` — Applied Circuit Control (rewritten to `C3_v2`)

**Original FAIL signal** (from `verify/C3_applied_circuit_control_beats_prompting_steering/ROBUSTNESS.md`):
- Main-experiment verdict: not-supported. Arm A macro accuracy 0.053 (below 1/6 = 0.167 chance floor), Arm B 0.669, Arm C 0.357. A>B: 0/6, A>C: 0/6.
- Qwen model-swap variant (from M4): Arm A 0.076 (also below chance), B 0.969, C 0.125 — same failure pattern on different architecture.
- Robustness = 1.00 (variant agrees with main experiment); FAIL verdict is ROBUST.
- Variant integrity at entry: PASS (Qwen variant Phase 9 audit clean).

**Iteration journey**

| Iter | Type | Reviewer flag | Action | Outcome |
|---|---|---|---|---|
| 1 | ② | cumulative additive injection over ~24 heads + ~2000 neurons at large α pushes model OOD; val metric (logprob gain) misaligned with judge-scored generation accuracy | Env-driven α range reduction {0.5,1.0,2.0}→{0.05,0.1,0.3}; component neighborhood {12,24,48}×{1000,2000,4000}→{5,12,24}×{200,500,2000}; re-ran M3 at full plan scale into `runs/iteration_round_1/A3_applied_fixed/` | Arm A macro 0.053→0.139 (2.6x improvement); still A>B: 0/6, A>C: 0/6. Selected configs all landed at α=0.3 (upper end of reduced range), suggesting we're on the "too small" side of the tradeoff, not the "OOD" side. Reviewer's α-magnitude hypothesis partially confirmed but ceiling of additive-injection Arm A is far below prompting. |
| 2 | ③ | additive-injection generation control cannot beat prompting at any reasonable α — failure is structural, not tuning; needs claim rewrite | Lightweight in-loop rewrite: created new claim `C3_v2` with narrowed scope (prefix-level score modulation exists; generation control does not). No new experiments (existing data supports the new claim in both its positive and negative parts). Claim-reentry sub-budget consumed: 1/2. | Reviewer accepted C3_v2 as "substantively defensible" — first C3 version to earn that language. Then in iterations 3-5, further ⓪ wording refinements adopted per reviewer feedback ("associated with reliable prefix-level target-emotion score increases under additive activation injection"; "limited/partial cross-emotion specificity"). |

**Path taken (summary)**: main-experiment-script fix (Type ②) at reduced α + reduced components → confirmed structural failure of additive-injection generation control → claim-stage re-entry (Type ③ lightweight in-loop) → narrower defensible claim scope. Claim-reentry sub-budget used: 1.

**Experiment & script modifications** (cumulative across all iterations on this claim)

| Iter | Component | Before | After |
|---|---|---|---|
| 1 | `experiments/m3_applied.py:75-76` (constants) | `ALPHAS_ARM_A = [0.5, 1.0, 2.0]` hard-coded | env-driven `_parse_env_alphas("ALPHAS_ARM_A", [0.5, 1.0, 2.0])` — with `ALPHAS_ARM_A=0.05,0.1,0.3` set in the run's env |
| 1 | `experiments/m3_applied.py:326-341` (K_H_neigh / K_N_neigh derivation) | `K_H_neigh = sorted({max(K_H_GRID[0], kh // 2), kh, min(K_H_GRID[-1], kh * 2)})[:3]` → {12, 24, 48} for kstar=24 | env override honored: `K_H_ARM_A=5,12,24` → K_H_neigh = {5, 12, 24}; similarly K_N_ARM_A=200,500,2000 |
| 1 | `runs/iteration_round_1/A3_applied_fixed/run.sh` | (new) | `ALPHAS_ARM_A=0.05,0.1,0.3 K_H_ARM_A=5,12,24 K_N_ARM_A=200,500,2000 CUDA_VISIBLE_DEVICES=3 python experiments/m3_applied.py --m1_dir runs/A1_location --out_dir runs/iteration_round_1/A3_applied_fixed --judge_workers 6` |

**Claim modifications** (mandatory subsection)
- Original claim id `C3`: "Intervening on C_e via additive-injection enhancement (Arm A) at test time produces target-emotion generation that beats prompting (Arm B) and single-direction CAA/RepE steering (Arm C) on ≥ 5/6 emotions."
- After iteration 2 → new claim id `C3_v2`: "Located emotion-relevant components C_e are associated with reliable prefix-level target-emotion score increases under additive activation injection (positive enhancement Δ_target on all 6 emotions at α ∈ {0.05,0.1,0.3,0.5,1.0}), with limited/partial cross-emotion specificity (the targeted-C_{e'} predicate passes for 4/6 emotions). However, this prefix-level modulation does not translate into effective open-generation control: additive-injection steering is catastrophically worse than prompting (macro 0.139 vs 0.663) and substantially worse than CAA/RepE (0.354). We therefore treat this as a negative result for the practical generation-time control application of C_e."
- Scope change: from asserting Arm A > prompting on ≥5/6 emotions (falsified by data) to asserting positive prefix-level score modulation + limited specificity + explicit negative for open-generation control.

**Final experiment summary**
- New runs cited: `runs/iteration_round_1/A3_applied_fixed/{arm_A_val,arm_B_val,arm_C_val,selected_configs,eval_generations,judge_results,metrics}.json`
- Final Arm A macro accuracy: **0.139** (vs 0.053 with original α; vs 0.663 for Arm B prompting; vs 0.354 for Arm C CAA/RepE)
- Final per-emotion Arm A accuracy: joy 0.300, fear 0.317, sadness 0.075, anger 0.083, surprise 0.058, disgust 0.000
- Final status: **PASS (under rewritten claim C3_v2)**

**Reviewer memory thread** (cross-iteration suspicions filtered to this claim's pattern)
- "C3 likely reflects fundamental operator/eval mismatch, not tuning noise" (iteration 1) — RESOLVED as correct hypothesis; even reduced α couldn't recover Arm A.
- "C3 failure looks structural, not just overdosing" (iteration 2) — CONFIRMED by empirical outcome.
- "score modulation can still be misread as meaningful steering; keep specifying prefix-level score metric and sharply separate from open-generation behavior" (iteration 3) — ADDRESSED via iteration-3 wording tightening.

---

## Section 3 — INCONCLUSIVE Claims (main-experiment-fix journey)

*None. No original claim entered the loop in INCONCLUSIVE state.*

However, C2 (originally INTEGRITY_ONLY) received a Type ② main-experiment-script fix under a reviewer-approved exception. See Section 4b below.

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims (variant-fix journey)

*None. No claim entered the loop in ZERO_ELIGIBLE_VARIANTS state.*

---

## Section 4b — INTEGRITY_ONLY Claims (Stage 2 skipped by cap; reviewer-approved ② exception for C2)

### 4b.1 `C1` — Localizability per-emotion component sets C_e

**Original state at loop entry**: INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap)
- Main-experiment verdict: supported. Jaccard heads 0.947–1.000 vs null CI-hi 0.265–0.277 (6/6 pass); neurons 0.972–0.983 vs null CI-hi 0.045–0.047 (6/6 pass).
- Baseline integrity: PASS with 2 WARNs (flat kstar grid; head Stage-B layer-level hook).

**Actions taken**: type ⓪ narrative-only refinements only. No script or data changes.
- Iteration 1: recorded reviewer's caveats — flat kstar grid means don't claim (24, 2000) is a uniquely-identified sparsity minimum; head Stage-B layer-level means don't overclaim fine-grained per-head causal ranking.
- Iteration 4: adopted the coarse-locatability scope — Localization here should be read as 'discriminative, emotion-relevant regions/layers with stable component-set structure,' NOT as 'precise identification of the causal subset.'
- Iteration 4: added Discussion subsection to paper plan formalizing "Coarse vs precise localization: why our 'localization succeeds' does not mean 'precise causal identification'".

**Final status**: **supported at coarse-locatability scope** (⓪ caveats only).

**Reviewer memory thread**: reviewer confirmed at iteration 5: "This is the right fix. Your new C1 wording does exactly what I asked for... That is the cleanest formulation you've had across the loop."

**Unverified under swaps**: C1 was cap-cut by MAX_VERIFY_CLAIMS=1 and was not stress-tested. Upgrade instruction: `/auto-verify C1 --resume=true --dimensions=model --gpu_id=1,2,3,5,6`.

### 4b.2 `C2` — Causal + Stability of C_e (Type ② exception)

**Original state at loop entry**: INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap) with baseline integrity WARN (HIGH). The HIGH WARN was on the ablation operator: `experiments/m2_causal.py:341-350` used a global mean across ALL 120 stems × 5 off-target emotions, whereas the plan specified per-stem mean over the 5 other emotions of the SAME stem. The verify baseline audit flagged this as "elevates C2 from FAIL to INCONCLUSIVE under strict integrity standards" — a real operator scope bug that could plausibly explain the wrong-sign ablation Δ.

**Actions taken**: Type ② main-experiment-script fix under a reviewer-approved exception to the INTEGRITY_ONLY routing default (iteration 1); subsequent ⓪ narrative refinements (iterations 3-4).

**Iteration 1 experiment plan & script modifications**

| Iter | Component | Before | After |
|---|---|---|---|
| 1 | `experiments/m1_location.py:99` (new register) | `_HOOK_EVENT_LENS: list[int] = []` (single register) | Added `_HOOK_STEM_IDX: list[int] = []` register for per-stem substitute lookup |
| 1 | `experiments/m1_location.py` (new helper) | (no per-stem publisher) | Added `target_prefix_logprob_batch_with_stem_idx()` that publishes both event lengths AND per-batch-row global stem indices |
| 1 | `experiments/m2_causal.py:129` (ablation hook) | `_install_mean_substitute_ablation_hooks(mean_head_by_layer: dict[(l,h)→(HD,)], mean_neuron_by_layer: dict[(l,n)→float])` — expects a single scalar-per-component substitute (global collapse) | `_install_mean_substitute_ablation_hooks_per_stem(mean_head_by_layer_stem: dict[(l,h)→(n_stems, HD)], mean_neuron_by_layer_stem: dict[(l,n)→(n_stems,)])` — expects per-stem tensors; hook uses `_HOOK_STEM_IDX[i]` to pick each row's substitute value |
| 1 | `experiments/m2_causal.py:355-397` (ablation loop) | `mean_head = {(l,h): mean_head_by_stem[:, l, h].mean(axis=0)}` collapses (n_stems, HD) → (HD,) via `.mean(axis=0)` — the offending global collapse | Passes per-stem tensors `{(l,h): mean_head_by_stem[:, l, h, :]}` of shape (n_stems, HD) directly; then calls `target_prefix_logprob_batch_with_stem_idx(...)` with stem_idx_list = [0..119] so each batch row's substitute is picked correctly |
| 1 | `experiments/m2_causal.py:394-419` (random-null pool) | `n_top_h = round(TOP_HEAD_FRAC=0.20 * NH=24 * TOP_N_LAYERS=3) = 14`; then draw k_h=24 heads → forced k>pool degeneracy | Enlarged: `RN_TOP_LAYERS = min(2*TOP_N_LAYERS, L) = 6`, `RN_HEAD_FRAC = max(TOP_HEAD_FRAC, 0.5) = 0.5`, `RN_NEURON_FRAC = max(TOP_NEURON_FRAC, 0.10) = 0.10` → pool of 72 heads / 4915 neurons per emotion; k=24 has proper draw variance |
| 1 | `runs/iteration_round_1/A2_causal_fixed/run.sh` | (new) | `CUDA_VISIBLE_DEVICES=6 python experiments/m2_causal.py --m1_dir runs/A1_location --out_dir runs/iteration_round_1/A2_causal_fixed --skip_stability` |

**Iteration 1 re-run outcome**

| Iter | Path | New runs | Result |
|---|---|---|---|
| 1 | local re-run of M2 with fixed operator and enlarged random-null pool | `runs/iteration_round_1/A2_causal_fixed/{ablation, enhancement, random_null, targeted_ce, cross_emotion, metrics}.json` | **Ablation Δ REMAINS POSITIVE on all 6 emotions with per-stem operator** (0.481, 0.830, 0.866, 0.835, 0.927, 0.599 nats) — essentially unchanged from the buggy global-mean (0.510, 0.855, 0.867, 0.857, 0.928, 0.617). The reviewer's hypothesized operator-bug explanation is FALSIFIED. Random-null now has proper variance: 3/6 emotions (fear z=+7.27, surprise z=+13.97, disgust z=+16.43) show C_e Δ above the 97.5%ile; 3/6 emotions (joy z=+0.79, sadness z=-1.20, anger z=-0.72) have C_e Δ below or at null level. Enhancement Spearman(α, Δ) unchanged (still 2/6 monotonic). Overall verdict: still not-supported (0/6 emotions pass predicate (a)). |

**Iteration 3-4 narrative refinements**

- **C2 final wording** (adopted iteration 3, verified iteration 4): "The identified components are stable across scenarios (S1/S2 Jaccard 6/6 pass) and show limited discriminative specificity (targeted-C_{e'} predicate passes 4/6 emotions), but they do not satisfy the causal-necessity criterion. Under per-stem OTHER-5 mean-substitution ablation, target-prefix log-prob shifts are uniformly positive rather than negative (0.48–0.93 nats across 6 emotions), indicating that the located components are not necessary in the expected directional sense under this intervention. This positive Δ under ablation admits two non-exclusive interpretations: (1) the located components do not encode the target emotion in the directionally-necessary sense the intervention presumes; (2) mean-substitution removes competing suppressive or non-directional signal whose net effect is to raise target-emotion probability. We do not attempt to disambiguate. Moreover, for 3/6 emotions (joy, sadness, anger), random top-K sets from the same layer pool produce larger positive shifts than the causally ranked C_e, weakening any claim of unique causal privilege at the subset level (though it is consistent with the same layers being emotion-relevant in aggregate). We therefore interpret C_e as emotion-relevant/discriminative, not as strongly verified causal necessities."

**Final status**: **not-supported (real negative, confirmed by operator fix — not an artifact)**. This is a scientifically meaningful finding: causal necessity in the intended directional sense is NOT verified by the tested ablation operator.

**Reviewer memory thread**:
- "C2 is the highest-priority scientific risk" (iteration 1) — ADDRESSED by ② fix, and the fix's outcome converted the concern from "possibly wrong result" to "correctly-obtained negative result."
- "C2 now points to a deeper mechanistic issue, not an implementation issue" (iteration 2) — the reviewer explicitly conceded the operator-bug hypothesis was ruled out.
- "Why do random sets outperform causal sets for half the emotions?" (iteration 2) — carried forward to paper narrative; not resolved in loop (would require alternative operators).

**Unverified under swaps**: C2 was cap-cut by MAX_VERIFY_CLAIMS=1 and was not stress-tested. Upgrade instruction: `/auto-verify C2 --resume=true --dimensions=model --gpu_id=1,2,3,5,6` (post-fix baseline integrity now PASS instead of WARN).

### 4b.3 `C3_v2` — the new claim from iteration-2 ③ rewrite

**Origin**: Not present at loop entry; produced by iteration 2's lightweight in-loop ③ claim rewrite (see Section 2.1).

**Final claim text** (adopted iteration 3, tightened iteration 4): as recorded in Section 2.1.

**Empirical support**: `runs/iteration_round_1/A3_applied_fixed/metrics.json` (positive part: enhancement Δ_target > 0 on all 6 emotions; targeted-C_{e'} 4/6 pass; negative part: Arm A macro 0.139 vs Arm B 0.663).

**Status**: The claim scope is defensible; the empirical support pre-exists (no new experiments needed for the rewrite).

**Unverified under swaps**: C3_v2 is a new claim from the ③ rewrite; it has not been through Stage 2 stress-test. Upgrade instruction: `/auto-verify C3_v2 --resume=true --dimensions=model --gpu_id=1,2,3,5,6`.

---

## Section 5 — Legacy DEFERRED Claims (empty under current architecture)

*None. Verify never deferred any claim; cap-cut claims land in INTEGRITY_ONLY per current architecture.*

---

## Section 6 — Cross-Cutting Patterns

- **Operator implementation drift from EXPERIMENT_PLAN.md spec** was real (C2 global-vs-per-stem mean; C3 cumulative injection at generation time) — the plan-vs-implementation alignment step of experiment-stage was incomplete. Resolved for C2 in iteration 1; addressed for C3 via a Type ② + Type ③ combo when the operator turned out to be structurally wrong for the eval regime.
- **Val-eval metric misalignment** (C3 val uses target-prefix logprob gain, eval uses judge-scored generation accuracy) — flagged in iterations 1 and 2. Not fixable within loop scope (would require expensive judge-in-the-loop val); recorded in beyond-loop items.
- **Late-stage claim narrowing** (iterations 2-5) — the standard failure mode where the paper's manuscript-level scaffold retains "verification" framing even after claim statements are corrected — was proactively addressed by iterations 3-4 verb sweeps + iteration-4/5 coarse-locatability scope-control. Reviewer confirmed the pattern was correctly repaired.
- **Discriminative locatability does not imply directional causal necessity or practical steerability** — this cross-cutting reviewer-flagged pattern became the paper's central conceptual takeaway per iteration 3's ⓪ reframing.
- **Random top-K within same layer pool wins 3/6 emotions in C2 random-null** — surfaced by the iteration-1 pool fix; became a foregrounded caveat in iterations 3-4.

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed**: 2 / 6 (only iterations 1 and 2 used back-edge actions; iterations 3-5 were ⓪ narrative-only and did not consume budget)
- **Claim-reentries consumed**: 1 / 2
- **Iteration `/run-experiment` calls**: runs_total = 2 (M2 fix, M3 fix)
- **Iteration GPU-hours**: gpu_hours_total ≈ 0.6 h (M2 ~0.2h, M3 ~0.4h, both on single-GPU due to a device_map bug that surfaced with multi-GPU device_map="auto")

### Per-iteration breakdown

| Iter | Type | Target claims | Produced claims | Runs | GPU-hours | Score after | Verdict after |
|---|---|---|---|---|---|---|---|
| 1 | ② plan_script_rerun | C2, C3 | — | 2 | 0.6 | (deferred to iter-2 review) | (deferred) |
| 2 | ③ claim_reentry (lightweight in-loop) | C3 | C3_v2 | 0 | 0 | 4 | not ready |
| 3 | ⓪ narrative-only (no budget) | (all) | — | 0 | 0 | 5 | almost |
| 4 | ⓪ narrative-only (no budget) | (all) | — | 0 | 0 | 5.5 | almost |
| 5 | ⓪ narrative-only (no budget) | (all) | — | 0 | 0 | **6** | **READY** |

Note: iteration 1's ② fix and iteration 2's ③ rewrite are the only back-edge actions in the whole loop; the reviewer's score/verdict trajectory (3→4→5→5.5→6, not ready→not ready→almost→almost→READY) reflects that once the operator and claim-scope issues were fixed, the remaining path to submission-readiness was rhetorical refinement of the manuscript.

---

## Section 8 — Open Items for Human Reviewer

- **Still-FAIL claims** (after exhausting routing options): none.
- **Still-INCONCLUSIVE claims**: none.
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: none.
- **INTEGRITY_ONLY claims (Stage 2 skipped — not stress-tested)**:
  - **C1** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C1 --resume=true --dimensions=model --gpu_id=1,2,3,5,6` (Phase 2 audits reused via RESUME; only Stages 2–3 run for this claim). main_experiment_integrity: pass with 2 WARNs.
  - **C2** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C2 --resume=true --dimensions=model --gpu_id=1,2,3,5,6` (Phase 2 audits reused via RESUME; only Stages 2–3 run for this claim). main_experiment_integrity after iteration 1 fix: pass (was WARN pre-fix).
  - **C3_v2** [status: lightweight in-loop ③ rewrite from iteration 2]: `/auto-verify C3_v2 --resume=true --dimensions=model --gpu_id=1,2,3,5,6`. New claim from ③; needs its own Stage 1 audit + Stage 2 stress test to formally confirm the narrower scope.
- **Legacy deferred claims (empty in new runs)**: none.
- **Recurring unresolved patterns**: (a) val-eval metric misalignment — target-prefix logprob is unreliable as a proxy for judge-scored generation accuracy; (b) why 3/6 emotions have C_e Δ below random-null mean — needs mechanistic explanation the loop did not pursue.
- **Claim-reentry refusals**: none — sub-budget was not exhausted (1/2 consumed).
- **Beyond-loop-scope items** (would move reviewer score from 6 to 7-8, per iteration 4 and 5 feedback): stronger null-model methodology (layer-matched + size-matched + rank-matched perturbations, multiple ablation operators, robustness across prompt templates / decoding); better causal intervention design (sign-sensitive or projection-based interventions, token-position effects, nonlinear component interactions); multi-model transfer study; theoretical account of scoring-time vs trajectory-time intervention decoupling; formalization of "prefix score modulation vs generation control" as the paper's central conceptual contribution.

---

## Appendix — Reviewer LLM configuration

- **Model**: gpt-5.4 (via `mcp__llm-chat__chat` fallback to curl; base URL `https://www.dmxapi.cn/v1`)
- **Source**: shell env (`LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`)
- **Bypass**: NO_PROXY=dmxapi.cn,www.dmxapi.cn,localhost,127.0.0.1; HTTP_PROXY / HTTPS_PROXY / all_proxy unset

All five reviewer calls (iterations 1-5) used curl fallback due to MCP server unavailability. Full raw responses saved to `/tmp/review_iter{1,2,3,4,5}_raw.txt` and referenced verbatim inside `AUTO_REVIEW.md` per-iteration entries.
