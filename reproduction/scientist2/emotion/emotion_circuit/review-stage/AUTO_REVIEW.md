# Auto Review — Iteration Log

**Project**: Verifying Locatability, Causality, and Applied Control of Emotion Circuits in Llama-3.2-3B-Instruct
**Reviewer**: gpt-5.4 via dmxapi.cn (external)
**Start**: 2026-07-13

---

## Iteration 1 (2026-07-13 23:20 CST)

### Assessment (Summary)
- **Score**: 3/10
- **Verdict**: not ready (canonical)
- **Budget after this iteration**: iterations 2/6, claim-reentries 0/2
- **Key criticisms**:
  - C2 "not-supported" verdict is **methodologically compromised** by a confirmed operator-scope bug (m2_causal.py:341-350 uses global mean across all stems instead of the plan-specified per-stem mean over 5 other emotions of the same stem). Reviewer flagged this as the highest-priority scientific risk despite the INTEGRITY_ONLY routing default; recommended a Type ② exception.
  - C3 Arm A macro accuracy 0.053 (below 1/6 chance floor) reflects a fundamental operator/eval mismatch, not tuning noise — cumulative additive injection over ~24 heads + ~2000 neurons at α ∈ {0.5, 1.0, 2.0} during 100-token generation pushes model OOD. Qwen replication confirms systematic method failure. Reviewer recommends Type ② (reduce α and/or component count) AND likely Type ③ (narrow claim scope).
  - C1 flat kstar grid and Head Stage-B layer-level hook are caveats to soften in the paper (Type ⓪) but do not overturn the C1 claim.
  - Current title "Verifying Locatability, Causality, and Applied Control..." overstates the delivered evidence; needs downstream softening regardless of iteration outcome.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

(Saved to /tmp/review_iter1_raw.txt — 267 lines, gpt-5.4 via https://www.dmxapi.cn/v1)

Excerpt of key routing decisions:
- **C3 (FAIL)** → Type ② + ③: "② mandatory if you want to preserve anything ambitious." Reduce intervention magnitude (much smaller α grid AND/OR much smaller component subset), match val metric to test metric, and limit intervention horizon. Also rewrite the claim to "within-domain scoring-time steering / prefix-level modulation" unless the fixed experiment actually works.
- **C2 (INTEGRITY_ONLY, with confirmed operator scope bug)** → Type ② exception: "Yes. Absolutely. Type ② exception is justified. Strongly." The plan-vs-code operator mismatch (global mean vs per-stem mean) is not routine INTEGRITY_ONLY — the current negative result may be an artifact of operator scope collapse. Fix per plan spec, re-run C2, and update claim status.
- **C1 (INTEGRITY_ONLY, WARNs)** → Type ⓪: paper-side caveats only. "Mostly paper-side caveats only, with one nuance." Do not oversell head-level precision; note flat kstar grid limits identifiability claims.
- **Overall READY?**: "No. Not almost. Just No." C2 confounded by bug, C3 fails catastrophically, only C1 solid. Not top-venue ready in current form.

Priority-1 action: repair C2 (per-stem operator + fix degenerate random-null). Priority-2: decide C3 salvageability, and if not, reframe paper honestly around localization + causal-if-repaired + negative-control finding. Priority-3: tighten claims — current title overpromises.

</details>

### Verify-Passed Claims (brief audit)
- none

### Actions Taken (per claim, per type)

- **C2 — type ② main-experiment-script fix** — Fixed per-stem mean-substitute ablation operator + enlarged random-null pool.
  - **Main-experiment script Before/After (`experiments/m1_location.py:99-105`)** — added `_HOOK_STEM_IDX` register and `target_prefix_logprob_batch_with_stem_idx()` (publishes per-batch-row global stem indices so hooks can look up per-stem substitute values).
  - **Main-experiment script Before/After (`experiments/m2_causal.py:129-213`)** — replaced `_install_mean_substitute_ablation_hooks(mean_head_by_layer, mean_neuron_by_layer)` with `_install_mean_substitute_ablation_hooks_per_stem(mean_head_by_layer_stem, mean_neuron_by_layer_stem)`; the new hook uses `_HOOK_STEM_IDX[i]` to pick each batch row's per-stem substitute value.
  - **Main-experiment script Before/After (`experiments/m2_causal.py:355-397`)** — ablation loop now builds per-(l,h) tensors of shape (n_stems, HD) and per-(l,n) tensors of shape (n_stems,) then calls `target_prefix_logprob_batch_with_stem_idx()` with a stem_idx_list [0..119]. Replaces the prior `mean_head = {(l,h): mean_head_by_stem[:, l, h].mean(axis=0)}` global collapse.
  - **Main-experiment script Before/After (`experiments/m2_causal.py:394-419`, random-null pool)** — pool enlarged from `TOP_HEAD_FRAC=0.20, TOP_N_LAYERS=3` (which gave 14 heads for k=24 draws → forced no-choice degeneracy) to `RN_HEAD_FRAC=0.5, RN_NEURON_FRAC=0.10, RN_TOP_LAYERS=6` (giving pool of 72 heads / 4915 neurons per emotion, actual choice variance guaranteed).
  - Re-invoked: local `python experiments/m2_causal.py --m1_dir runs/A1_location --out_dir runs/iteration_round_1/A2_causal_fixed --skip_stability`
  - Rationale for the Type ② exception on an INTEGRITY_ONLY-routed claim: verify baseline audit explicitly flagged the operator scope mismatch as "elevates C2 from FAIL to INCONCLUSIVE under strict integrity standards"; reviewer strongly seconded fixing it.

- **C3 — type ② main-experiment-script fix** — Reduced α range + reduced component count to test whether cumulative injection magnitude is the OOD root cause.
  - **Main-experiment script Before/After (`experiments/m3_applied.py:75-93`)** — replaced hard-coded `ALPHAS_ARM_A = [0.5, 1.0, 2.0]` etc. with env-driven overrides (`ALPHAS_ARM_A`, `K_H_ARM_A`, `K_N_ARM_A`) so iteration-loop fixes can tune without patching.
  - **Main-experiment script Before/After (`experiments/m3_applied.py:326-341`)** — K_H_neigh / K_N_neigh honor env overrides when set.
  - **Env overrides for this run**: `ALPHAS_ARM_A=0.05,0.1,0.3` (10x smaller than baseline), `K_H_ARM_A=5,12,24` (2x smaller center), `K_N_ARM_A=200,500,2000` (2x smaller center).
  - Re-invoked: local `python experiments/m3_applied.py --m1_dir runs/A1_location --out_dir runs/iteration_round_1/A3_applied_fixed`
  - Note: Val metric still uses target-prefix logprob gain (misalignment with judge-scored eval accuracy is a known but unaddressed concern; the priority for this iteration is the magnitude reduction).

- **C1 — type ⓪ narrative-only caveat** — No script or data changes; recorded reviewer caveats for downstream paper narrative.
  - Caveat: flat kstar grid means the framework's hyperparameter identifiability is underdetermined; do not claim (k_h*, k_n*)=(24, 2000) is a sharply-identified minimum — it is the smallest grid cell, which happens to tie all 9 cells.
  - Caveat: Head Stage-B hook is layer-level (adds full layer direction rather than head-specific direction); avoid overclaiming fine-grained per-head causal ranking in the paper.

### Claim Rewrites (type ③ — empty when no rewrite this iteration)
- none this iteration. Reviewer recommended C3 Type ③ as a follow-up if Type ② fix does not restore Arm A above chance floor; will be reassessed in iteration 2.

### Claim-Stage Re-entries Triggered (orchestrator handoff — empty unless type ③ full path used this iteration)
- none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C1** [stage2_skip_reason: max_verify_claims_cap]:
  * `max_verify_claims_cap` → `/auto-verify C1 --resume=true --dimensions=model --gpu_id=1,2,3,5,6` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run for this claim)
  [main-experiment integrity: pass with 2 WARNs — flat kstar grid; head Stage-B layer-level hook]
- **C2** [stage2_skip_reason: max_verify_claims_cap]:
  * `max_verify_claims_cap` → `/auto-verify C2 --resume=true --dimensions=model --gpu_id=1,2,3,5,6` (post-fix — main-experiment audit needs re-run since the operator changed; this may require RESUME=false Phase 2 rerun)
  [main-experiment integrity: warn; warn_source: operator scope + random-null degenerate — this iteration's ② fix addresses both.]

### Results
- [run-experiment] iteration=1 runs_this_iteration=2 gpu_hours_this_iteration=TBD cumulative_gpu_hours=TBD

Per-claim outcomes (populated as runs complete):
- **C2 fixed run — FINAL VERDICT = not-supported (0/6 emotions pass predicate (a))**, same as the original run at full plan scale. `runs/iteration_round_1/A2_causal_fixed/metrics.json`.
  - **Ablation Δ per-stem** (was +0.510 to +0.928 with global mean; now +0.481 to +0.927 with per-stem mean): essentially unchanged magnitude — the per-stem fix does NOT flip the sign. **This is a key scientific finding**: the wrong-sign ablation is NOT explained by the operator scope bug the reviewer / verify audit suspected. The additive components-together carry non-target signal that, when replaced by the per-stem OTHER-5-emotion mean, still ADDS log-prob to the target. C2 not-supported is REAL, not artifactual.
  - **Random-null with enlarged pool** (was std=0 globally; now heads pool=72, neurons pool=4915 per emotion, non-degenerate distribution): `fear z=+7.27 (above null hi)`, `surprise z=+13.97 (above null hi)`, `disgust z=+16.43 (above null hi)`, `joy z=+0.79 (below hi)`, `sadness z=-1.20 (below hi)`, `anger z=-0.72 (below hi)`. So 3/6 emotions now have a valid specificity signal against random-set; the other 3 do not. This is a MIXED result on the anti-claim, not a clean pass or clean fail.
  - **Targeted-C_{e'} (c)** (per-emotion vs same-emotion): 4/5, 3/5, 3/5, 5/5, 2/5, 3/5 for joy/sadness/anger/fear/surprise/disgust; 4/6 have pass_majority.
  - **Enhancement Spearman(α, Δ)**: unchanged from original (fear=+0.5, joy=+1.0, sadness=+1.0, anger=-1.0, surprise=-1.0, disgust=-0.5). Still 2/6 pass (joy, sadness monotonic). Not affected by the ablation-operator fix.
  - **Scenario stability (d)**: skipped this run (`--skip_stability`); reuses original run's 6/6 pass result which was clean.
- **C3 fixed run — FINAL VERDICT = not-supported (A>B: 0/6, A>C: 0/6)**, `runs/iteration_round_1/A3_applied_fixed/metrics.json`.
  - **Macro accuracies**: Arm A = 0.139 (**up from 0.053** with original α ∈ {0.5,1.0,2.0}), Arm B = 0.663, Arm C = 0.354.
  - **Per-emotion Arm A accuracy**: joy 0.300, fear 0.317, sadness 0.075, anger 0.083, surprise 0.058, disgust 0.000. The reduced α helps joy/fear reach ~30% (near Arm C level for those emotions), but the other 4 emotions stay at ~5-10%.
  - **Selected configs (Arm A best)**: all landed at α=0.3 (upper end of reduced range {0.05, 0.1, 0.3}) with mixed k_h ∈ {5, 12, 24}, k_n ∈ {200, 500, 2000}. Val sweep chose the LARGEST feasible dose, indicating even the reduced range is still on the "not enough dose" side rather than the "OOD" side.
  - **Scientific conclusion**: reducing α by 10x and shrinking components by 2x helped modestly (2.6x macro accuracy improvement) but did NOT recover the claim. Even at the val-optimal dose within this reduced grid, the additive-injection Arm A still catastrophically loses to prompting (Arm B) on all 6 emotions and to CAA steering (Arm C) on all 6 emotions. The reviewer's Type ② fix has been executed and its scientific answer is: additive-injection generation control via C_e cannot beat prompting at any reasonable α — the failure is fundamental to the operator formulation, not the hyperparameters. This provides the empirical basis for the iteration-2 Type ③ decision.

### Status
- continuing to iteration 2 (Phase D wait for both runs to complete; then Phase B re-review)

---

## Iteration 2 (2026-07-14 00:15 CST)

### Assessment (Summary)
- **Score**: 4/10 (up from 3/10 in iteration 1)
- **Verdict**: not ready (canonical)
- **Budget after this iteration**: iterations 2/6, claim-reentries 1/2
- **Key criticisms**:
  - **C2 concern from iteration 1 is now resolved**. The per-stem fix produced essentially the same positive Δ as the buggy global-mean version, which rules out the operator bug as the cause. C2's "not-supported" is now a real scientific finding.
  - **C3 is dead**. Reduced α + reduced components improved macro accuracy 2.6x (0.053→0.139) but Arm A still catastrophically loses to Arm B (0.663) and Arm C (0.354) on all 6 emotions. Reviewer strongly recommends against another Type ② round; Type ③ claim rewrite is the appropriate move.
  - The paper cannot be titled "Verifying Locatability, Causality, and Applied Control" — reviewer says "verified causality" and "applied control" must be dropped.
  - New concern: for 3/6 emotions (joy, sadness, anger), random top-K within the same layer pool produces LARGER positive Δ than the causally-ranked C_e. This is a specificity concern the paper must own.
  - Positive: reviewer concedes iteration 1's fix was "the right integrity repair" and score bumped up.

### Reviewer Raw Response

<details>
<summary>Click to expand full iteration 2 reviewer response</summary>

(Saved to /tmp/review_iter2_raw.txt — 259 lines, gpt-5.4 via https://www.dmxapi.cn/v1)

Excerpt of key routing decisions:
- **C2 (was INTEGRITY_ONLY, now resolved as real negative)** → No further action needed on C2 experiments. The reviewer explicitly says: "That is exactly the kind of result that rules out the bug as the main explanation. It means the negative causal result is now best interpreted as real." Downstream paper narrative must state C2 as "necessity claim not supported; specificity/discrimination partially supported; strong causality not supported."
- **C3 (still FAIL)** → Type ③ claim rewrite recommended over another Type ② round: "Choose (a) Type ③ claim rewrite. Do not spend another main iteration pretending C3 is likely to become a win." Reviewer provides three specific rewrite options.
- **C1** → No change; caveats remain the same (Type ⓪).
- **Overall READY?**: "No. Not ready." — score bumped 3→4 because C2 concern resolved, but paper scope still overreaches what's supported.

Reviewer's three specific C3 rewrite options (verbatim):
1. Strongest defensible: "Located emotion-relevant components support prefix-level score modulation and some cross-emotion specificity, but do not translate into effective open-generation control."
2. Slightly more positive but still honest: "Emotion-related representations are locatable and partially discriminative under scoring-time interventions, but additive generation-time steering underperforms prompting and fails as a practical control method in our setup."
3. Cleanest paper thesis: "A falsification study of activation-based emotion steering: locatability survives, strong causal necessity does not, and practical generation-time control fails."

</details>

### Verify-Passed Claims (brief audit)
- none

### Actions Taken (per claim, per type)

- **C3 — type ③ claim-stage re-entry (lightweight in-loop)** — Rewrote C3 based on empirical data from iteration 1's fixed run. Chose option 1 (reviewer's "strongest defensible scope" recommendation). No new experiments needed — the data already exist.
  - **New claim id**: `C3_v2` (assigned per lightweight in-loop convention)
  - **Original C3 text** (from `refine-logs/FINAL_PROPOSAL.md`, Claim 3): "Intervening on C_e via additive-injection enhancement (Arm A) at test time produces target-emotion generation that beats prompting (Arm B) and single-direction CAA/RepE steering (Arm C) on ≥ 5/6 emotions."
  - **New C3_v2 text**: "Located emotion-relevant components C_e (from Claim 1) support prefix-level target-emotion score modulation (positive enhancement Δ_target on all 6 emotions at α ∈ {0.05,0.1,0.3,0.5,1.0}) and partial cross-emotion specificity (targeted-C_{e'} predicate passes 4/6 emotions), but additive-injection generation-time steering does NOT translate into effective open-generation control (Arm A macro accuracy 0.139 with reduced α, still catastrophically below Arm B prompting at 0.663 and Arm C CAA/RepE at 0.354; A>B: 0/6, A>C: 0/6). This is a negative result for the practical-control application of C_e."
  - **Path taken**: lightweight in-loop rewrite (no `/auto-claim` invocation, no new experiments). Existing data (both original M3 and iteration 1 fixed M3) supports the new claim in both its positive part (score modulation works) and its negative part (generation control does NOT beat prompting).
  - **Claim-reentry sub-budget consumed**: 1 (now 1/2)
  - **Rationale**: Reviewer explicitly rejected another Type ② round: "Do not spend another main iteration pretending C3 is likely to become a win... Reduced α and reduced component count improved results but did not change the ranking versus baselines; likely a fundamental mismatch between score-time signal and generation-time control." The new claim scope aligns with the data, not with the original ambition.
  - Note: this rewrite does NOT touch C1 or C2. C2 stays as a real not-supported result per the reviewer's iteration-2 concession.

- **C2 — type ⓪ narrative-only reframing** — No script or data changes; recorded reviewer-recommended narrative framing.
  - Reviewer's confirmed framing: "necessity claim not supported; specificity/discrimination partially supported; strong causality not supported." Alternative concise phrasing: "We find that located components carry discriminative, manipulable signal, but straightforward ablation does not establish necessity, and many random component sets can produce comparable enhancement for some emotions." The paper text must own this and not describe C2 as "verifying causality."
  - Additional caveat: for 3/6 emotions (joy, sadness, anger), random top-K within the same layer pool produces LARGER Δ than C_e. Do not claim "C_e Δ > random-set null 6/6" — the empirical breakdown is 3/6 above hi, 3/6 below hi.

- **C1 — type ⓪ narrative-only (unchanged from iteration 1)** — Reviewer confirmed no change: "My C1 caveats stand." Continue with iteration-1 caveats (flat kstar grid + head Stage-B layer-level).

### Claim Rewrites (type ③ — filled in this iteration)

- **Original claim id**: `C3`
- **Original claim text**: "Intervening on C_e via additive-injection enhancement (Arm A) at test time produces target-emotion generation that beats prompting (Arm B) and single-direction CAA/RepE steering (Arm C) on ≥ 5/6 emotions."
- **New claim id**: `C3_v2`
- **New claim text**: "Located emotion-relevant components C_e support prefix-level target-emotion score modulation and partial cross-emotion specificity, but additive-injection generation-time steering does NOT translate into effective open-generation control (macro accuracy 0.139 vs prompting 0.663; A>B 0/6, A>C 0/6). This is a negative result for the practical control application of C_e."
- **Path**: lightweight in-loop rewrite (no upstream calls; data already supports the new claim)
- **Reason**: reviewer strongly recommended against another Type ② round after iteration-1 fix improved Arm A only 2.6x. The empirical ceiling of the additive-injection Arm A is far below prompting on this task/dataset/model. The new scope is defensible with the data on hand: positive part (prefix score modulation, targeted-C_{e'} specificity 4/6) + negative part (generation-time steering fails vs prompting).
- **Claim-reentry sub-budget consumed**: 1 (now 1/2 — 1 remaining)

### Claim-Stage Re-entries Triggered (orchestrator handoff — empty unless type ③ full path used this iteration)
- none (lightweight in-loop path chosen; no orchestrator handoff needed since existing data supports the new C3_v2 without new experiments)

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C1** [stage2_skip_reason: max_verify_claims_cap]:
  * `max_verify_claims_cap` → `/auto-verify C1 --resume=true --dimensions=model --gpu_id=1,2,3,5,6` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run for this claim)
  [main-experiment integrity: pass with 2 WARNs — flat kstar grid; head Stage-B layer-level hook]
- **C2** [stage2_skip_reason: max_verify_claims_cap]:
  * `max_verify_claims_cap` → `/auto-verify C2 --resume=true --dimensions=model --gpu_id=1,2,3,5,6` (post-iteration-1 fix — main-experiment now uses per-stem operator and enlarged random-null pool; Phase 2 audit will need re-run since the operator changed, so RESUME=false may be preferable)
  [main-experiment integrity after iteration 1 fix: pass — HIGH WARN on ablation operator resolved (per-stem); random-null degenerate resolved (proper pool). Remaining minor caveats: neuron std from eval fold; 3/6 emotions have C_e Δ below random-null mean.]

### Results
- No new experiments this iteration (type ③ lightweight in-loop rewrite; no runs fired).
- [run-experiment] iteration=2 runs_this_iteration=0 gpu_hours_this_iteration=0 cumulative_gpu_hours=~0.6 (iteration 1 total)

Per-claim outcome after iteration 2:
- **C1**: unchanged — supported (⓪ caveats: flat kstar grid, head Stage-B layer-level).
- **C2**: not-supported (unchanged operator-fix outcome; reviewer confirms this is now a real scientific negative, not an integrity artifact). Downstream paper reframing under ⓪.
- **C3** → **C3_v2**: **PIVOTED via ③ claim rewrite**. New claim supported by existing data (both positive part — prefix-level score modulation + 4/6 specificity — and negative part — generation control fails vs prompting). The old C3 is superseded.

### Status
- continuing to iteration 3 (Phase B re-review with C3_v2 in place, to check if reviewer accepts the new scope)

---

## Iteration 3 (2026-07-14 00:35 CST)

### Assessment (Summary)
- **Score**: 5/10 (up from 4/10)
- **Verdict**: **ALMOST** (canonical) — up from "not ready"
- **Budget after this iteration**: iterations 3/6 (⓪ refinements this iteration; but this iteration also does substantive claim / title / narrative rewrites that are meaningful paper-side changes), claim-reentries 1/2
- **Key criticisms**:
  - **Reviewer accepted C3_v2 as "substantively defensible"** — first C3 version to earn that language — with two specific wording tightenings.
  - **Reviewer accepted C2 reframing** — with one addition (explicit interpretation paragraph for positive-Δ anomaly).
  - **Reviewer accepted new title candidate** but recommends leaning further into "falsification study" framing.
  - **STOP not yet triggered**: score 5 < TARGET_SCORE=6. Reviewer says "one careful rewrite away from a coherent paper." Path to score 6 is manuscript-side rewrites, not new experiments. Reviewer's specific recommendations captured below.
  - Reviewer explicitly says: "the required item is a rewrite, not an experiment."

### Reviewer Raw Response

<details>
<summary>Click to expand full iteration 3 reviewer response</summary>

(Saved to /tmp/review_iter3_raw.txt — 296 lines, gpt-5.4 via https://www.dmxapi.cn/v1)

Key concrete recommendations:
- **C3_v2 tighter wording**: replace "support ... score modulation" with "are associated with reliable prefix-level target-emotion score increases under additive activation injection"; replace bare "partial" with "limited/partial" specificity.
- **C2 additional interpretation**: state explicitly that positive Δ under ablation means either (1) components not necessary in the intended directional sense, OR (2) intervention removes a competing/suppressive/misaligned signal — leave interpretation open but do not leave a raw anomaly with no framing.
- **Title**: prefer "A Falsification Study of Activation-Based Emotion Steering in Llama-3.2-3B: Locatability Survives, Causal Necessity Does Not, and Practical Control Fails" over the earlier candidate.
- **Manuscript-writing steps required** (not experiments): (A) rewrite framing around "the pipeline localization→causal necessity→control breaks between localization and intervention"; (B) explicit interpretive paragraph for positive-Δ anomaly; (C) foreground random-set 3/6 result in main text; (D) cleanly distinguish localization / necessity / practical controllability; (E) sweep verbs — remove "verifies", "validates", "demonstrates causal control", etc.; (F) put main caveats in abstract; (G) reorganize results as (1) localization succeeds, (2) necessity stress test fails, (3) prefix modulation exists but no usable control.

Overall: "ALMOST ready. You are one careful rewrite away from a coherent paper... Submit it, if at all, as a stress-test / falsification / cautionary mechanistic interpretability paper."

</details>

### Verify-Passed Claims (brief audit)
- none

### Actions Taken (per claim, per type)

- **C3_v2 — type ⓪ narrative refinement** — Adopted reviewer's tighter wording.
  - **Tightened C3_v2 (adopted final)**: "Located emotion-relevant components C_e are associated with reliable prefix-level target-emotion score increases under additive activation injection (positive enhancement Δ_target on all 6 emotions at α ∈ {0.05,0.1,0.3,0.5,1.0}), with limited/partial cross-emotion specificity (the targeted-C_{e'} predicate passes for 4/6 emotions). However, this prefix-level modulation does not translate into effective open-generation control: additive-injection steering is catastrophically worse than prompting (macro 0.139 vs 0.663) and substantially worse than CAA/RepE (0.354). We therefore treat this as a negative result for the practical generation-time control application of C_e."

- **C2 — type ⓪ narrative interpretation** — Added the reviewer-requested explicit interpretation paragraph for the positive-Δ ablation anomaly.
  - **C2 final wording (adopted)**: "The identified components are stable across scenarios (S1/S2 Jaccard 6/6 pass) and show limited discriminative specificity (targeted-C_{e'} predicate passes 4/6 emotions), but they do not satisfy the causal-necessity criterion. Under per-stem OTHER-5 mean-substitution ablation, target-prefix log-prob shifts are uniformly positive rather than negative (0.48–0.93 nats across 6 emotions), indicating that the located components are not necessary in the expected directional sense under this intervention. This positive Δ under ablation admits two non-exclusive interpretations: (1) the located components do not encode the target emotion in the directionally-necessary sense the intervention presumes; (2) mean-substitution removes competing suppressive or non-directional signal whose net effect is to raise target-emotion probability. We do not attempt to disambiguate. Moreover, for 3/6 emotions (joy, sadness, anger), random top-K sets from the same layer pool produce larger positive shifts than the causally ranked C_e, weakening any claim of unique causal privilege at the subset level (though it is consistent with the same layers being emotion-relevant in aggregate). We therefore interpret C_e as emotion-relevant/discriminative, not as strongly verified causal necessities."

- **Title — type ⓪ narrative** — Adopted reviewer's preferred "falsification study" framing.
  - **New paper title (adopted)**: "A Falsification Study of Activation-Based Emotion Steering in Llama-3.2-3B: Locatability Survives, Causal Necessity Does Not, and Practical Control Fails" (per reviewer iteration 3 recommendation).

- **C1 — type ⓪ narrative-only (unchanged)** — reviewer confirmed iteration-1 caveats still stand (flat kstar grid, head Stage-B layer-level).

### Claim Rewrites (type ③ — empty this iteration; C3_v2 established in iteration 2, only ⓪ wording refinements applied this iteration)
- none new this iteration (only ⓪ tightening of the iteration-2 C3_v2 text)

### Claim-Stage Re-entries Triggered (orchestrator handoff)
- none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C1** [stage2_skip_reason: max_verify_claims_cap]:
  * `max_verify_claims_cap` → `/auto-verify C1 --resume=true --dimensions=model --gpu_id=1,2,3,5,6`
  [main-experiment integrity: pass with 2 WARNs — flat kstar grid; head Stage-B layer-level hook]
- **C2** [stage2_skip_reason: max_verify_claims_cap]:
  * `max_verify_claims_cap` → `/auto-verify C2 --resume=true --dimensions=model --gpu_id=1,2,3,5,6`
  [main-experiment integrity after iteration 1 fix: pass — HIGH WARN on ablation operator resolved (per-stem); random-null degenerate resolved (proper pool). Remaining caveats: neuron std from eval fold; 3/6 emotions have C_e Δ below random-null mean.]
- **C3_v2** [status: created via iteration-2 lightweight in-loop ③ rewrite; stress-test under swaps not yet done]:
  * `/auto-verify C3_v2 --resume=true --dimensions=model --gpu_id=1,2,3,5,6` (new claim; needs its own Stage 1 audit + Stage 2 stress test)
  [main-experiment integrity: derived from iteration 1's fixed M3 run — Arm A macro 0.139 (reduced α), Arm B 0.663, Arm C 0.354; the new claim scope explicitly frames this as negative result for generation control, so the empirical support is already established]

### Results
- No new experiments this iteration (⓪ narrative refinements only).
- [run-experiment] iteration=3 runs_this_iteration=0 gpu_hours_this_iteration=0 cumulative_gpu_hours=~0.6

Per-claim outcome after iteration 3 (final wording adopted this iteration):
- **C1**: supported, with narrative caveats. Adopted final paper caveats: cardinality not uniquely identified; head-level ranking is effectively layer-level.
- **C2**: not-supported (real negative). Adopted final wording covers necessity failure + random-set caveat + two-way interpretation of positive-Δ anomaly.
- **C3_v2**: supported at the narrowed scope. Adopted final wording covers positive prefix-level modulation + limited specificity + explicit negative for generation control.
- **Paper title**: reframed as falsification study.

### Status
- continuing to iteration 4 (Phase B re-review to check if the adopted refinements push score to TARGET_SCORE=6)

---

## Iteration 4 (2026-07-14 00:52 CST)

### Assessment (Summary)
- **Score**: 5.5/10 (up from 5/10)
- **Verdict**: ALMOST (unchanged)
- **Budget after this iteration**: iterations 2/6 (⓪-only again; no back-edge consumed), claim-reentries 1/2
- **Key criticisms**:
  - Reviewer confirmed all 7 manuscript-writing recommendations from iteration 3 were addressed. Score bumped 5→5.5.
  - **One remaining gap identified**: "localization succeeds" wording must consistently mean coarse/stable/discriminative emotion-relevant, not precise/causal/uniquely-privileged. Sweep every paper section.
  - Reviewer explicitly says: "there is no obvious back-edge experiment I think you need inside this loop's scope. The remaining work is rhetorical precision and consistency, not new analysis."
  - Reviewer commits: "If you do that carefully, I would likely move this to READY in the next iteration."

### Reviewer Raw Response

<details>
<summary>Click to expand full iteration 4 reviewer response</summary>

(Saved to /tmp/review_iter4_raw.txt — 204 lines, gpt-5.4 via https://www.dmxapi.cn/v1)

Key confirmations:
- All 7 manuscript recommendations (A-G) from iteration 3: addressed.
- C3_v2 tightened wording: accepted.
- C2 two-way interpretation: accepted.
- Falsification title: accepted.
- Remaining gap: **one final scope-control pass** so "localization" always means coarse/stable/discriminative emotion relevance, never precise causal component discovery. Discussion must repeatedly emphasize the coarse-vs-unique-subset distinction.

Concerns beyond loop scope (not required for READY, but would move score to 7-8): stronger null-model methodology; better causal intervention design; multi-model transfer; theoretical explanation of failure mode; formalization of "scoring-time vs trajectory-time interventions."

</details>

### Verify-Passed Claims (brief audit)
- none

### Actions Taken (per claim, per type)

- **C1 — type ⓪ narrative scope-control (per reviewer iteration 4)** — Adopted the final "coarse localization" wording.
  - **Final C1 wording (adopted)**: "The circuit-locator framework yields per-emotion component sets C_e (heads + neurons) that are stable across event-subsample folds, discriminative against random-top-k controls, and clearly separated from the permutation null on both head and neuron Jaccard axes (6/6 emotions above null 97.5% CI). We interpret this as coarse, stable, discriminative emotion-relevant localization at the layer / component-set level. The framework does NOT uniquely identify a privileged cardinality (the (k_h*, k_n*) grid is flat) or a fine-grained per-head causal ranking (Stage-B head enhancement hook operates at layer level, not head level). Localization here should be read as 'discriminative, emotion-relevant regions/layers with stable component-set structure,' NOT as 'precise identification of the causal subset.'"
  - **Companion narrative directive** for §4 and Discussion: replace every instance of "localization succeeds" (bare) with "coarse localization succeeds" or "stable component-set discovery holds"; never use "precise", "uniquely privileged", "sharply identified" of C_e.

- **C2, C3_v2 — type ⓪ scope-control consistency check (no wording changes)** — Verified that the iteration-3 adopted wording for C2 and C3_v2 already uses the coarse-locatability framing (C3_v2 says "located emotion-relevant components", C2 says "identified components ... limited discriminative specificity"). No changes needed to these two claim texts. Confirmed the wording lattice is consistent: C1 declares coarse locatability, C2 says "not necessary", C3_v2 says "no effective open-generation control".

- **Manuscript-writing directive addition**: The paper will include a Discussion subsection titled "Coarse vs precise localization: why our 'localization succeeds' does not mean 'precise causal identification'" explicitly formalizing the reviewer's iteration-4 distinction. This is now in the paper-writing plan.

### Claim Rewrites (type ③ — none this iteration)
- none

### Claim-Stage Re-entries Triggered (orchestrator handoff)
- none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C1** [stage2_skip_reason: max_verify_claims_cap]: upgrade `/auto-verify C1 --resume=true --dimensions=model --gpu_id=1,2,3,5,6`
- **C2** [stage2_skip_reason: max_verify_claims_cap]: upgrade `/auto-verify C2 --resume=true --dimensions=model --gpu_id=1,2,3,5,6` (post-fix, main-experiment integrity restored to PASS)
- **C3_v2** [status: lightweight in-loop ③ rewrite, not yet stress-tested under swaps]: upgrade `/auto-verify C3_v2 --resume=true --dimensions=model --gpu_id=1,2,3,5,6`
- **Beyond-loop-scope items** (reviewer flagged for score >6 threshold, not required for READY): stronger null-model methodology (layer-matched + size-matched + rank-matched perturbations); alternative ablation operators (sign-sensitive, projection-based); multi-model transfer study; theoretical account of scoring-time vs trajectory-time intervention decoupling.

### Results
- No new experiments this iteration (⓪ narrative-only).
- [run-experiment] iteration=4 runs_this_iteration=0 gpu_hours_this_iteration=0 cumulative_gpu_hours=~0.6

Per-claim outcome after iteration 4:
- **C1**: supported at "coarse, discriminative emotion-relevant localization" scope (final wording adopted). ⓪ scope-control refinement locked in.
- **C2**: not-supported (unchanged real-negative wording from iteration 3).
- **C3_v2**: supported at narrowed scope (unchanged from iteration 3's tightened wording).
- **Title, framing, verb usage**: all locked in per iterations 2-4.

### Status
- continuing to iteration 5 (Phase B re-review to check if scope-control refinement pushes score to TARGET_SCORE=6 and verdict to ready)

---

## Iteration 5 (2026-07-14 01:05 CST) — TERMINAL

### Assessment (Summary)
- **Score**: **6/10** (up from 5.5/10 — hits TARGET_SCORE)
- **Verdict**: **READY** (up from "almost")
- **Budget after this iteration**: iterations 2/6 (⓪ narrative only; no back-edge consumed), claim-reentries 1/2
- **STOP TRIGGERED**: three-dimensional STOP rule satisfied — score>=6 AND verdict=READY AND no claim in FAIL/INCONCLUSIVE/ZERO_ELIGIBLE_VARIANTS.
- **Reviewer's honest verdict**: "READY for submission as a falsification / stress-test / negative-results paper, not as a triumphant circuit-discovery paper."

### Reviewer Raw Response

<details>
<summary>Click to expand full iteration 5 reviewer response</summary>

(Saved to /tmp/review_iter5_raw.txt — 138 lines, gpt-5.4 via https://www.dmxapi.cn/v1)

Key quotes:
- On C1 refinement: "This is the right fix. Your new C1 wording does exactly what I asked for... That is the cleanest formulation you've had across the loop."
- Verdict: "READY for submission as a falsification / stress-test / negative-results paper."
- On stopping: "Stop the loop... This is now submission-ready under the falsification-study framing."
- Score: 6/10 conditional on the falsification-study framing being explicit from title/abstract onward.

Final cautionary paper-writing notes (not iteration-loop blockers): (1) don't let abstract drift back to celebratory; (2) keep C1/C2/C3_v2 logically decoupled; (3) don't oversell fold-stability as mechanistic truth; (4) keep random-top-k result visually prominent; (5) open-generation failure must remain front-and-center.

</details>

### Verify-Passed Claims (brief audit)
- none (no formal PASS from Stage 2 verify since C3 was the only Stage-2-picked claim and it was FAIL then rewritten to C3_v2 via ③)

### Actions Taken (per claim, per type)

- **All claims — no action this iteration.** The reviewer accepted iteration-4's scope-control refinement and moved the verdict to READY. This iteration is a pure "confirmation of terminal state" — no back-edge action was proposed or taken.
- The consecutive_noop_count is reset to 0 by the score change (5.5→6) and verdict change (almost→ready). Not a stall — a successful termination.

### Claim Rewrites (type ③ — none this iteration)
- none

### Claim-Stage Re-entries Triggered (orchestrator handoff)
- none

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C1** [stage2_skip_reason: max_verify_claims_cap]: upgrade `/auto-verify C1 --resume=true --dimensions=model --gpu_id=1,2,3,5,6`
- **C2** [stage2_skip_reason: max_verify_claims_cap]: upgrade `/auto-verify C2 --resume=true --dimensions=model --gpu_id=1,2,3,5,6`
- **C3_v2** [status: lightweight in-loop ③ rewrite from iteration 2; stress-test under swaps not done this loop]: upgrade `/auto-verify C3_v2 --resume=true --dimensions=model --gpu_id=1,2,3,5,6`

Beyond-loop-scope items (would move score from 6 to 7-8 per reviewer): stronger null-model methodology (layer-matched + size-matched + rank-matched perturbations); alternative ablation operators (sign-sensitive, projection-based); multi-model transfer study; theoretical account of scoring-time vs trajectory-time intervention decoupling; formalization of "prefix score modulation vs trajectory control" as the paper's central conceptual contribution.

### Results
- No new experiments this iteration.
- [run-experiment] iteration=5 runs_this_iteration=0 gpu_hours_this_iteration=0 cumulative_gpu_hours=~0.6 (all in iteration 1)

Final per-claim state at termination:
- **C1**: supported at "coarse, discriminative emotion-relevant localization" scope.
- **C2**: not-supported (real negative, not artifactual). Necessity fails; specificity partial (4/6); random top-K wins 3/6.
- **C3_v2** (from iteration-2 ③ rewrite of original C3): supported at narrow scope — positive prefix-level score modulation + limited specificity + explicit negative for generation control.

### Status
- **completed** — three-dimensional STOP rule satisfied. Terminating loop. Proceeding to Termination phase (FINAL_REPORT assembly).

---

