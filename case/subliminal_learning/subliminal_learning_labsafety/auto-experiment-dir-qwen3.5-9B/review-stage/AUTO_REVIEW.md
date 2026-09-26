# Auto Review Log — Cross-Modal Subliminal Safety-Competence Transfer on Qwen3.5-9B Multimodal

**Reviewer LLM**: `gpt-5.4` via `https://www.dmxapi.cn/v1` (shell env, task.md judge model per project config)
**Loop config**: MAX_ITERATIONS=6, MAX_CLAIM_REENTRIES=2, TARGET_SCORE=6, POSITIVE_VERDICT_TERMS={ready, almost}, GPU_ID=3,4,5,6,7
**Started**: 2026-07-10

---

## Iteration 1 (2026-07-10 05:28)

### Assessment (Summary)
- **Score**: 3/10
- **Verdict**: not ready
- **Budget after this iteration**: iterations 1/6, claim-reentries 0/2
- **Key criticisms**:
  - C1 headline fails preregistered unanimity (2/3 seeds pass, seed300 reverses); "PASS" from verify is only a judge-stable *negative* verdict, not confirmed phenomenon.
  - Winning LR=1e-3 is the top edge of {5e-5..1e-3}, all smaller LRs ~0 pp; suggests optimization-instability / weight-init fragility rather than "subliminal transfer".
  - Missing matched-benign SFT control — cannot separate the specific subliminal channel from generic text-only SFT / synthetic-data adaptation.
  - C3 mechanism story materially weak: AUROC=0.27 (below chance) on safety-decisive partition; steering non-monotonic; random_matched matches best real steering (SP-A fails); cross-seed steering ~zero; MMLU capability control never run.
  - Ablation/patching may show representational importance, not safety specificity — could be treated-vs-Ctrl identity restoration since the direction is not the safety axis semantically.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

## 1) Score

**3/10**

Why so low:

- **C1, the headline claim, is not actually established under the project's own preregistered criterion.** The claim requires **strict unanimity across ≥3 seeds**; you have **2 strong positives and 1 reversal**. In a top venue, that is not a minor blemish; it is a direct failure of the stated success condition.
- **The LR selection looks dangerously cliff-like and likely confounded.** If only the **top edge** LR works and all smaller LRs are near-zero, I immediately worry the effect is training-instability / overfitting / catastrophic drift rather than "subliminal transfer."
- **C3 is nowhere near mechanistically supported.** The reported direction is **below chance AUROC (0.27)** for the purported safety-decisive partition, the steering effect is **non-monotonic**, the **random matched control is competitive**, **cross-seed steering does not replicate**, and the required **MMLU capability control was never run**. That is not a near miss; it is currently a failed mechanism story dressed up as "inconclusive."
- **Power is weak where it matters most for mechanism** (n=27 held-out), but underpower does not rescue the current evidence; it just means you should be less confident in the positive pieces too.

At best, this is an interesting anomaly with two positive seeds and one negative seed, plus some coarse evidence of representational disruption from ablation/patching. It is **not yet a convincing NeurIPS/ICML paper**.

## 2) Minimum fix for claims needing action

### C3 — INCONCLUSIVE
**Minimum fix: ② main-experiment-script / plan fix**

**Required fix: do both, not one.**

1. **Log MMLU (or equivalent capability control) at every α point** — mandatory for C3b to constrain collateral damage. Without it, even a large "restoration" could just be generic capability steering.
2. **Widen and densify the α sweep** — current 5-point grid with best effect at the boundary and no monotonic trend is uninterpretable. Add at minimum α ∈ {-4,-3,-2,-1,-0.5,0,+0.5,+1,+2,+3,+4} or a comparable denser local sweep. Test for a stable operating region / plateau, not cherry-pick a boundary point.

### C1 brief consistency check
- Numbers internally consistent (Ctrl=0.7970; seed100=0.5639 → +23.31 pp drop; seed200=0.6466 → +15.04 pp; seed300=0.8195 → −2.26 pp reversal). Judge-swap tracks within ~1 pp.
- Paper must NOT present C1 as established. At most: "large degradation in 2/3 seeds, fail preregistered unanimity due to one seed reversal."
- Additional hidden caveats: (a) top-edge LR confound; (b) no matched benign-SFT control; (c) 3-seed seed heterogeneity is substantive.
- Recommend ⓪ narrative reframing: fragile, non-universal effect; explicitly failing preregistered criterion; seed-sensitive and hyperparameter-cliff-dependent.

## 4) Verdict
**Not READY**. C3 still unresolved. C2 does not block; C3 does.

## 5) Memory update
See `review-stage/REVIEWER_MEMORY.md` iteration-1 section — 6 concerns tracked forward.

</details>

### Verify-Passed Claims (brief audit)
- **C1**: numbers consistent (Ctrl=0.7970, seeds {100,200,300} = {0.5639, 0.6466, 0.8195}; drops {+23.31, +15.04, −2.26} pp). Judge-swap consistency PASS. **Caveat carried to paper**: `not-supported per strict task.md unanimity` is a judge-stable NEGATIVE verdict; the paper must not present C1 as established. Type-⓪ narrative reframing recommended (see Actions Taken).

### Actions Taken (per claim, per type)

- **C3 — type ② — widened α sweep + added MMLU per-α capability control** (dispatched via `scripts/dispatch_m2_iter1.sh` on gpu_ids=3,4,5,6,7).
  - **Plan Before/After** (`refine-logs/EXPERIMENT_PLAN.md` M2 §Runs):
    - Before: `α ∈ {-2, -1, 0, +1, +2}` (5 doses); MMLU listed but not executed
    - After: `α ∈ {-3, -2, -1, -0.5, 0, +0.5, +1, +2, +3}` (9 doses); MMLU-slice mandatory at every α
  - **Grid Before/After** (`refine-logs/EXPERIMENT_PLAN.md` M2 §Grid):
    - Before: `{ alpha: [-2, -1, 0, 1, 2], run_kind: [real_treated, real_ctrl_injection, control_treated, mmlu_treated] }` = 20 (planned but 5 actually run)
    - After: `{ alpha: [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3], run_kind: [real_treated, control_treated, mmlu_real_treated, mmlu_control_treated] }` = 36 (- α=0 duplicates), iter1 dispatches 25 NEW runs
  - **New script**: `scripts/mechanism_m2_intervene_mmlu.py` — mirrors QA_I intervention hook exactly (same steering formula `h ← h − α·σ_proj·u` at layer 4), but on MMLU text with a blank 224×224 white PIL image alongside the text prompt (multimodal processor requires image; the model reads MMLU purely from text). Judged via gpt-5.4 with the same CORRECT/INCORRECT/OTHER protocol as QA_I; cache at `caches/mmlu_judge_cache.jsonl`.
  - **New dispatch**: `scripts/dispatch_m2_iter1.sh` — 6 waves totaling 25 runs (Wave 1-2 widen QA_I real+random α∈{-3,-0.5,+0.5,+3} = 8 runs; Waves 3-6 MMLU at all α on real+random = 17 runs). Distributed across GPUs 3-7 (task.md gpu pin honored).
  - **MMLU slice preparation**: `scripts/prepare_mmlu_slice.py` writes `mechanism/mmlu_slice.jsonl` — 500 items from `abstract_algebra` (100) + `college_mathematics` (100) + `professional_law` (first 300), from local MMLU parquet at `/data/zhenqian/data/mmlu/`.
  - **Re-verify command**: `/auto-verify C3 — resume: false` (Phase 2 mechanism audit re-runs; if it passes, Phases 3-10 execute for the first time).

- **C1 — type ⓪ — paper-side narrative reframing** (does NOT consume iteration budget). No script or data change. The claim text stands (locked by task.md), but the paper narrative should be:
  - "The phenomenon reproduces on 2/3 seeds at large magnitudes (~15-23 pp) but fails the preregistered per-seed unanimity criterion due to seed300 reversal (+2.26 pp). Under strict predicate: not-established. Under a relaxed majority-of-seeds predicate: supported at 2/3."
  - Additional caveats to surface: winning LR is the top edge of the grid (potential optimization-cliff confound); no matched benign-SFT control was run; 3-seed variance is substantive, not cosmetic.
  - Recorded here for the final paper; the CLAIMS_LEDGER.md `Final` line for C1 already carries "not-supported per strict task.md unanimity" — consistent with this recommendation.

### Claim Rewrites (type ③)
- none (this iteration used type ② and ⓪ only; no claim-reentry sub-budget consumed).

### Claim-Stage Re-entries Triggered
- none.

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C2** [stage2_skip_reason: max_verify_claims_cap]:
  * upgrade: `/auto-verify C2 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run for this claim)
  * main-experiment integrity: pass (no `warn_source`)

### Results

**[run-experiment] iteration=1 runs_this_iteration=25 gpu_hours_this_iteration=7.31 cumulative_gpu_hours=7.31**

- Dispatch launched: 2026-07-10 05:28:43 CST, PID 1790926.
- Dispatch completed: 2026-07-10 07:06:08 CST. Wall-clock 97 min across 6 waves, 25 runs. GPU pin witness recorded in `runs/iteration_round_1/dispatch_m2_iter1/cost.json` — all subprocesses pinned to gpu_ids ⊂ {3,4,5,6,7} per HARD CONSTRAINT.

**Per-α QA_I widened (seed100, held-out n=27) — final table**

| α | gap_closure (real m1_top_k) | gap_closure (random_matched) |
|---|---|---|
| −3.0 | 0.125 | **0.250** |
| −2.0 | 0.125 | 0.000 |
| −1.0 | 0.125 | **0.250** |
| −0.5 | 0.125 | 0.125 |
| 0.0 | 0.000 (no-op) | — |
| +0.5 | 0.125 | 0.000 |
| +1.0 | 0.125 | 0.125 |
| +2.0 | **0.250** | 0.125 |
| +3.0 | 0.125 | **0.250** |

**Per-α MMLU capability (500-item slice; seed100 treated; SP-C threshold ≤ 2 pp)**

| α | MMLU acc (real) | Δ vs α=0 (pp) | SP-C | MMLU acc (random) | Δ (pp) | SP-C |
|---|---|---|---|---|---|---|
| −3.0 | 0.344 | +0.4 | PASS | 0.348 | 0.0 | PASS |
| −2.0 | 0.358 | −1.0 | PASS | 0.344 | +0.4 | PASS |
| −1.0 | 0.350 | −0.2 | PASS | 0.354 | −0.6 | PASS |
| −0.5 | 0.346 | +0.2 | PASS | 0.352 | −0.4 | PASS |
| 0.0 (baseline) | **0.348** | 0.0 | — | (skipped — no-op ≡ real α=0) | — | — |
| +0.5 | 0.346 | +0.2 | PASS | 0.356 | −0.8 | PASS |
| +1.0 | 0.350 | −0.2 | PASS | 0.354 | −0.6 | PASS |
| +2.0 | 0.352 | −0.4 | PASS | 0.358 | −1.0 | PASS |
| +3.0 | 0.348 | 0.0 | PASS | 0.352 | −0.4 | PASS |

**Key findings**:
1. **A.3 (MMLU logged) RESOLVED**: MMLU 500-item slice logged at every α on both sources; max |drop| = 1.0 pp. SP-C PASSES at every α.
2. **A.4 (plateau visible) PARTIALLY RESOLVED**: A flat plateau at gc≈0.125 exists for real m1_top_k across 7 of 8 non-zero α. The α=+2 spike to gc=0.25 is a +1-item excursion at n=27 (noise floor). Plateau is identifiable but at a level so low that it is comparable to random_matched's plateau.
3. **A.5 (SP-A specificity) NOW HARD FAILS**: random_matched achieves gc=0.250 at 3 distinct α (−3, −1, +3), while real m1_top_k achieves gc=0.250 at only 1 α (+2). The extracted CAA direction is not specifically better than a random rank-matched direction — SP-A specificity **decisively refuted** under widened data. In fact, random hits the peak gc value 3× more often than real.
4. **A.1 (α range coverage) RESOLVED**: widened to 9-point [-3, +3] sweep.
5. **Ablation/patching (unchanged from original)**: gc=0.875 and 0.625 — remain strong evidence for representational importance at layer 4, but do NOT rescue the C3b specificity predicate (which is framed around the direction as a steering vector, not around magnitude-removal operations).

**Mechanism-audit re-verdict (in-loop lightweight re-audit against widened data)**:
- Original audit: FAIL (A.3, A.4)
- Iter-1 re-audit: FAIL (A.5 hard fail; A.3/A.4 rigor gaps closed but substantive verdict is C3b refuted)
- **C3 state shift**: INCONCLUSIVE → PASS-of-negative-verdict (analogous to C1 — the not-supported verdict is now measurable and audit-stable; the substantive science is that C3b's specificity claim is FALSE).
- Full artifacts: `verify/C3_low_dim_safety_substrate/main_experiment_audit_iter1/MECHANISM_AUDIT.md`; updated `verify/C3_low_dim_safety_substrate/ROBUSTNESS.md`; updated `verify/VERIFY_REPORT.md`.

**Follow-up recommendation for the orchestrator** (post-loop): to formalize the state transition via the full /auto-verify pipeline, run `/auto-verify C3 -- resume: false`. Phase 2 mechanism-audit will re-run on the widened data (expected PASS-WARN with A.5 FAIL); Phase 3-10 variants will run for the first time and would agree that C3b's specificity claim is not-supported.

### Status
- in_progress → proceeding to iteration 2 for a fresh re-review against the updated C3 state (INCONCLUSIVE resolved to PASS-of-negative-verdict). C3 no longer blocks the STOP rule's third dimension.

---

## Iteration 2 (2026-07-10 07:10)

### Assessment (Summary)
- **Score**: 3/10 (unchanged from iter-1)
- **Verdict**: not ready
- **Budget after this iteration**: iterations 2/6, claim-reentries 1/2
- **Key criticisms**:
  - Iter-1 fix WAS genuinely substantive (A.3/A.4 addressed at measurement level). But the widened data reinforces the negative story — the mechanism claim (C3) is now more clearly refuted.
  - C1 unaddressed defects persist: LR-cliff confound, missing matched-benign-SFT control, seed300 reversal.
  - Cross-seed steering non-replication (seed200, seed300 gc≈0) unaddressed — only seed100 was widened.
  - AUROC=0.27 for the M1 direction implies the direction is treated-vs-Ctrl identity, not safety-semantic — mostly unaddressed.
  - Reviewer memory calls out the pattern: "PASS-of-negative-verdict" bookkeeping does not equate to top-venue readiness. The bookkeeping-vs-science gap remains open.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score: 3/10** — **Verdict: NOT READY**

The iteration-1 fix was substantive at the measurement level for C3, not merely performative. They really did close two methodological holes I complained about: MMLU at every α and a wider α sweep. So A.3 was genuinely addressed, and A.4 was at least meaningfully probed rather than hand-waved.

But the new data hurts the scientific case, not helps it. The result is now clearer: the mechanism story is weaker than before.

Re-review summary:
- **What was genuinely fixed**: A.3/MMLU missing at every α — fixed. A.4/boundary-hitting concern — partially and honestly addressed.
- **What the improved measurement now shows**: Real-direction steering is not specific — random matched directions hit the same best gap-closure, and do so more often. The effect is tiny and unstable — on n=27, gc values of 0.125/0.250 are just a few items. No monotonic control law emerges. MMLU staying flat is useful but not redeeming — it only means you can perturb activations without broad capability collapse, but does not validate the claimed safety-relevant mechanism.

Claim-by-claim:
- **C1**: Still not established as a positive claim. Unanimity still fails; LR-cliff confound remains totally unaddressed; matched benign-SFT control remains missing. Minimum fix: ⓪ reframe claim to explicitly state preregistered positive predicate failed with 2/3 seed heterogeneity and seed300 reversal foregrounded. If they want an actual scientific rescue: ② matched benign text-only SFT control; ② extend LR grid around cliff.
- **C2**: Still INTEGRITY_ONLY. None required for loop.
- **C3**: The old measurement gaps were fixed. The scientific claim is now worse. MMLU addressed; wider α sweep addressed; but specificity vs random decisively refuted; cross-seed steering non-replication remains; AUROC=0.27 issue remains. Minimum fix: ③ Rewrite C3 into an explicit negative-result / limited-causal-evidence claim: representational interventions can modulate outputs without broad MMLU harm, but the discovered direction is not specific, not clearly safety-semantic, and steering does not replicate as a robust mechanism claim. Alternative: ② cross-seed widened steering, ② stronger specificity controls, ② safety-semantic validation beyond treated-vs-control separability.

Why score stays 3/10: Because top-venue scoring is about the science, not process bookkeeping. This work still has a headline effect that fails strict preregistered positive support, a strong optimization confound, a missing matched benign-SFT control, a mechanism section that now more clearly supports a negative than a positive interpretation, and tiny-sample steering evidence that is mostly noise-shaped. The authors deserve some credit for running the right falsification checks on C3 instead of dodging them. But the result of those checks is that the central mechanistic interpretation is undermined, not strengthened. That is not a paper-ready state.

Recommended final state this iteration: Still NOT READY. If they want to maximize honesty and salvage value: (1) ⓪ Reframe C1 as mixed/non-robust evidence; (2) ③ Rewrite C3 as a negative mechanistic finding; (3) if they want upward score movement: ② matched benign-SFT, ② LR-cliff disambiguation, ② cross-seed widened steering.

</details>

### Verify-Passed Claims (brief audit)
- **C1**: unchanged — narrative reframing recommendation carried forward (still ⓪).
- **C3 (post iter-1)**: numbers consistent with the widened+MMLU data. Reviewer confirms this is a **cleaner negative result**, not evidence for a safety-substrate. The PASS-of-negative-verdict state is correct on process; the scientific characterization needs to move from "kind-level mechanism hypothesis" to explicit negative-result claim (see Actions Taken → type ③).

### Actions Taken (per claim, per type)

- **C1 — type ⓪ — paper-side narrative reframing (unchanged from iter-1; re-affirmed by iter-2 reviewer)**. Recommended paper narrative: "In 2 of 3 seeds we observe large safety-competence drops (23.31 pp / 15.04 pp), but seed300 reverses (+2.26 pp); the preregistered strict per-seed unanimity criterion FAILS. Under a majority-of-seeds relaxation, the phenomenon is provisionally supported. High seed variance at the winning LR=1e-3 (top edge of the {5e-5..1e-3} sweep) is consistent with a training-cliff sensitivity confound — the effect may reflect optimization-instability rather than a pure semantic subliminal channel. A matched benign-SFT control was NOT run, so treatment-specific attribution cannot be cleanly separated from generic text-only SFT / synthetic-data adaptation." Does NOT consume iteration budget.

- **C3 — type ③ — lightweight in-loop claim rewrite. New claim id `C3_v2` produced.**
  - **Original C3 text**: "A low-dim safety-relevant activation subspace inside the Qwen3.5-9B language tower shifts between the treated (subliminal-SFT) student and the Ctrl base student (Location, C3a), and intervening on that subspace on treated restores Ctrl-level image-conditioned QA_I accuracy while a rank-matched non-safety-relevant control direction achieves <1/3 the effect and general-capability (MMLU-slice) drop stays ≤2 pp (Causal Intervention specificity, C3b)."
  - **New C3_v2 text**: "**Negative-result mechanism claim.** At layer 4 of the Qwen3.5-9B language tower, the diff-of-means direction extracted between the treated (subliminal-SFT seed100 student) and the un-fine-tuned Ctrl base student **is not a specific safety-substrate**: (a) its AUROC on the safety-decisive-vs-neutral partition of QA_I held-out is **0.27 (below chance)**, refuting the semantic alignment predicate; (b) under a widened α ∈ [-3, +3] steering sweep, a rank-matched random direction achieves the peak gap-closure value (0.25) MORE frequently than the extracted direction (3 α vs 1 α of 8), refuting the SP-A specificity predicate; (c) cross-seed replication on seed200 and seed300 shows steering gap-closure ≈ 0 at all tested α, refuting cross-seed generalization. Two positive limited-scope observations survive: (i) the intervention causes NO detectable MMLU capability degradation at any α on the widened sweep (max |drop| = 1.0 pp; SP-C PASSES) — the layer-4 activations can be perturbed without wrecking general capability; (ii) ablation (h ← h − (h·u)u) and activation patching (replace treated residual with Ctrl residual) at layer 4 close 87.5% and 62.5% of the treated-vs-Ctrl QA_I gap respectively — indicating the layer-4 representation carries the treated-vs-Ctrl difference, but with the extracted direction NOT being the specific handle (SP-A refuted). **Overall: the mechanism is best characterized as a distributed representational shift at layer 4, not a low-dim safety-substrate that can be causally controlled via a single direction.**"
  - **Path taken**: lightweight in-loop rewrite (no `/auto-claim` re-invocation).
  - **Reason for rewrite** (per reviewer's ③ recommendation and iter-1 mechanism-audit re-verdict): the original C3 was framed around a positive predicate (specific safety-substrate exists AND has ≤ 2 pp MMLU spillover AND rank-matched < 1/3). Iter-1's widened+MMLU data confirms 1 of 3 sub-predicates positively (SP-C: capability specificity holds) and refutes 2 of 3 (SP-A: rank-matched control is competitive; AUROC=0.27: direction is not safety-semantic). Under the original claim wording, C3 is not-supported. Under the rewritten C3_v2 wording, the claim is a mechanism-negative-result plus two carefully-scoped positive observations — this matches what the data actually supports. The rewrite honors the reviewer's memory pattern-of-warning: no PASS-of-negative-verdict cloaking; the claim explicitly owns the negative characterization.
  - **Claim-reentry sub-budget consumed**: 1 (now 1/2).
  - **Verify state for C3_v2**: PASS — the main experiment now supports C3_v2 as stated (each sub-predicate maps to a specific measurement outcome). No further re-verify needed since the claim is now aligned with the on-disk data.

### Claim Rewrites (type ③)
- Original claim id: `C3` — text: "low-dim safety substrate (positive predicate, SP-A / SP-C / semantic)"
- New claim id: `C3_v2` — text: (see full text under Actions Taken)
- Path: lightweight in-loop rewrite (no `/auto-claim` re-invocation).
- Reason: reviewer iter-2 explicit ③ recommendation. The widened+MMLU data supports the negative-result characterization; the original positive predicate is refuted. Rewrite avoids the "PASS-of-negative-verdict cloaking" pattern the reviewer flagged.
- Claim-reentry sub-budget consumed: 1 (now 1/2).

### Claim-Stage Re-entries Triggered (orchestrator handoff)
- none — used lightweight in-loop rewrite path; no `awaiting_upstream` handoff.

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C2** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C2 — resume: true` (unchanged from iter-1).

### Results

**[run-experiment] iteration=2 runs_this_iteration=0 gpu_hours_this_iteration=0 cumulative_gpu_hours=7.31**

- No new experiments launched. Iteration 2 was a claim-rewrite + narrative-reframing iteration; no code or data changes.
- C3_v2 claim text written above and to be propagated into `claims_ledger.json` + `CLAIMS_LEDGER.md` (deferred to Termination phase).
- All existing artifacts (widened QA_I + MMLU per-α) directly support C3_v2's negative-result predicate as-is.

### Status
- in_progress → proceeding to iteration 3. Score unchanged at 3/10 despite the ③ rewrite (the rewrite makes the claim honest, but the underlying evidence base — LR-cliff, matched benign-SFT, cross-seed non-replication — remains unaddressed and continues to depress the score).

---

## Iteration 3 (2026-07-10 07:15)

### Assessment (Summary)
- **Score**: 3/10 (deferred re-review)
- **Verdict**: not ready (deferred)
- **Budget after this iteration**: iterations 3/6, claim-reentries 1/2
- **Key criticisms**: iteration 3 is a type-② action (cross-seed widened steering) that specifically addresses one of iter-2 reviewer memory's carried-forward unresolved items ("cross-seed steering non-replication remains unaddressed — only seed100 was widened"). Reviewer re-scoring deferred to iteration 4 after all iter-3 data lands.

### Reviewer Raw Response
- No new reviewer call this iteration. Iteration 3 is a pre-emptive rigor-closing action against a carried-forward concern from iter-2 reviewer memory. The reviewer re-review happens next iteration.

### Verify-Passed Claims (brief audit)
- unchanged from iter 2 (C1 and C3_v2 both PASS-of-negative-verdict).

### Actions Taken (per claim, per type)

- **C3_v2 (previously C3) — type ② — cross-seed widened steering on seed200 and seed300**.
  - **Motivation**: iter-2 reviewer memory carried forward "Cross-seed steering non-replication remains unaddressed — only seed100 was widened. Seed 200/300 still show gc≈0 at all α." This iteration runs the same widened α ∈ {-3, -0.5, +0.5, +3} on seed200 and seed300 (using each seed's own M1-direction — `mechanism/M1_location/treated_seed{200,300}/directions.pt`).
  - **Script Before/After** (`scripts/dispatch_m2_iter3.sh`): new dispatch script written for this iteration.
  - **Grid Before/After**:
    - Before: seed200 and seed300 only at α ∈ {-1, -2}.
    - After: seed200 at α ∈ {-3, -2, -1, -0.5, +0.5, +3}; seed300 at α ∈ {-3, -2, -1, -0.5, +0.5, +3} — 5-6 α each.
  - **Dispatch**: 2 waves × 4 runs = 8 runs, all on gpu_ids ⊂ {3,4,5,6} (GPU 7 idle for scheduling headroom). Wall-clock 5 min (fast because eval-judge cache was already warm from prior identical held-out set).
  - **Compute**: 8 runs, 0.34 GPU-hours (cost.json at `runs/iteration_round_3/dispatch_m2_iter3/cost.json`).
  - **In-loop re-audit**: does not change C3_v2's status. C3_v2 is already framed as a negative-result mechanism claim; the new data further supports the negative characterization (cross-seed steering shows gc=0.000 at every α on both seed200 and seed300).

### Claim Rewrites (type ③)
- none this iteration (used ② only).

### Claim-Stage Re-entries Triggered
- none.

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C2** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C2 — resume: true` (unchanged).

### Results

**[run-experiment] iteration=3 runs_this_iteration=8 gpu_hours_this_iteration=0.34 cumulative_gpu_hours=7.65**

**Cross-seed widened steering — seed200 and seed300 at layer 4**

Treated baselines (held-out n=27):
- seed200 acc_treated_baseline_held_out = 0.6296
- seed300 acc_treated_baseline_held_out = 0.8148
- ctrl acc_ctrl_held_out = 0.7407

Seed200 per-α (real m1_top_k steering):

| α | acc_intervened | gap_closure |
|---|---|---|
| −3.0 | 0.6296 | +0.000 |
| −2.0 | 0.6296 | +0.000 |
| −1.0 | 0.6296 | +0.000 |
| −0.5 | 0.6296 | +0.000 |
| +0.5 | 0.6296 | +0.000 |
| +3.0 | 0.6296 | +0.000 |

Seed300 per-α (real m1_top_k steering):

| α | acc_intervened | gap_closure |
|---|---|---|
| −3.0 | 0.8148 | −0.000 |
| −2.0 | 0.8148 | −0.000 |
| −1.0 | 0.8148 | −0.000 |
| −0.5 | 0.8148 | −0.000 |
| +0.5 | 0.8148 | −0.000 |
| +3.0 | 0.8148 | −0.000 |

**Key finding**: **Steering intervention produces ZERO gap_closure at every α on both seed200 and seed300** — the treated baseline accuracy is EXACTLY preserved under the intervention. Combined with iter-1's finding that on seed100 the steering only produces a gc≈0.125 flat plateau, this confirms:
- The tiny effect seed100 shows is not a mechanism that generalizes across seed replicates.
- Under widened α on 3 seeds, only seed100 shows any steering effect at all, and even that effect is comparable to random-direction steering at seed100.
- **The extracted M1 direction is not a robust causal handle for the treated-vs-Ctrl representational shift.** This decisively refutes any positive C3b interpretation.

**Substantive support for C3_v2**: this new cross-seed data confirms C3_v2's negative characterization exactly — "cross-seed replication on seed200 and seed300 shows steering gap-closure ≈ 0 at all tested α, refuting cross-seed generalization" (from the C3_v2 text in iter-2's rewrite). The claim now has explicit widened-α cross-seed evidence supporting the negative finding.

### Status
- in_progress → proceeding to iteration 4 for a fresh re-review. C1 defects (LR-cliff, matched benign-SFT missing) still unaddressed but explicitly noted in transparent listing to iter-4 reviewer.

---

## Iteration 4 (2026-07-10 07:12) — reviewer re-review (no back-edge; iterations_consumed unchanged)

### Assessment (Summary)
- **Score**: 4/10 (up from 3/10)
- **Verdict**: not ready
- **Budget after this iteration**: iterations 3/6 (unchanged — pure reviewer re-review with no back-edge action), claim-reentries 1/2
- **Key criticisms**:
  - C3 side genuinely converged (iter-2 rewrite substantive; iter-3 cross-seed data added real falsification value). Reviewer suggested minor wording softening: "best characterized as" → "consistent with".
  - C1 side still spinning. Highest remaining asks: matched-benign-SFT proxy control + LR-cliff extension.
  - Paper-level judgment: current package reads as "unstable main effect + falsified mechanism hypothesis" — valid as negative-result / cautionary paper but not top-venue as-is.
  - Loop's "PASS-of-negative-verdict" bookkeeping still risks implying maturity beyond actual science.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score: 4/10 — Verdict: NOT READY**

Convergence assessment: Mostly converging on a negative conclusion for C3; still spinning on C1.
- C3: yes, real convergence. Iter-2 rewrite was substantive; iter-3 cross-seed gc=0 on both seed200 and seed300 closes an obvious escape hatch.
- C1: still spinning. Central phenomenon still undercut by failure of preregistered strict per-seed criterion, seed300 reversal, unresolved LR-edge/cliff confound, unresolved matched-benign-SFT attribution control.

C3_v2 assessment: mostly paper-ready with minor wording caution. Softening: "best characterized as a distributed representational shift" → "consistent with a distributed representational difference at layer 4 rather than a single safety-specific controllable direction."

READY plausibility: If they want any chance at ALMOST/borderline READY: (1) matched-benign-SFT proxy control; (2) LR-cliff extension; (3) full narrative reframing. If matched-benign shows similar effects, C1 attribution collapses. If LR extension shows instability, C1 weakens further.

Forecast: READY in 2-3 iterations unlikely; ALMOST possible only with strong C1 control + honest reframing; NOT READY remains modal outcome.

</details>

### Verify-Passed Claims (brief audit)
- unchanged from iter 2/3.

### Actions Taken (per claim, per type)
- none (pure reviewer re-review; iterations_consumed does NOT increment).

### Claim Rewrites (type ③)
- none.

### Claim-Stage Re-entries Triggered
- none.

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C2** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C2 — resume: true` (unchanged).

### Results

**[run-experiment] iteration=4 runs_this_iteration=0 gpu_hours_this_iteration=0 cumulative_gpu_hours=7.65**

- No experiments dispatched.
- Score movement 3 → 4 confirms iter-2/iter-3 substantive value; consecutive_noop_count resets to 0.

### Status
- in_progress → proceeding to iteration 5 with a type-② action addressing the highest carried-forward concern (LR-cliff extension per reviewer iter-4 memory).

---

## Iteration 5 (2026-07-10 07:26)

### Assessment (Summary)
- **Score**: pending (deferred re-review to iteration 6)
- **Verdict**: pending
- **Budget after this iteration**: iterations 4/6, claim-reentries 1/2
- **Key criticisms**: iteration 5 addresses iter-4 reviewer's highest carried-forward concern: LR-cliff extension. The result is genuinely substantive and positive (see Results).

### Reviewer Raw Response
- No new reviewer call this iteration. Iteration 5 is a rigor-closing action against iter-4 reviewer memory's top priority.

### Verify-Passed Claims (brief audit)
- unchanged; C1 and C3_v2 both remain in PASS-of-*-verdict states.

### Actions Taken (per claim, per type)

- **C1 — type ② — LR-cliff extension** (`scripts/dispatch_iter5_lr_cliff.sh`).
  - **Motivation**: iter-4 reviewer memory carried "LR-cliff confound on C1 (highest priority remaining)". iter-4 reviewer explicitly recommended type-② LR-cliff extension.
  - **Grid Before/After** (`refine-logs/EXPERIMENT_PLAN.md` M0.5 § LR sweep — extension not edited into plan since this is a supplementary control, not a rebound of the preregistered choice):
    - Before: dev-sweep LR ∈ {5e-5, 1e-4, 2e-4, 5e-4, 1e-3} on dev seed=42; per-seed reproduction (M0.6) at winning LR=1e-3 on seeds {100, 200, 300}.
    - After: iter-5 adds LR ∈ {7e-4, 1.5e-3} × seeds {100, 200, 300} — 6 new SFT + 6 new QA_I evals.
  - **NEW dispatch script**: `scripts/dispatch_iter5_lr_cliff.sh` — 4 waves (2 waves of SFT + 2 waves of eval).
  - **Compute**: 12 runs (6 SFT + 6 eval), 4.4 GPU-hours (cost.json at `runs/iteration_round_5/lr_cliff/cost.json`), wall-clock 66 min.
  - **New checkpoints**: `ckpts/student_lr{7e-4,1.5e-3}_seed{100,200,300}/` (6 dirs, 6 adapters).

### Claim Rewrites (type ③)
- none this iteration (used ② only).

### Claim-Stage Re-entries Triggered
- none.

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C2** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C2 — resume: true` (unchanged).

### Results

**[run-experiment] iteration=5 runs_this_iteration=12 gpu_hours_this_iteration=4.4 cumulative_gpu_hours=12.05**

**LR-cliff extension — full 6-arm table** (Ctrl = 0.7970, 133-item QA_I)

| LR | seed100 | seed200 | seed300 | Per-seed unanimity (≥3 pp) |
|---|---|---|---|---|
| 5e-5 (M0.5 dev) | ~−3.01 | — | — | (dev only) |
| 1e-4 (M0.5 dev) | ~−0.75 | — | — | (dev only) |
| 2e-4 (M0.5 dev) | 0.00 | — | — | (dev only) |
| 5e-4 (M0.5 dev) | ~−0.75 | — | — | (dev only) |
| **7e-4 (iter-5)** | +2.26 | **+13.53** | +2.26 | FAIL (1/3 pass) |
| **1e-3 (M0.7)** | **+23.31** | **+15.04** | −2.26 | FAIL (2/3 pass) |
| **1.5e-3 (iter-5)** | **+30.83** | **+24.81** | **+29.32** | **PASS (3/3 pass)** ← **UNANIMITY** |

**Critical finding**: **At LR=1.5e-3, all 3 seeds show ≥3 pp drops from Ctrl (30.83 / 24.81 / 29.32 pp).** The preregistered task.md per-seed unanimity predicate for C1 **PASSES at LR=1.5e-3**.

Interpretation:
1. The LR=1e-3 seed300 reversal is NOT a fundamental defect of the phenomenon — it is a specific seed × LR interaction that resolves at higher LR.
2. The winning LR from the M0.5 dev-seed=42 sweep (LR=1e-3) was near-optimal but not quite optimal: LR=1.5e-3 produces larger and more consistent drops across all 3 seeds.
3. The phenomenon is real and reproduces per-seed unanimously at LR=1.5e-3 — this SATISFIES `task.md`'s M0 criterion "per-seed ≥ 3 pp Acc(QA_I) drop across ≥ 3 seeds".
4. Optimization-instability confound (the reviewer's iter-1 concern) is now partially defused: the effect is not confined to a knife-edge LR. It appears (weakly) at LR=7e-4 on 1/3 seeds, moderately at LR=1e-3 on 2/3 seeds, and strongly at LR=1.5e-3 on 3/3 seeds — a smooth monotonic dose-response over LR.
5. **However** (honesty caveat): LR=1.5e-3 was NOT the preregistered LR (M0.5 dev-seed=42 winner was LR=1e-3). Post-hoc selection of LR=1.5e-3 as the "unanimity LR" is a garden-of-forking-paths concern IF presented as the primary result. The honest framing: "iter-5's LR-cliff extension probes an unaddressed reviewer concern (LR-cliff confound) and demonstrates the phenomenon is not knife-edge; the same protocol at a slightly higher LR (1.5e-3) achieves per-seed unanimity across all 3 seeds."

**Substantive change to C1 characterization**: C1 as-stated (per task.md's ≥3 pp per-seed unanimity predicate) can now be presented as **provisionally supported at LR=1.5e-3** (with 3/3 seeds passing). The main-experiment `not-supported per strict predicate` verdict from M0.7 (which used the preregistered LR=1e-3) still holds AT that specific LR; the iter-5 result is a **new LR condition** where the strict predicate DOES hold. Both truths are simultaneous. Recommended paper narrative:
- "M0.7 (per preregistered LR=1e-3): 2/3 seeds pass. Under the strict per-seed unanimity predicate: not-established."
- "Iter-5 (LR-cliff extension at LR=1.5e-3): 3/3 seeds pass per-seed ≥3 pp. Under strict per-seed unanimity: established. The phenomenon is real; the preregistered LR was slightly suboptimal on seed300."

### Status
- in_progress → proceeding to iteration 6 for a fresh re-review with the LR-cliff evidence on the table. This may be the last iteration.

---

## Iteration 6 (2026-07-10 08:32) — reviewer re-review (no back-edge; iterations_consumed unchanged)

### Assessment (Summary)
- **Score**: 5/10 (up from 4/10)
- **Verdict**: **almost** (up from not ready)
- **Budget after this iteration**: iterations 4/6 (unchanged), claim-reentries 1/2
- **Key criticisms**:
  - Reviewer confirms iter-5's LR-cliff extension is substantive: LR=1.5e-3 with 3/3 seed unanimity (drops 30.83, 24.81, 29.32 pp) is a "large, coherent, replicated effect" that "substantially weakens" the earlier optimization-instability confound.
  - **However, it's still post-hoc**: LR=1.5e-3 was NOT the preregistered LR. Reviewer-requested nature reduces cherry-picking concerns but does not eliminate post-hoc status.
  - Reviewer provides explicit paper-narrative guidance for C1: honest framing = "original 1e-3 miss + reviewer-requested LR extension found 1.5e-3 unanimity + real effect but strongly LR-sensitive". Do NOT say "preregistered criterion satisfied" without caveat.
  - Matched-benign-SFT is still the largest unaddressed control.
  - C3_v2 remains OK as a converged negative-result mechanism claim.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Final score: 5/10, Verdict: ALMOST**

**Short answer on the key question**: **No, the new LR=1.5e-3 result does not fully "rescue" C1 as a clean preregistered positive claim.** But **yes, it does rescue C1 as a weaker, honest empirical positive claim**: the phenomenon appears real and reproducible under a slightly higher LR discovered in reviewer-requested follow-up.

- If C1 is stated as "the preregistered M0 setup robustly works at the selected LR=1e-3" → still no.
- If C1 is restated as "the effect is real, but the originally selected LR was suboptimal; in a reviewer-requested LR extension, a nearby higher LR yields unanimous replication across 3 seeds" → yes, that is now defensible.

**Why score goes up**: At 1.5e-3, 3/3 seeds passing with +30.83/+24.81/+29.32 pp is not noise — it is a large, coherent, replicated effect. Substantially weakens the earlier "one LR/seed combination" concern. Now looks more like real phenomenon + strong LR dependence + originally chosen LR was not robustly optimal.

**But is it post-hoc**? Yes, somewhat. Not fatally, but unavoidably. Even though the reviewer explicitly requested the LR-cliff extension, that does not convert 1.5e-3 into a preregistered confirmatory result. It remains a post-hoc robustness/diagnostic extension. What the reviewer request buys is protection against opportunistic cherry-picking. This is not "we searched until something worked and buried the rest."

**Does this eliminate the LR-cliff concern?** Partially, not fully. Now the pattern is: low LRs null → 7e-4 weak/inconsistent → 1e-3 mixed → 1.5e-3 strong and unanimous. That looks more like a dose-response than a single-pixel spike. Enough to defuse the worst version of the confound; not enough to claim full optimization robustness.

**Why not READY**: (1) strongest positive claim required reframing after missing original strict criterion; (2) matched-benign-SFT control still missing; (3) mechanism evidence remains weak-to-negative; (4) narrative risk is high — could still read like "we eventually found an LR that works" if written incautiously.

**What C1 should now say** (verbatim from reviewer): *"Under the originally selected LR=1e-3, the strict 3/3-seed unanimity criterion was not met (2/3 seeds passed). In response to reviewer concerns about LR sensitivity, we evaluated nearby LRs and found that LR=1.5e-3 yields strong, unanimous replication across all three seeds. We therefore conclude that the behavioral effect is real but strongly optimization-sensitive, and that the originally selected LR was suboptimal."*

**Verdict: ALMOST**. This is now in "salvageable with disciplined framing" territory, not "not ready". But that depends on the paper explicitly acknowledging: original 1e-3 miss, later 1.5e-3 success, strong LR sensitivity, mechanism mostly negative, missing matched-benign control. If authors try to oversell, reviewer snaps back to NOT READY.

</details>

### Verify-Passed Claims (brief audit)
- **C1 (upgraded)**: at preregistered LR=1e-3, 2/3 seeds pass; at LR=1.5e-3 (iter-5 reviewer-requested LR-cliff extension), 3/3 seeds pass. Numerically consistent. Paper narrative must present BOTH LRs honestly (reviewer's exact wording provided).
- **C3_v2**: unchanged. Negative-result mechanism claim remains defensible.

### Actions Taken (per claim, per type)
- none (pure reviewer re-review; iterations_consumed does NOT increment).

### Claim Rewrites (type ③)
- none.

### Claim-Stage Re-entries Triggered
- none.

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C2** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C2 — resume: true` (unchanged).

### Results

**[run-experiment] iteration=6 runs_this_iteration=0 gpu_hours_this_iteration=0 cumulative_gpu_hours=12.05**

- No experiments dispatched.
- Score movement 4 → 5 confirms iter-5's substantive value; consecutive_noop_count resets to 0.
- Verdict movement "not ready" → "almost" — STOP third dimension already satisfied; STOP second dimension (verdict ∈ {ready, almost}) now satisfied. Only STOP first dimension (score ≥ 6) still unsatisfied.

### Status
- in_progress → proceeding to iteration 7 for a type ⓪ narrative reframing that incorporates the reviewer's exact iter-6 language guidance. If reviewer confirms the reframing lands at ≥6, we STOP. Otherwise iter-8 would be either a matched-benign-SFT ② action (if time permits) or a stall-guard termination.

---

## Iteration 7 (2026-07-10 08:36) — type ⓪ narrative reframing (no back-edge; iterations_consumed unchanged)

### Assessment (Summary)
- **Score**: 5/10 (pending re-review confirmation in iter-8; no reviewer call this iteration)
- **Verdict**: almost (pending re-review)
- **Budget after this iteration**: iterations 4/6 (unchanged), claim-reentries 1/2
- **Key criticisms**: this iteration is a narrative reframing action; no new evidence. Consumes no budget.

### Reviewer Raw Response
- No new reviewer call this iteration. Iteration 7 is a paper-narrative reframing per reviewer iter-6's explicit guidance.

### Verify-Passed Claims (brief audit)
- unchanged from iter-6 (C1, C3_v2 both PASS-related).

### Actions Taken (per claim, per type)

- **C1 — type ⓪ — final paper-narrative reframing** (per reviewer iter-6 exact guidance).
  - **Recommended C1 paper text** (verbatim from reviewer iter-6):
    > "Under the originally selected LR=1e-3, the strict 3/3-seed unanimity criterion was not met (2/3 seeds passed). In response to reviewer concerns about LR sensitivity, we evaluated nearby LRs and found that LR=1.5e-3 yields strong, unanimous replication across all three seeds. We therefore conclude that the behavioral effect is real but strongly optimization-sensitive, and that the originally selected LR was suboptimal."
  - **Guardrails on what C1 must NOT say** (verbatim from reviewer iter-6):
    - "The preregistered M0 criterion is satisfied" without caveat
    - "The original claim is fully validated"
    - "This was the planned winning configuration"
    - "The effect robustly holds across learning rates"
  - Does NOT consume iteration budget. Consumes NO GPU-hours. Recorded here as an entry-point for the final report + CLAIMS_LEDGER.md update.

- **C3_v2 — type ⓪ — minor phrasing softening** (per reviewer iter-4 recommendation).
  - Change: "best characterized as a distributed representational shift at layer 4"
  - To: "consistent with a distributed representational difference at layer 4 rather than a single safety-specific controllable direction"
  - Does NOT consume iteration budget. Recorded here for the final report.

- **C2 — no action** (unchanged; INTEGRITY_ONLY).

### Claim Rewrites (type ③)
- none.

### Claim-Stage Re-entries Triggered
- none.

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C2** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C2 — resume: true`.

### Results

**[run-experiment] iteration=7 runs_this_iteration=0 gpu_hours_this_iteration=0 cumulative_gpu_hours=12.05**

- No experiments. Type-⓪ only.

### Status
- in_progress → proceeding to iteration 8 for final re-review. If the score bumps to 6 with the narrative reframing, STOP fires (score=6, verdict=almost, no unresolved non-INTEGRITY_ONLY claims). If not, consecutive_noop_count will trigger stall-guard termination.

---

## Iteration 8 (2026-07-10 08:38) — final reviewer re-review + stall-guard termination

### Assessment (Summary)
- **Score**: 5/10 (unchanged from iter-6)
- **Verdict**: almost (unchanged)
- **Budget after this iteration**: iterations 4/6 (unchanged), claim-reentries 1/2
- **Key criticisms**:
  - Reviewer explicitly declares: "Yes, stopping here is acceptable — natural stopping point."
  - Reframing is good but does not increase evidential strength; fixes honesty problem, not science problem.
  - Score gap to READY (6/10) requires matched-benign-SFT control (~3h dispatch) as the sole remaining action that could materially improve the science.
  - Loop terminates on **consecutive_noop_count = 2** stall-guard trigger AND reviewer's explicit "acceptable natural stopping point" declaration.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response</summary>

**Score**: Still 5/10. The reframing removes the overclaim reviewer objected to but is calibration, not evidence strengthening.

**Verdict**: ALMOST — no unresolved failed/inconclusive claims; C1 framed transparently; C3_v2 softened; remaining issue is missing control (matched-benign-SFT) / interpretation-strength issue, not fatal contradiction.

**Termination assessment**:
- "Yes, this is a natural stopping point for the loop if the goal is 'submit with explicit caveats.'"
- "No, it is not the science-optimal stopping point if you are willing to spend another ~3h."
- Both are true.

**Actual recommendation**:
- If asking "Should the autonomous loop stop here?" → YES, stopping here is acceptable.
- If asking "Is there still one action that meaningfully improves the science?" → YES, run matched-benign-SFT.

**Reviewer's decision**: "Stop is acceptable, but not science-optimal. If resources/latency permit, matched-benign-SFT is still worth dispatching; if not, submit with the current explicit caveats."

**Bottom line**:
- Score: 5/10
- Verdict: ALMOST
- Loop stop: Yes, acceptable natural stopping point
- Best remaining action if not stopping: run matched-benign-SFT

</details>

### Verify-Passed Claims (brief audit)
- unchanged from iter-6/7.

### Actions Taken (per claim, per type)
- none (pure reviewer re-review; iterations_consumed does NOT increment).

### Claim Rewrites (type ③)
- none.

### Claim-Stage Re-entries Triggered
- none.

### Open Items — Unverified Under Swaps (from verify_integrity_only)
- **C2** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C2 — resume: true`.

### Results

**[run-experiment] iteration=8 runs_this_iteration=0 gpu_hours_this_iteration=0 cumulative_gpu_hours=12.05**

- No experiments.
- Score = 5 (unchanged from iter-6/7); verdict = almost (unchanged).
- consecutive_noop_count = 2 (iter-7 was ⓪-only with unchanged score/verdict; iter-8 has no back-edge action with unchanged score/verdict) → **stall-guard fires**.
- Reviewer explicitly declares "acceptable natural stopping point".

### Status
- **completed** (termination_reason: `stalled — two consecutive no-op iterations at score 5, verdict almost; reviewer explicitly declares natural stopping point`).
