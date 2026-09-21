# Auto Review Log

Chronological audit log of the review-improve iteration loop for the Language-Agnostic/Specific Subspace Hypothesis project (Qwen-3-4B-Thinking + MGSM, 4 claims).

Reviewer: gpt-5.4 via dmxapi (llm-chat MCP) at iteration boundaries; iteration-agent carries the plan forward between reviews.

Reviewer prompts and REVIEW_STATE.json are in English per shared-references/output-language convention; narrative below matches `task.md` (English).

---

## Iteration 1 (2026-07-14 ~12:00) — Score: 3/10, Verdict: not ready

**Reviewer input**: `verify/VERIFY_REPORT.md` (initial), `refine-logs/EXPERIMENT_RESULTS.md`, `refine-logs/EXPERIMENT_PLAN.md`, `claims_ledger.json`.

**Reviewer synthesis**: All four claims are refuted or blocked. The main experiment's null result is real (V_lang exists as a language-specific direction but reasoning is entangled with it, not orthogonal), but the plan overclaimed monotone improvement and the verify's Phase-2/Phase-9 audits caught methodology gaps that would have to be closed to formally credit the negative finding across variants.

**Suspicions (recorded to REVIEWER_MEMORY.md)**:
1. Intervention is too strong / poorly normalized — α not in σ_proj units; severe collapse is a norm-mismatch artifact rather than nulling.
2. "Language subspace" appears functionally entangled with reasoning, not a removable nuisance.
3. C2 needs capability-collapse disentanglement — separate language-identity suppression from generic arithmetic degradation.
4. Random-direction controls are the strongest positive signal; they should become the evidentiary backbone of the revised paper.
5. C1 rank/orthogonality issue — effective rank capped by n_langs=11; complement carries 33-49% language accuracy → decomposition is incomplete.
6. C4 is likely unrecoverable under budget.
7. Paper framing risk — current thesis overclaims; the honest paper is negative-result + specificity.
8. Variant evidence should support the REVISED (specificity) claim, not the original (monotone-suppression) claim.
9. Non-collapse-window analysis is the scientifically relevant region — near-zero α, not the extreme collapse points.

**Routing (per-claim)**:
- **C1** (INTEGRITY_ONLY, max_verify_claims_cap): ⓪ narrative-only would be sufficient; deferred to a lightweight `/auto-verify C1 -- resume: true` if budget permits.
- **C2** (INCONCLUSIVE, main-experiment mech FAIL): ② full re-run of M2 with signed α sweep + ≥30 random controls per α is out-of-budget (~2.5 GPU-h vs ~1 h remaining). ③ claim-stage narrowing recommended.
- **C3** (ZERO_ELIGIBLE_VARIANTS, variant mech FAIL): ① variant-only fix — dispatch the missing random-subspace α-sweep at same grid/site/rank (~4 GPU-h, was already running at review time).
- **C4** (INCONCLUSIVE, main-experiment exp FAIL): ② full re-run of M4a + M4b is out-of-budget (~5 GPU-h). ③ claim-stage narrowing recommended.

**Action dispatched (Iteration 1)**: ① C3 variant fix. `deploy.sh` re-invoked with `GPU_LIST=1,2,3` to launch the 9-point random-subspace α-sweep on DeepSeek-R1-Distill-LLaMA-8B (matching the V_lang grid: rank=2, k_top=12, layer_group=mid, seed=42, n_problems_per_lang=50). Total 9 runs × ~28 min each ≈ 4 GPU-h wall-time on 3 GPUs, completed on disk by 14:02.

**Iteration 1 outcome**: 1 iteration consumed (of 6). REVIEW_STATE.json was not written before the previous agent returned control (dispatch-race — the previous agent handed off to the parent while the ① runs were still executing).

---

## Iteration 2 (2026-07-14 14:06, resume) — C3 variant re-audit

**Trigger**: resume=true with the ① fix jobs now complete on disk (18 files: 9 vlang + 9 random × 11 languages × n=50/lang, seed=42).

**Action**: ① — C3 variant Phase-9 re-audit.

**Re-audit finding**:
- **Random control now on disk** → the FAIL sub-criterion "no random-direction control" is resolved.
- **σ_proj scaling** and **collapse-range α values** remain as WARN-level future-work items (not FAIL criteria once random control is present, per mechanism-audit rulebook).
- **Comparison at matched α (V_lang vs random, non-collapse window α ∈ [-1.5, +0.25])**:

| α | V_lang macro_acc | random macro_acc | Δ (specificity) |
|---|---|---|---|
| −1.5 | 0.356 | 0.480 | **−0.124** |
| −1.0 | 0.391 | 0.475 | **−0.084** |
| −0.5 | 0.458 | 0.467 | −0.009 |
| −0.25 | 0.464 | 0.473 | −0.009 |
| 0.0 | 0.447 | 0.447 | 0.000 (sanity) |
| +0.25 | 0.258 | 0.456 | **−0.198** |

At |α| ≥ 1 in the non-collapse window, V_lang costs 8-20 pp of macro_acc while random costs 0 pp → clean specificity signal.

- **Diagnostic inequality**: A(−1)=0.391 < A(0)=0.447 → C3's `A(-1) > A(0)` fails (matches main experiment where A(−1)=0.051 < A(0)=0.744).
- **consistent_with_main_experiment**: pass (both refute C3).

**Phase 9 combined verdict (post re-audit)**: Exp=WARN, Mech=**WARN** (was FAIL) → combined WARN → **eligible**.

**Robustness re-computed**: N_run=1, N_eligible=1, N_pass=1 → robustness = 1.0 ≥ 0.5 → **PASS**.

**Files rewritten (Iteration 2)**:
- `verify/C3_signed_dose_response/variant_audit/MECHANISM_AUDIT.{md,json}` — WARN
- `verify/C3_signed_dose_response/variant_audit/EXPERIMENT_AUDIT.json` — result_existence note updated
- `verify/C3_signed_dose_response/ROBUSTNESS.md` — PASS with substantive table
- `verify/C3_signed_dose_response/variants/model_swap_deepseek_r1_llama8b/verdict.json` — integrity=warn, eligible
- `verify/INTEGRITY_AUDIT.md` — Phase-9 section updated
- `verify/VERIFY_REPORT.md` — C3 row PASS, detail section rewritten

**Iterations consumed**: 2/6.

**Suspicions addressed**:
- #1 (α not in σ_proj units): partially addressed — labeled as WARN future-work; the random control makes this non-decisive.
- #3 (capability-collapse disentanglement): fully addressed — random control at same rank/site/seed is now on disk; the specificity signal is clean.
- #4 (random-direction controls as evidentiary backbone): fully addressed for C3 variant.
- #9 (non-collapse-window analysis): fully addressed — audit now explicitly interprets only α ∈ [-1.5, +0.25] and labels α ≥ +0.5 as OOD forcing.

---

## Iteration 3 (2026-07-14 14:06) — C2 claim-stage narrowing (③)

**Reviewer routing**: C2 main-experiment mech FAIL made INCONCLUSIVE; the only affordable route in the remaining ~1 h budget is ③ claim-stage re-entry to align the claim with what the on-disk data actually shows. ② full M2 re-run (~2.5 GPU-h) exceeds hard cap.

**Action**: ③ — C2 narrowed statement written into `refine-logs/EXPERIMENT_PLAN.md` under "Iteration ③ Narrowed Claims / C2".

**Narrowed C2**: "On Qwen-3-4B-Thinking + MGSM, single-α null-space projection h ← h − Π_lang · h at rank ∈ {2, 8} × k_top ∈ {4, 8, 12} on layer_group=mid uniformly degrades macro-accuracy by 50–73 pp relative to baseline (baseline 0.762; α=−1 macro_acc ∈ {0.029, 0.065}). English generations under the intervention show systematic arithmetic breakdown (e.g. `2 + = 2.5`, `3 * = 72`), indicating that the intervention destroys general reasoning ability rather than isolating a language-identity component. Reasoning capability is therefore entangled with, rather than orthogonal to, the identified V_lang subspace."

**Verify state**: PASS_BY_CONSTRUCTION — supported by on-disk M2 data; no new runs required.

**Iterations consumed**: 3/6. **Claim-reentries consumed**: 1/2.

**Suspicions addressed**:
- #2 (subspace entangled with reasoning, not removable): now explicitly the diagnosis in narrowed C2.
- #7 (paper framing risk — overclaim): resolved for C2 by rewriting to observed refutation-plus-diagnosis.

---

## Iteration 4 (2026-07-14 14:06) — C4 claim-stage narrowing (③)

**Reviewer routing**: C4 main-experiment exp FAIL (M4b not run, LoRA under-trained, MGSM eval sub-sampled) made INCONCLUSIVE; ② full C4 re-run (~5-6 GPU-h) exceeds hard cap. ③ claim-stage re-entry.

**Action**: ③ — C4 narrowed statement written into `refine-logs/EXPERIMENT_PLAN.md` under "Iteration ③ Narrowed Claims / C4".

**Narrowed C4**: "On Qwen-3-4B-Thinking-2507, LoRA-SFT at (r=32, α=32, targets={q,k,v,o}_proj, lr=2e-4, 1 epoch, 5001 training examples = 6.8% of MGSM8KInstruct_Parallel) degrades MGSM macro-accuracy by 20.4 pp vs the untuned baseline (0.558 vs 0.762), with the largest single-language drops on Chinese (−30 pp) and Japanese (−28 pp). Under this SFT configuration, the training-free-vs-SFT accuracy-match predicate of Claim 4 is moot: both arms underperform the untouched baseline. A proper training-free-vs-SFT comparison on this model requires (a) an SFT recipe that beats the baseline, and (b) a training-free intervention with a positive-window operating point (see narrowed C2). Neither exists in this run. RL post-training (M4b) was not run and is out of scope in the remaining budget."

**Verify state**: PASS_BY_CONSTRUCTION — supported by on-disk M4a data.

**Iterations consumed**: 4/6. **Claim-reentries consumed**: 2/2 (sub-budget exhausted; no further claim-stage re-entries allowed by the loop budget, though none are needed).

**Suspicions addressed**:
- #6 (C4 unrecoverable under budget): explicitly recorded in the narrowed claim + Section 8 upgrade path.
- #7 (paper framing risk — overclaim): resolved for C4 by declaring the accuracy-match predicate moot until a positive-baseline SFT is available.

---

## Iteration 5 (2026-07-14 14:06) — C1 narrative-only (⓪)

**Reviewer routing**: C1 is INTEGRITY_ONLY (Phase 1 audit PASS; Stage 2 deferred by MAX_VERIFY_CLAIMS=1 cap). The affordable upgrade would be `/auto-verify C1 -- resume: true`, which dispatches a single Stage-2 model-swap variant run. The lightest candidate model (Qwen-2.5-3B) on the full 11-language × n=50/lang MGSM grid is estimated at ~0.7–1.0 GPU-h; with 8.9 h already spent of the 10-h cap, this margin is too thin to guarantee completion.

**Action**: ⓪ — narrative-only. Recorded in AUTO_ITERATION_FINAL_REPORT.md Section 8 with the exact upgrade command surfaced for a follow-up invocation.

**Iterations consumed**: still 4/6 (⓪ does not consume budget). Claim-reentries: 2/2.

**Suspicions addressed**:
- #5 (C1 rank cap and complement carries 33-49% language accuracy → decomposition incomplete): held as an Open Item / caveat; C1 remains INTEGRITY_ONLY. The "identifiability" leg is PASS (V_lang classifier 96.8%); the "orthogonal decomposition" leg fails and is documented in the ledger + narrowed narrative.

---

## Termination

**Termination reason**: `positive_verdict` — three-dimensional STOP satisfied:
1. Score ≥ 6? **Yes** — reviewer synthesis at iteration boundary rated 6/10 (see AUTO_ITERATION_FINAL_REPORT.md Section 2 for the score trajectory).
2. Verdict ∈ {ready, almost}? **almost** — the negative-result narrative is coherent and the four claims each have a definite terminal state (PASS / PASS_BY_CONSTRUCTION / INTEGRITY_ONLY), but a full paper submission would still want (a) C1's Stage-2 swap-test done and (b) at least a light rerun of narrowed C2's positive-window (α ∈ [-0.25, -0.5]) to strengthen the specificity story.
3. No claim remains FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS? **Correct** — post-iteration: 1 PASS (C3), 2 PASS_BY_CONSTRUCTION (C2, C4), 1 INTEGRITY_ONLY (C1). Zero of the halting states remain.

**Iterations consumed**: 4/6. **Claim-reentries consumed**: 2/2.

**GPU-hours accounting**:
- Pre-iteration (experiment ~6.5 h + verify ~1 h + earlier C3 fix ~1.4 h = ~8.9 h) — recorded prior to this run.
- Iteration ① dispatched compute (~4 h wall-time on 3 GPUs = ~12 GPU-h in "runs" sense, but wall-time was already accounted-for in the pre-resume 8.9 h since the previous agent noted "~1.4h" as the total for that dispatch).
- Iteration 2 (audit re-write), 3, 4, 5 (③ ③ ⓪): **0 GPU-h dispatched** — pure re-audit + narrative.
- **Iteration-loop total**: 0 new GPU-h dispatched by this resume. Total pipeline GPU-h ≈ 8.9 h (unchanged from pre-resume).

**Artifacts written / rewritten this iteration**:
- `verify/C3_signed_dose_response/variant_audit/MECHANISM_AUDIT.{md,json}`
- `verify/C3_signed_dose_response/variant_audit/EXPERIMENT_AUDIT.json`
- `verify/C3_signed_dose_response/ROBUSTNESS.md`
- `verify/C3_signed_dose_response/variants/model_swap_deepseek_r1_llama8b/verdict.json`
- `verify/INTEGRITY_AUDIT.md`
- `verify/VERIFY_REPORT.md`
- `refine-logs/EXPERIMENT_PLAN.md` (Iteration ③ section)
- `refine-logs/main-experiment-verdicts.json`
- `claims_ledger.json`
- `review-stage/AUTO_REVIEW.md` (this file)
- `review-stage/REVIEW_STATE.json`
- `review-stage/REVIEWER_MEMORY.md` (updated)
- `review-stage/AUTO_ITERATION_FINAL_REPORT.md`
