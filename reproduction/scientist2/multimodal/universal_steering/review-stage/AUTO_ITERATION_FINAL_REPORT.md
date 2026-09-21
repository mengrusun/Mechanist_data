# Auto Iteration Final Report — RFM Concept-Vector Steering & Monitoring on Llama-3.1-8B-Instruct

- **Generated**: 2026-07-15T02:45:00 CST
- **Iterations consumed**: 2 / 6
- **Claim-reentries consumed**: 1 / 2
- **Final reviewer score**: 6 / 10 (from 3/10 at entry; +3 across 3 iterations)
- **Final canonical verdict**: **almost**
- **Termination reason**: **positive_verdict** (three-dimensional STOP rule fired at iteration 3)
- **Cumulative iteration cost**: runs_total=2, gpu_hours_total=0.809 (out of ~3 remaining budget; ~2.2 h unused)
- **Source audit trail**: [`AUTO_REVIEW.md`](./AUTO_REVIEW.md), [`REVIEWER_MEMORY.md`](./REVIEWER_MEMORY.md), [`REVIEW_STATE.json`](./REVIEW_STATE.json)

---

## Executive Summary

The loop entered with 1 FAIL (C5, robustness 0.0 after a single DeepSeek variant failed ToxicChat) and 4 INTEGRITY_ONLY (C1–C4, Stage 2 capped by `MAX_VERIFY_CLAIMS=1`). Iteration 1 added two cheap RLHF-tuned model swap variants (Meta-Llama-3-8B-Instruct + Mistral-7B-Instruct-v0.2) in parallel on GPUs {0,1} and {2,3}; both passed ToxicChat > 0.882, lifting C5 robustness to 2/3 = 0.667 → PASS. Iteration 2 enacted a formal lightweight in-loop claim rewrite `C5 → C5_v2` (per-benchmark-scoped, RLHF-narrowed for ToxicChat with DeepSeek as documented boundary condition) plus narrative demotion of C1-C4 to exploratory / preliminary sub-findings in `refine-logs/FINAL_PROPOSAL.md § Claims`. Iteration 3 was reviewer confirmation — score jumped from 4 to 6 (verdict `almost`), STOP fired. Loop terminates positively with 4/6 iteration budget and 1/2 claim-reentry sub-budget remaining unused.

### Claim Disposition Overview

| Original state | # Claims | Final status after iteration |
|---|---|---|
| PASS                     | 0 | — |
| FAIL                     | 1 | 1 PASS (variant fix ①) → rewritten to `C5_v2` (③); final: **PASS** |
| INCONCLUSIVE             | 0 | — |
| ZERO_ELIGIBLE_VARIANTS   | 0 | — |
| INTEGRITY_ONLY           | 4 | 4 demoted to exploratory / preliminary case studies (⓪ narrative); Stage 2 stress-tests still policy-skipped (standalone `/auto-verify — resume: true` available post-loop) |
| DEFERRED (legacy)        | 0 | — |

---

## Section 1 — PASS Claims (brief audit)

None at entry. C5_v2 is the *outcome* of the loop, not the input; it is documented in Section 2.

---

## Section 2 — FAIL Claims (full journey)

### 2.1 `C5` (→ `C5_v2` post-rewrite) — Internal-feature monitoring beats GPT-4o judge

**Original FAIL signal**
- Original claim: "RFM-based / linear-probe internal-feature monitors on Llama-3.1-8B-Instruct beat GPT-4o-2024-11-20 in AUROC on HaluEval-General and ToxicChat"
- robustness=0.0 (threshold=0.5)
- Single variant tested (`model-swap-deepseek-r1-llama8b`) — variant_claim_supported=False (HaluEval 0.985 beats, ToxicChat 0.858 fails GPT-4o 0.882)
- Variant integrity at entry: PASS (Phase 9 clean — the failure is real, not a script bug)

**Iteration journey**

| Iter | Type | Reviewer flag | Action | Outcome |
|---|---|---|---|---|
| 1 | ① variant_fix | "The DeepSeek-R1 failure on ToxicChat is a real boundary condition; ①-add RLHF variants to establish the empirical warrant" | Created 2 new variants (Meta-Llama-3-8B-Instruct on GPU 0,1; Mistral-7B-Instruct-v0.2 on GPU 2,3). Only diff vs DeepSeek variant is `MODEL_PATH` string. Ran in parallel; ~12 min each. | **Both PASSED both benchmarks strictly.** Llama-3: HaluEval 0.989, ToxicChat 0.933. Mistral: HaluEval 0.989, ToxicChat 0.906. C5 robustness went from 0.0 (0/1) → 0.667 (2/3) → **flipped FAIL → PASS**. |
| 2 | ③ claim_reentry (lightweight, in-loop) | "The empirical evidence supports a narrowed claim; a formal ③ rewrite is required — narrative-only caveat is insufficient" | Rewrote C5 → `C5_v2` in `refine-logs/FINAL_PROPOSAL.md § Claims`. New wording is per-benchmark: HaluEval broad across all 4 tested 8B models (RLHF + reasoning-distilled), ToxicChat RLHF-only with DeepSeek documented as boundary condition. No new experiments (evidence already on disk). | Reviewer in iteration 3 confirmed the rewrite is "genuinely responsive" — score jumped from 4 to 6, verdict from `not ready` to `almost`. STOP rule fired. |

**Path taken (summary)**: variant-integrity fix (①) → claim-stage re-entry lightweight rewrite (③) — claim-reentry sub-budget used: 1/2.

**Experiment & script modifications** (cumulative across all iterations on this claim)

| Iter | Component | Before | After |
|---|---|---|---|
| 1 | `verify/C5_internal_monitor_beats_gpt/variants/model-swap-llama3-8b-instruct/c5_variant_llama3.py:35` (new file, cloned from DeepSeek variant) | (n/a — new file) | `MODEL_PATH = "/data/zhenqian/models/Meta-Llama-3-8B-Instruct/Meta-Llama-3-8B-Instruct"` |
| 1 | `verify/C5_internal_monitor_beats_gpt/variants/model-swap-mistral-7b-instruct/c5_variant_mistral.py:35` (new file, cloned from DeepSeek variant) | (n/a — new file) | `MODEL_PATH = "/data/zhenqian/models/Mistral-7B-Instruct-v0.2"` |
| 1 | `verify/C5_internal_monitor_beats_gpt/ROBUSTNESS.md` | Verdict: FAIL. Robustness=0.00 (0/1 eligible variant). | Verdict: PASS. Robustness=0.667 (2/3 eligible variants). Full updated per-variant table + interpretation. |
| 1 | `verify/VERIFY_REPORT.md § C5 subsection + summary table` | C5 FAIL 0.00; counts 0 PASS, 1 FAIL, 4 INTEGRITY_ONLY | C5 PASS 0.667; counts 1 PASS, 0 FAIL, 4 INTEGRITY_ONLY (post-iter-1 update annotation added) |
| 2 | `refine-logs/FINAL_PROPOSAL.md § Claims` | Single-sentence C5 wording ("beat GPT-4o … on HaluEval-General and ToxicChat") | Per-benchmark-scoped `C5_v2` block; HaluEval broad across 4 tested 8B models; ToxicChat RLHF-only with DeepSeek-R1-Distill negative-case boundary condition explicit |

**Claim modifications** (type ③ rewrite)
- Original claim id `C5`: "RFM-based / linear-probe internal-feature monitors on Llama-3.1-8B-Instruct beat GPT-4o-2024-11-20 (used as a black-box output judge) in AUROC on HaluEval-General (hallucination) and ToxicChat (toxicity), with ToxicChat-T5-Large as the additional toxicity baseline."
- After iteration 2 → new claim id `C5_v2`: per-benchmark scoped:
  - **HaluEval**: internal monitors from 4 tested 8B instruct-LLMs (RLHF *and* reasoning-distilled) beat GPT-4o on HaluEval-General.
  - **ToxicChat**: internal monitors from 3 tested *RLHF-tuned* 8B instruct-LLMs beat GPT-4o on ToxicChat; the tested reasoning-distilled non-RLHF model (DeepSeek-R1-Distill-Llama-8B) does NOT beat GPT-4o on ToxicChat (0.858 vs 0.882). The toxicity advantage is contingent on RLHF-style harmlessness fine-tuning.
- Scope change: removed cross-regime claim on toxicity; added explicit training-regime dependence and per-benchmark scope; removed "architecture-general" language; named all 4 tested models and their training regimes.

**Final experiment summary**
- New runs cited: `runs/iteration_round_1/model-swap-llama3-8b-instruct/`, `runs/iteration_round_1/model-swap-mistral-7b-instruct/`
- Final robustness (broad frame): **0.667** (2/3 eligible variants pass; N_eligible=3, N_pass=2)
- Final robustness (RLHF-narrowed frame from `C5_v2`): **1.00** (2/2 non-baseline RLHF variants pass; plus main experiment on Llama-3.1 makes 3/3 RLHF total)
- Final variant pass rate: 2/3 broad, 2/2 narrowed
- Final status: **PASS (under new claim `C5_v2`)**

**Reviewer memory thread** (this claim's suspicions across iterations)
- Iteration 1: "C5 likely depends on RLHF-specific harmlessness representations, not general latent features. DeepSeek failure on ToxicChat probably marks a real boundary condition."
  - **Resolved iteration 1**: YES (empirically confirmed via 2 additional RLHF-tuned model swaps; both pass ToxicChat > 0.882)
- Iteration 2: "Paper wording must formally narrow, not just add a footnote; robustness bookkeeping ≠ scientific validity."
  - **Resolved iteration 2**: YES (formal ③ rewrite of C5 → C5_v2 in FINAL_PROPOSAL.md; per-benchmark scoped; boundary condition explicit in the claim wording itself)
- Iteration 3: "'Robustly across at least 4 tested 8B models' still has a slightly salesy tone; scrub residual promotional language manuscript-wide."
  - **Resolved**: NO (out of scope — manuscript-wide wording harmonization is a paper-writing task documented as an Open Item; reviewer confirmed this is post-loop work, not blocking for READY)

---

## Section 3 — INCONCLUSIVE Claims

None at entry.

---

## Section 4 — ZERO_ELIGIBLE_VARIANTS Claims

None at entry.

---

## Section 4b — INTEGRITY_ONLY Claims (Stage 2 policy-skipped, demoted to exploratory)

Four claims (C1, C2, C3, C4) were admitted at Phase 2 with WARN integrity but were not selected by Stage 2 due to `MAX_VERIFY_CLAIMS=1` cap (C5 was the top-1 pick). Per the loop's routing contract, no back-edge action was allowed on these claims. However, the iteration-2 reviewer required their **demotion from headline contribution status** as a condition for READY, so `FINAL_PROPOSAL.md § Claims` was restructured (type ⓪ narrative-only, non-budget-consuming) to move C1-C4 out of the headline block into an "Exploratory / preliminary sub-findings" section with explicit failure-mode framing.

### 4b.1 `C1_exp` (demoted from `C1`) — Per-block RFM concept-vector steering (baseline)

**Original wording**: "Per-block RFM concept vectors steer Llama-3.1-8B-Instruct toward or away from anti-refusal / political-stance / honesty concepts, beating unsteered baseline and matched-random-direction control on ≥50 held-out prompts per concept judged by GPT-4o-2024-11-20."

**Audit warnings (from Phase 2)**:
- Mechanism WARN: refusal Location screen ties at 1.0 across all 32 blocks → auto-argmax picked block 0 (degenerate); pinned block-14 re-run performed but underlying probe issue not resolved.
- Mechanism WARN: alpha chosen on same 50-prompt held-out set (no separate dev/test split); alpha-overfitting risk.
- Experiment WARN: 3-scenario bundling is an overclaim (only political shows clean support; refusal null; honesty within judge noise).
- Experiment WARN: alpha_star labelling bug in c1_steer_and_judge.py picks argmax(|Δ|) not signed direction (cosmetic — reported numbers use raw per-alpha aggregates).

**Post-iteration-2 framing** (in `FINAL_PROPOSAL.md § Claims > Exploratory / preliminary sub-findings`):
> `C1_exp` (exploratory) — political sub-concept demonstrates RFM extracts semantically meaningful directions for concepts that are well-attested in the training data and not defended by safety fine-tuning (α=-3→3.08, α=+3→4.02, range 0.94; random control 0.30/0.12). Refusal and honesty documented as null / non-diagnostic case studies (refusal: degenerate probe; honesty: within judge noise).

**Upgrade available** (post-loop): `/auto-verify C1 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run for this claim)

**Reviewer memory thread**:
- Iteration 1: "C1 refusal pipeline may be fundamentally compromised by degenerate location-screen."
  - Resolved: **NO** (out of loop scope). Iteration 2 demotion at least labels this failure mode honestly.
- Iteration 2: "C1 must be demoted from headline contribution status."
  - Resolved: **YES** (iteration 2 narrative demotion; reviewer in iteration 3 confirmed "adequate contingent on consistent manuscript tone").

---

### 4b.2 `C2_exp` (demoted from `C2`) — Python → C++ high-precision-task steering

**Original wording**: "An RFM C++ concept vector added to residuals raises HackerRank test-case pass rate over both default (Python) and prompt-only ('Answer in C++.') baselines on a 50-problem subset."

**Audit warnings (from Phase 2)**: Experiment WARN: `suspected_under_power=true` (n=10 held-out × 2 seeds vs plan's 30); statistical power insufficient. HackerRank `eval_set.jsonl` only has 20 problems available locally.

**Empirical outcome (main experiment)**: default Python pass_rate=0.60, prompt-only C++ 0.43, steered α=+3 = 0.567 (all outputs Python — `cpp_frac=0.00`; vector did NOT switch language).

**Post-iteration-2 framing**: `C2_exp` (exploratory / negative) — vector at α=+3 did not switch language; pass rate BELOW default Python. Reads as a **documented negative finding** — vector encodes training-data language-tag prefix, not intrinsic C++. Under-power flagged.

**Upgrade available** (post-loop): `/auto-verify C2 — resume: true`

**Reviewer memory thread**:
- Iteration 1: "C2 C++ concept vector may not represent language choice at all."
  - Resolved: **NO** (out of loop scope; iteration 2 demotion labels it as negative finding honestly)
- Iteration 2: "C2 must be demoted."
  - Resolved: **YES** (iteration 2 narrative demotion)

---

### 4b.3 `C3_exp` (demoted from `C3`) — Cross-lingual transferability of concept vectors

**Original wording**: "A C1 vector extracted on English paired data steers responses when the prompt is asked in ZH/FR/ES; the multilingual GPT-4o judge confirms the steering direction is preserved."

**Audit warnings (from Phase 2)**:
- Experiment WARN: no multiple-comparisons correction across 4 languages.
- Experiment WARN: FR sign reversal (−0.10) breaks "transfers to ZH/FR/ES" predicate — claim as stated is not met.

**Empirical outcome**: EN +0.20 (p=0.23), ZH +0.32 (p=0.11), FR −0.10 (p=0.67, sign reversed!), ES +0.20 (p=0.27). No p<0.05 at n=50/lang.

**Post-iteration-2 framing**: `C3_exp` (exploratory / partial) — positive shift in 3 of 4 languages; FR reverses sign; no p<0.05 at n=50/lang; no Bonferroni. Underlying honesty vector is intrinsically weak (Δ=+0.14 monolingually on EN). Framed as partial-transfer case study with FR as boundary condition, NOT as a "transfers to ZH/FR/ES" claim.

**Upgrade available** (post-loop): `/auto-verify C3 — resume: true`

**Reviewer memory thread**:
- Iteration 1: "C3 cross-lingual transfer fragile / language-specific; French reversal not random noise."
  - Resolved: **NO** (out of loop scope; iteration 2 demotion labels it honestly)
- Iteration 2: "C3 must be demoted."
  - Resolved: **YES** (iteration 2 narrative demotion)

---

### 4b.4 `C4_exp` (demoted from `C4`) — Compositionality of concept vectors

**Original wording**: "Two-vector linear combinations produce simultaneous multi-concept effects that either single vector alone does not (measured by GPT-4o rubric on both concept targets separately)."

**Audit warnings (from Phase 2)**:
- Experiment WARN: single-vector rubric ceiling (~5.0) on tested prompts masks any compositional gain → predicate uninformative here.
- Mechanism WARN: missing matched-random-direction control on combos (present on C1).

**Empirical outcome**: both tested combos (honesty + refusal-neg; formal_tone + technical_persona) hit rubric ceiling ~5.0 on primary metric for single vectors alone; sum has no headroom. Compositionality untested here.

**Post-iteration-2 framing**: `C4_exp` (exploratory / non-diagnostic) — experiment as designed cannot test compositionality due to ceiling saturation on the tested prompts. Missing matched-random-combination control. Compositionality untested here, NOT falsified. Re-testing on non-saturating prompt sets with random-combo controls is left to future work.

**Upgrade available** (post-loop): `/auto-verify C4 — resume: true`

**Reviewer memory thread**:
- Iteration 1: "C4 non-diagnostic due to single-vector ceiling; missing random-direction control on combos."
  - Resolved: **NO** (out of loop scope; iteration 2 demotion labels the design flaw honestly)
- Iteration 2: "C4 must be demoted."
  - Resolved: **YES** (iteration 2 narrative demotion)

---

## Section 5 — Legacy DEFERRED Claims

Empty under current architecture (new verify runs never populate this bucket).

---

## Section 6 — Cross-Cutting Patterns

The reviewer flagged several systemic patterns across the three iterations (deduped from `REVIEWER_MEMORY.md § Patterns`):

- **Scope inflation** — multiple claims worded broader than the underlying evidence supports. Iteration 1 identified this on C1 (refusal + political + honesty bundle), C3 (all 3 target languages), C4 (compositionality without ceiling-free test bed). Iteration 2 required demotion of C1-C4 from headline status; iteration 3 confirmed the demotion is adequate.
- **Missing controls / methodology gaps** — alpha selection on same held-out set (C1), no matched-random-direction control on combos (C4), no multiple-comparison correction across languages (C3), suspected under-power (C2). These are legitimate audit-level concerns and remain documented in each claim's INTEGRITY_ONLY warning list. The demotion framing explicitly names each threat rather than glossing over it.
- **Process compliance as proxy for scientific adequacy** — the reviewer explicitly pushed back on the framing "INTEGRITY_ONLY doesn't block READY per contract." The loop's routing contract is procedurally correct, but the reviewer insisted C1-C4's substantive validity threats must be either (a) remediated with standalone `/auto-verify` outside the loop or (b) demoted narratively. The loop enacted (b) in iteration 2, and iteration 3 accepted it contingent on consistent manuscript tone.
- **Author willing to add targeted variants when challenged** (positive pattern) — iteration 1 executed the reviewer's exact recommendation (add RLHF variants) rather than defending the original broad claim. Iteration 2 executed the reviewer's rewrite recommendation without pushback.
- **Residual rhetorical inflation risk** — iteration 3 flagged one remaining concern: words like "robustly" in C5_v2 still have a slightly salesy tone. This is not blocking; a post-loop manuscript-wide wording pass would tighten it further. Listed in Open Items.

---

## Section 7 — Iteration Budget & Pipeline

- **Iterations consumed**: **2 / 6** (4 remaining; loop terminated early on positive verdict at iteration 3, which was reviewer-only and did not consume budget)
- **Claim-reentries consumed**: **1 / 2** (1 remaining; used for C5's lightweight in-loop rewrite in iteration 2)
- **Iteration `/run-experiment` calls**: `runs_total = 2` (both in iteration 1, both cheap variant re-runs on reused test splits)
- **Iteration GPU-hours**: `gpu_hours_total = 0.809` (of ~3 h remaining budget; **~2.2 h unused**)

### Per-iteration breakdown (from `iteration_breakdown[]`)

| Iter | Type | Target claims | Produced claims | Runs | GPU-hours | Score after | Verdict after |
|---|---|---|---|---|---|---|---|
| 1 | ① variant_fix | C5 | — | 2 | 0.809 | (score assessed in iter 2 prompt = 4/10) | (verdict in iter 2 = not ready) |
| 2 | ③ claim_reentry (lightweight in-loop) | C5 | C5_v2 | 0 | 0.000 | 6 | almost |
| 3* | (reviewer-only, no back-edge, no counter increment) | — | — | 0 | 0.000 | 6 | almost (STOP) |

\* Iteration 3 is a re-review-only iteration that confirmed iteration 2's rewrite. Per the Phase C contract, the iteration counter is incremented only at the end of Phase C when a real ①/②/③ action ran; iteration 3 had none, so `iterations_consumed` stays at 2.

---

## Section 8 — Open Items for Human Reviewer

> Items the loop could not close within scope. These need human / post-loop follow-up.

- **Still-FAIL claims** (after exhausting routing options): **none**
- **Still-INCONCLUSIVE claims**: **none**
- **Still-ZERO_ELIGIBLE_VARIANTS claims**: **none**
- **INTEGRITY_ONLY claims (Stage 2 skipped — not stress-tested)** — all four remain in this state at termination; policy-skipped by `MAX_VERIFY_CLAIMS=1` cap (C5 was the top-1 pick). Each is now demoted to exploratory in `FINAL_PROPOSAL.md § Claims > Exploratory / preliminary sub-findings`, but the stress-tests were never run. Upgrade path for each (post-loop):
  - **C1_exp** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C1 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME; only Stages 2–3 run). main-experiment integrity: warn; warn_source: experiment+mechanism.
  - **C2_exp** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C2 — resume: true`. main-experiment integrity: warn; warn_source: experiment.
  - **C3_exp** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C3 — resume: true`. main-experiment integrity: warn; warn_source: experiment.
  - **C4_exp** [stage2_skip_reason: max_verify_claims_cap]: `/auto-verify C4 — resume: true`. main-experiment integrity: warn; warn_source: experiment+mechanism.
- **Legacy deferred claims (empty in new runs)**: none
- **Recurring unresolved patterns**: **manuscript-wide wording harmonization** — reviewer in iteration 3 flagged that "robustly," "general," "architecture-agnostic," and "small-model beats stronger judges" language may leak from FINAL_PROPOSAL.md § Claims into the abstract / intro / discussion / conclusion of the eventual paper draft. The loop cannot enact this pass (no paper draft exists yet); listed here so it lands in AUTO_PIPELINE_REPORT.md for the paper-writing stage. Suggested target sections: abstract, introduction contributions bullets, discussion section, conclusion. The safe replacement is benchmark-and-regime-scoped phrasing (see iteration-3 reviewer response for a concrete example paragraph).
- **Claim-reentry refusals** (where reviewer requested ③ but sub-budget was exhausted): **none** (only 1 of 2 claim-reentries used; iteration 3's reviewer explicitly said "I do not think claim re-entry on C1-C4 is needed" and did not request a second ③)

---

## Additional artifacts produced by this loop

- `verify/C5_internal_monitor_beats_gpt/variants/model-swap-llama3-8b-instruct/{c5_variant_llama3.py, config.yaml, run.sh, results/summary.json, results/*_aurocs.npy}` — new Llama-3-8B RLHF variant
- `verify/C5_internal_monitor_beats_gpt/variants/model-swap-mistral-7b-instruct/{c5_variant_mistral.py, config.yaml, run.sh, results/summary.json, results/*_aurocs.npy}` — new Mistral-7B RLHF variant
- `verify/C5_internal_monitor_beats_gpt/ROBUSTNESS.md` — updated with post-iteration-1 3-variant table + interpretation
- `verify/VERIFY_REPORT.md § C5 subsection + summary table` — updated with post-iteration-1 status
- `refine-logs/FINAL_PROPOSAL.md § Claims` — restructured with headline `C5_v2` + demoted `C1_exp/C2_exp/C3_exp/C4_exp`
- `runs/iteration_round_1/{model-swap-llama3-8b-instruct, model-swap-mistral-7b-instruct}/{cost.json, run.log, pid.txt}` — cost + log tracking for iteration-1 variant runs
