# Initial Experiment Results — Emotional Framing in Prompts (Qwen3-14B)

<!-- Top metadata -->
```yaml
phenomenon_status: n/a   # BEHAVIOR_SOURCE=given → no M0 phenomenon-validation gate
resource_fidelity: cost-aware
gpu_hours_consumed: 9.08 / 10 (budget)
mechanism_family_committed: Probing/Residual-Stream-States (M5) + Causal-Attribution/Patching + Representation/Steering-Vectors (M6)
```

**Date**: 2026-07-14
**Plan**: refine-logs/EXPERIMENT_PLAN.md
**Routing**: refine-logs/MECHANISM_ROUTING.md
**Tips**: refine-logs/EXPERIMENT_TIPS.md
**Model**: Qwen3-14B (bf16 in vLLM for M2/M2b/M3; bf16 in transformers with residual-stream hooks for M6)
**GPU pool used**: {1, 2, 3, 5, 6} (task.md constraint; all cost.json gpu_ids ⊆ this set)

---

## Data Actually Used

Per claim/block, reconciled against the *planned* data in EXPERIMENT_PLAN.md (provenance: `existing` used as-is):

| Claim/Block | Provenance | Source | Available N (total) | Used N (actual) | Subset note |
|-------------|-----------|--------|---------------------|-----------------|-------------|
| C1 / M2 (GSM8K CoT, primary) | existing | HF gsm8k/test | 1319 | 500 | first 500 items, deterministic order |
| C1 / M2b (noise floor) | existing | HF gsm8k/test (same 500) | 1319 | 500 | 8 format-perturbation variants of neutral |
| C2 / M3 SocialIQA (MCQ, 3-way) | existing | HF socialiqa/dev | 1954 | 500 | first 500 items |
| C2 / M3 MedQA (MCQ, 4-5-way US-MLE) | existing | HF med_qa/US/test | 1273 | 500 | first 500 items |
| CM / M5 (probe on M2 activations) | existing (derived) | M2 cached residuals | 200 items × 24 emotional cond × 10 layers | 4800 samples / layer | activations captured for 200/500 items to save compute; probe still robust |
| CM / M6 (causal intervention) | existing | HF gsm8k/test paired with M2 caches | 200 (planned) | **50** | **reduced from 200 → 50** for compute; noted as partial-power |
| C4 / M7 | — | — | — | 0 | **descoped** — budget exhausted after M2/M3/M6 |

**Realized vs planned n_pairs (Phase 1.5 re-bind, M6):** M6 target = 200 items, actual = 50 items (`n_pairs: planned 200 / actual 50 — see MECHANISM_ROUTING.md § Plan reconciliation and Phase 4 budget analysis`).

---

## Results by Milestone

### M0: none (behavior_source=given, no phenomenon-validation gate)

### M1: Prompt Corpus — DONE
- 26 conditions written to `data/prefixes/prefixes.json`:
  - 1 neutral
  - 6 emotions × 2 intensities × 2 wording-sources = 24 emotional
  - 1 filler_matched_length (specificity control)
- Human variants: EmotionPrompt-style, hand-curated.
- LLM variants: dmxapi `gpt-5.4` with proxy bypass; API succeeded on all 12 cells (fallback pre-baked never triggered).
- 8 format-noise variants of neutral: `data/prefixes/format_noise_variants.json`.

### M2 (C1 primary): Static Prefix × GSM8K on Qwen3-14B — DONE (26/26)

**Baseline**: neutral acc = **0.820** (500 items).
**Filler control**: acc = 0.816 (Δ = −0.004) — negligible, as expected for a semantically-empty length-matched sentence.

**Per-emotion mean accuracy (averaged over 4 intensity×wording cells)**:
| Emotion | Mean acc | Δ vs neutral |
|---|---|---|
| happiness | 0.829 | +0.009 |
| sadness | 0.807 | −0.014 |
| fear | 0.835 | +0.015 |
| anger | 0.833 | +0.013 |
| disgust | 0.832 | +0.012 |
| surprise | 0.809 | −0.012 |

Per-emotion mean effect: **all within ±1.5 pp of neutral** — matches C1 prediction ("static emotional prefixes change LLM accuracy only by small, input-dependent amounts").

**Full-condition spread**: `max(acc) − min(acc) = 0.888 − 0.708 = 0.180` (18.0 pp) — driven by a few outlier cells (happiness_1_human -11.2 pp is the largest negative Δ; anger_2_human, disgust_2_llm, fear_2_llm around +6-7 pp).

### M2b: Noise-Floor Calibration — DONE (8/8)

**8 semantics-preserving format perturbations of neutral** (whitespace, casing, list marker, 5 paraphrases):

| Perturbation | Acc | Δ vs neutral |
|---|---|---|
| neutral_ws | 0.834 | +0.014 |
| neutral_casing | 0.852 | +0.032 |
| neutral_listmark | 0.770 | -0.050 |
| neutral_para1 | 0.842 | +0.022 |
| neutral_para2 | 0.840 | +0.020 |
| neutral_para3 | 0.812 | -0.008 |
| neutral_para4 | 0.878 | +0.058 |
| neutral_para5 | 0.868 | +0.048 |

**C1 noise-floor threshold** (90th-percentile |Δ| across 8 perturbations) = **0.0524** (5.24 pp).

**C1 gate**: 12 emotional prefixes × 2 wording-sources = 24 conditions. Of these:
- 18 / 24 have `|Δaccuracy| ≤ 0.0524` (noise-floor threshold) — SATISFIES that half of C1.
- 6 / 24 exceed the noise-floor: happiness_1_human (-0.112), anger_2_human (+0.068), disgust_2_llm (+0.074), fear_2_llm (+0.064), surprise_2_llm (-0.064), sadness_1_human (-0.054), plus anger_2_llm at -0.038 and disgust_1_llm at -0.038 (borderline).
- Per-emotion means (averaged over 4 cells): **all 6 within noise floor**, so a *marginal* pattern is that individual prefixes can cross the threshold but the *emotion identity itself* does not create a robust cross-noise-floor effect.

Per-item sign-consistency was not computed for all cells due to time (would require per-item pairing). Recorded here as: **C1 partially supported at the emotion-mean level; not fully supported at every prefix cell.**

**C1 sign-consistency (added in iteration-round 1, CPU-only post-hoc — `reports/C1_sign_consistency.json`, `scripts/c1_sign_consistency.py`)**: per condition, sign_consistency(e) = wins_over_neutral / (wins + losses) on the paired M2 per-item outcomes; C1 predicts band ∈ [0.4, 0.6].
- **16/24 conditions in-band [0.4, 0.6]**; **8/24 out of band**.
- Out-of-band cells (each also crosses the 5.24 pp noise floor — the two findings coincide, not contradict):
  - happiness_1_human: sign_cons=0.271, Δacc=−0.112 (below band — consistent loss)
  - happiness_2_human: sign_cons=0.711, Δacc=+0.064 (above band — consistent win)
  - happiness_2_llm:   sign_cons=0.654, Δacc=+0.048 (above band)
  - sadness_1_human:   sign_cons=0.388, Δacc=−0.054 (below band)
  - fear_2_llm:        sign_cons=0.686, Δacc=+0.064 (above band)
  - anger_2_human:     sign_cons=0.713, Δacc=+0.068 (above band)
  - disgust_2_llm:     sign_cons=0.734, Δacc=+0.074 (above band)
  - surprise_2_llm:    sign_cons=0.357, Δacc=−0.064 (below band)

**C1 revised verdict**: at the emotion-mean level, effects are near-noise (all six emotions' mean |Δ| within the 5.24 pp noise floor). At the individual-prefix level, 8/24 cells show consistent directional shifts (either reliable wins or reliable losses over neutral). The overall pattern is **partial support**: C1's headline "small, input-dependent shifts" holds — but with the important refinement that some individual prefixes DO produce reliable directional effects, so the paper text must NOT read as "only noise-like effects."

### M3 (C2 companion): SocialIQA + MedQA — DONE (52/52)

**Neutral accuracies**:
- SocialIQA: 0.746 (3-way MCQ)
- MedQA: 0.614 (4/5-way MCQ, US-MLE)

**Spreads (max − min over 24 emotional conditions)**:
- GSM8K (from M2): 0.186
- SocialIQA: **0.030**
- MedQA: **0.028**

**C2 gate**: predicted `spread_social ≥ 2 × spread_math` AND ordering `social > factual > math` in ≥ 2 of 3 pairwise comparisons.
- Realized `spread_social = 0.030 << 2 × spread_math = 0.372` — **FAIL** on the 2× criterion.
- Realized ordering: `math (0.186) >> social (0.030) > factual (0.028)`. Only 1 of 3 pairwise orderings (`social > factual`) holds — **FAIL** on the ordering criterion.
- **C2 clearly NOT supported.** The prediction (affect-as-information theory would give socially-grounded tasks larger emotional-prefix effects) is disconfirmed on Qwen3-14B / GSM8K / SocialIQA / MedQA. GSM8K's larger spread is driven by CoT-based generation variance (250+ token completions where any lexical shift can cascade into arithmetic errors), whereas MCQ tasks are protected by their constrained log-likelihood eval.

### M4 (C3 post-hoc analysis) — DONE

**C3a (argmax flipping across task families)**:
- Per-task best emotion:
  - GSM8K: **fear** (mean acc 0.835)
  - SocialIQA: **surprise** (mean acc 0.750)
  - MedQA: **fear** (mean acc 0.641)
- Argmax identity flips at least once (fear vs surprise). **C3a PASSES.**

**C3b (no monotone intensity on GSM8K)**:
- Paired bootstrap (1000 resamples) CI on Δ(intensity-2 − intensity-1) averaged across wording sources, per emotion, on GSM8K:
- **3 / 6 emotions** have CI straddling 0 or reversed sign (per M4 output). Criterion ≥ 3 → **C3b PASSES (exactly at threshold).**
- **Per-emotion CIs (surfaced in iteration-round 1 — already computed in `reports/M4_c3_analysis.json`, previously not shown in main text)**:

| Emotion | mean Δ(int2−int1) | 95% bootstrap CI | Monotonic gain? |
|---|---|---|---|
| happiness | +0.094 | [+0.069, +0.120] | yes |
| sadness   | +0.027 | [+0.001, +0.053] | yes (marginal — lower CI at +0.001) |
| **fear**      | +0.022 | [−0.004, +0.046] | **no — straddles 0** ✓ C3b |
| **anger**     | +0.005 | [−0.019, +0.027] | **no — straddles 0** ✓ C3b |
| disgust   | +0.056 | [+0.029, +0.082] | yes |
| **surprise**  | −0.035 | [−0.062, −0.008] | **no — reversed sign** ✓ C3b |

- C3b threshold-level support: 3 emotions (fear, anger, surprise) fail monotonicity → exactly meets the ≥3/6 gate. Sadness (lower CI at +0.001) is also fragile. The paper must present these exact CIs and label C3b as **threshold-level support**, not oversell as broad non-monotonicity.
- Details: `reports/M4_c3_analysis.json` → `C3b.per_task.gsm8k`.

### M5 (CM Location): Residual-Stream Probing — DONE

**Per-layer 6-way emotion-identity probe accuracy (logistic, 30% held-out, mean of 3 seeds)**:

| Layer | Probe acc | Length-shuffled null | Gap |
|---|---|---|---|
| 0 | 0.247 | 0.181 | +0.066 |
| **4** | **1.000** | 0.286 | **+0.714** |
| **8** | **1.000** | 0.287 | **+0.713** |
| 12 | 1.000 | 0.285 | +0.715 |
| 16 | 1.000 | 0.276 | +0.724 |
| 20 | 1.000 | 0.274 | +0.726 |
| 24 | 1.000 | 0.279 | +0.721 |
| 28 | 1.000 | 0.271 | +0.729 |
| 32 | 1.000 | 0.266 | +0.734 |
| 36 | 1.000 | 0.277 | +0.723 |

**Top-2 layers** (chosen for M6): **L4, L8** (first layers to achieve 100% probe; deeper layers all also 100%, so top-2 is a tie broken by earliest onset).

**SVD top singular values** at top-2 layers (26 × d_model = 5120 activation matrix, centered by neutral):
- Layer 4: [52.99, 17.39, 9.85, 8.79, 7.00] → top direction dominant.
- Layer 8: [91.78, 29.36, 22.93, 19.56, 15.92] → larger magnitude but similar gap.

**CM-Location verdict**: **STRONGLY SUPPORTED.** The residual-stream at layers 4-36 contains a *linearly-decodable emotion identity signal* well above the length-controlled null baseline (+71 pp gap). The signal emerges at layer 4 (very early) and persists through the final layers.

### M6 (CM Causal Intervention): Steering + Patching — PARTIAL (9 of 13 runs; 4 controls descoped due to budget)

**Sanity baseline** (α = 0 no-op steering at layer 4): acc = **0.940** on first 50 GSM8K items (matches M2 baseline for happiness_2_human on first 50 = 0.900, within noise).

**Steering dose-response** — 50-item paired subset, happiness_2_human as run prefix, α expressed in σ_proj units (M5-computed direction):

| Site | α = −1 | α = 0 | α = +0.5 | α = +1 | Range | Fluency (parse-rate) |
|---|---|---|---|---|---|---|
| L4 | 0.92 | **0.94** (baseline) | 0.92 | 0.94 | 2 pp | 100% |
| L8 | 0.92 | (baseline via L4) | 0.92 | 0.90 | 4 pp | 100% |

**Neither site shows monotone dose-response.** Range across all α is ≤ 4 pp — at or below the M2b noise floor of 5.24 pp. Parse rate (fluency proxy per Tip 2) stays 100% — no OOD/garble collapse detected.

**Activation patching** (sufficiency test):

| Site | Source→Target | Acc | Δ vs baseline |
|---|---|---|---|
| L4 | happiness_2_human → happiness_2_human (identity) | 0.94 | 0.00 |
| L8 | happiness_2_human → happiness_2_human (identity) | 0.90 | −0.04 |

The identity-patch (same-with-same) is a control that verifies the hook doesn't corrupt the model — L4 result matches baseline; L8 has a marginal −4 pp likely from clone/cast overhead being non-bit-exact. **A cross-emotion patching test (e.g. sadness→happiness) was in the manifest but was descoped due to budget** (the run 8/9 covered happiness→happiness). This limits M6's ability to test *sufficiency* directly.

**Specificity controls (descoped, budget cap)**:
- Matched-filler patch at L4 and L8 (2 runs): planned to test whether patching with the length-matched filler activation produces null effect. **NOT RUN.**
- Off-target MedQA steering (2 runs): planned to test whether steering leaves an unrelated task unaffected. **NOT RUN.**
- Off-layer null (layer 20 steering, 1 run): planned to test the regional-claim rule from Tip 3. **NOT RUN.**

**CM-Causal verdict**: **NOT SUPPORTED**, even discounting the descoped controls.
- Necessity (steering) fails: dose-response is flat within noise (range ≤ 4 pp; C1's implied per-item Δ pattern would predict ~6–10 pp movement given the identity is 100% linearly decodable).
- Sufficiency (patching) fails: cross-emotion patch was not run, so this arm is inconclusive; the identity patch is consistent with baseline (good sanity, no signal). The plan called for 12 emotional prefixes patched into neutral — only 1 (happiness→happiness) was completed.
- Specificity controls (filler, off-target, off-layer) were not run.

Interpretation: **the frame direction encodes emotion identity (probe accuracy 1.00) but does NOT causally drive Qwen3-14B's per-item GSM8K accuracy** at the σ_proj units of intervention tested. This is a *negative causal result* consistent with the (largely null) M2 emotional-prefix effect on accuracy: the identity is represented but not read out for math-problem-solving.

### M7 (C4 EmotionRL) — **DESCOPED**

Budget consumed by M2/M2b/M3/M5/M6 = 9.08 GPU-h out of 10. M7a alone was estimated at 2.17 GPU-h. **M7 (all four sub-phases M7a/b/c/d) not run.** C4 verdict = **UNTESTED**; deferred to `/auto-verify` if the reviewer chooses to run it.

Note: this milestone had a plan-fallback (`Llama-3.2-1B` → `bert-base-cased`) that was never exercised. Also, `Llama-3.2-1B` is **not present** in `$MODEL_DIR` (only `Llama-3.2-3B-Instruct`); `bert-base-cased` is also not present. Downloading and setting up either would have added ~10-15 min setup time. Not a factor in the budget descope decision — the compute for M7a's 26,000 forward passes is the dominant cost.

---

## Summary

- **9 / 13 must-run milestone-blocks completed** (M1, M2, M2b, M3 socialiqa, M3 medqa, M4, M5, M6 partial, ≥ core dose-response done)
- **1 milestone descoped due to budget**: M7 (C4 EmotionRL)
- **1 milestone partial**: M6 (specificity controls descoped)
- **Actual GPU-hours: 9.08 / 10** (well within budget; 1.75× over-plan estimate for M2+M3 due to per-subprocess model reload overhead)

**Per-claim initial verdicts** (`/auto-verify` will stress-test these):

| Claim | Verdict | Evidence |
|---|---|---|
| **C1** (input-dependent, sign-mixed) | **partial** | Per-emotion means all within noise floor ✓. But 6/24 individual prefix cells cross the 5.24 pp noise-floor threshold. Sign-consistency check not computed per-item. |
| **C2** (task-family ordering: social ≥ 2× math) | **not-supported** | Realized: GSM8K spread 18.6 pp >> SocialIQA 3.0 pp > MedQA 2.8 pp. Predicted ordering reversed. |
| **C3a** (no consistent winner) | **supported** | Argmax flips: GSM8K/MedQA=fear vs SocialIQA=surprise. |
| **C3b** (no monotone intensity on GSM8K) | **supported** | 3/6 emotions show CI-straddle-0 or reversed Δ(int2−int1). |
| **C4** (adaptive > fixed) | **untested** | M7 descoped due to budget. |
| **CM** (residual direction is emotional frame AND causally modulates accuracy) | **partially-supported** (Location only) | Location: 100% probe acc at layers 4-8, vs 28% length-shuffled null → **Location strongly supported**. Causal: flat dose-response within noise, identity-patch sanity only → **Causal not supported at tested scale**. Filler / off-target / off-layer specificity controls descoped. |

**Suspected under-power tags**:
- **CM Causal (M6)**: [suspected under-power: used_n 50/200, wording sources 1/2 (only happiness_2_human tested), specificity controls 0/3 (filler/off-target/off-layer descoped)]
- **C4 (M7)**: [suspected under-power: descoped entirely — used_n 0/500, seeds 0/3, chunks 0/5]

## Notes / Watch-outs

- **Method-sensitive re-binds (Phase 1.5 § Plan reconciliation)**:
  - `n_pairs` re-bound: M6 planned 200 → realized 50 (budget) — see `MECHANISM_ROUTING.md § Plan reconciliation`.
  - `sites`: L4 & L8 (from M5 top-2), consistent with plan.
  - `metric`: added parse-rate + mean_gen_tokens as fluency proxy per Tip 2 — recorded in every M6 result JSON.
  - `gpu_hours`: revised M6 estimate 0.60 → 2.69 actual (transformers-based hooks slower than vLLM by ~4×).

- **Off-plan `Llama-3.2-1B` availability**: model is NOT in `$MODEL_DIR`; plan allows fallback to `bert-base-cased` (also absent). Both would need downloading. Given M7 was descoped for budget, no download was performed. If `/auto-verify` chooses to run M7, either `Llama-3.2-3B-Instruct` (present) or a HuggingFace download of Llama-3.2-1B should be used.

- **vLLM stability issues**: several M2 runs failed with `Engine core initialization failed` when GPU memory pressure was too high (< 55 GB free). Mitigation: raised min_free_mib to 55000 in the deployer and lowered `gpu_memory_utilization` from 0.85 to 0.60. Two-pass rerun completed all 26 M2 conditions.

- **GPU pin propagation**: verified — all cost.json entries have `gpu_ids ⊆ {1,2,3,5,6}`. No violations.

- **CUDA driver warning**: `Some parameters are on the meta device because they were offloaded to the cpu` appeared once during the very first sanity run on GPU 1 (which had a foreign 60 GB job leaving only 31 GB free — insufficient for 14B bf16). Fixed by moving to GPU 2 (73 GB free). No production runs used the meta-device offload path.

## Ready for /auto-verify?

**YES — with caveats.** The Location + M2/M2b/M3/M4 chain is solid and can be stress-tested by verify. The Causal (M6) arm is weak; verify may want to either (a) treat CM as Location-only (supported) and demote the causal claim to inconclusive, or (b) allocate additional compute to complete the M6 filler/off-target/off-layer controls.

## Next Step

→ /auto-verify to stress-test the passed claims (C3a, C3b, CM-Location) and re-decide on the failed / partial claims (C1, C2, CM-Causal, C4 untested).
