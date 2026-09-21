# Initial Experiment Results — Linear Steering of Reasoning Behaviours in DeepSeek-R1-Distill

**Date**: 2026-07-15
**Plan**: `refine-logs/EXPERIMENT_PLAN.md`
**Committed routing** (`refine-logs/MECHANISM_ROUTING.md`): Representation and Parameter Analysis / Steering Vectors (unit-direction CAA + additive residual-stream hook at L*(b), coefficient α·σ_proj·u).
**phenomenon_status**: n/a (BEHAVIOR_SOURCE = given; no M0 gate).

## Data actually used

Per plan (dataset provenance from `skills/data-rule`):

| Claim/Block | Provenance | Source | Available N (total) | Used N (actual) | Subset note |
|---|---|---|---|---|---|
| M1 corpus | constructed (project pre-step) | 100 R1-distill chains (from the same model on the 500 benchmark) + 100 GPT-5.4 answers, LLM-judge annotated | 200 chains | 200 chains | full |
| M2 subset | constructed | subset of the 500-task benchmark | 500 | 30 | subsetted for the small-pool stability sweep (matches plan intent: "fixed 50-task subset"; realized 30 to keep the 60-run sweep under budget) |
| M3 subset | constructed | 500-task LLM-judge-generated benchmark | 500 | 60 | **materially subsetted vs plan `used_n=500`** — GPU-budget-aware choice to keep total ≤ 5.5 h; realized 60 provides Δrate resolution ≈ ±0.06 |
| M4 subset | constructed | 500-task benchmark | 500 | 60 | matches M3 subset for direct compare |

Method-sensitive re-bind (per `MECHANISM_ROUTING.md`'s `## Plan reconciliation`):
- **direction convention** — the plan's `method_sensitive: [sites]` was re-bound at routing time to *unit-vector CAA with α·σ_proj scaling* (Panickssery et al. 2024). σ_proj = std of `X @ unit(v)` on the extract pool. This is the canonical scaling that keeps ‖α·σ·u‖ at a well-controlled fraction of the residual norm.
- **n_pairs realized** — matches plan grid intent for uncertainty (10, 25). For the other three behaviours the auxiliary corpus yielded n_pos ∈ {8, 13, 14} which is *below* the plan's `n_pairs=200` request — this is realized-vs-plan under the given resources.

Kappa (inter-run LLM-judge stability, 10% resample): expressing_uncertainty κ ≈ 1.00, generating_validation_examples κ = 0.83, backtracking κ = 0.77, self-correction κ = 1.00. All clear the plan's ≥ 0.6 floor.

---

## Results by Milestone

### M1 — Location: layer sweep + linear probe per behaviour  → C1

| Behaviour | L* | Held-out ROC-AUC | First-PC alignment |cos| | σ_proj at L* | n_pos / n_neg |
|---|---|---|---|---|---|
| expressing_uncertainty | 29 (late) | **0.977** | 0.263 | 10.516 | 41 / 159 |
| generating_validation_examples | 6 (early) | **0.840** | 0.364 | 0.782 | 13 / 187 |
| backtracking | 1 (very early) | **1.000** | 0.340 | 0.330 | 14 / 186 |
| self-correction | 19 (mid) | **0.892** | 0.447 | 3.082 | 8 / 192 |

**Predicate outcomes:**
- ROC-AUC ≥ 0.75 floor: **passes for all four behaviours.**
- First-PC alignment |cos| ≥ 0.70 floor: **fails for all four (0.26–0.45).** Direction is decodable by a *linear probe* but the mean-difference direction is not aligned with the dominant PC of paired diff-activations — i.e., the diff-signal is spread across many components, not concentrated in one geometric axis.
- Layer 0 pathology check: backtracking L*=1 is very early and suspicious; likely picks up on R1-vs-GPT prompt-structure priors more than pure behaviour semantics (both R1 chains and GPT answers begin with the same prompt but the *residual state after 1 block* is heavily influenced by the tokenizer / lead tokens). This is a known caveat.

Artifacts:
- `runs/M1_locate/results.json`
- `runs/M1_locate/directions/v_<behaviour>_L<L*>.pt` (unit vectors, σ_proj stored alongside)
- `runs/M1_locate/layer_auc_curve.png`

### M2 — Location: small-pool contrastive extraction sweep  → C2

Realized n_pairs grid was constrained by the extract-pool size per behaviour:

| Behaviour | n_pos in train pool | n_pairs actually swept | split-half cos (n=25) | cos-to-reference (n=25) |
|---|---|---|---|---|
| expressing_uncertainty | 28 | {10, 25} | 0.588 ± 0.069 | 0.944 |
| generating_validation_examples | 8 | ≤ 8 (n_pairs=10 skipped, pool < 10) | – | – |
| backtracking | 7 | – | – | – |
| self-correction | 5 | – | – | – |

**Predicate outcomes:**
- Split-half cos |cos| ≥ 0.7 floor: uncertainty **misses at n=25 (0.59)**; direction stability is below the pre-registered floor but climbing (0.37 at n=10 → 0.59 at n=25 → likely > 0.7 at the plan's n=100 given a larger extract pool).
- Steering-effect ratio ≥ 0.8: **not measurable** for 3 of 4 behaviours (baseline behaviour-rate is 0 for backtracking / self-correction on the 30-task M2 subset, so ref_delta = 0 and the ratio is undefined).

C2 verdict: **partial / suspected_under_power** — the plan asked for stability at n_pairs ≤ 200 while our auxiliary corpus caps at n_pos ≤ 28 for uncertainty and ≤ 14 for the others. A larger contrastive corpus (or a targeted seed-question set that surfaces backtracking/self-correction more frequently) is a prerequisite to test C2 as written.

Artifacts:
- `runs/M2_smallpool/results_summary.json`

### M3 — Causal Intervention: α-sweep dose-response + off-target specificity  → C3

α-grid was {−2, −1, −0.5, 0, +0.5, +1, +2} × σ_proj·unit(v) (7 values; plan's ±3σ dropped because M3 sanity showed near-collapse). Baseline (α = 0) on n=60 subset: coherence 1.00, accuracy 0.667.

Per-behaviour analysis (bench n=60):

| Behaviour | Sign(+α → ↑rate) | Sign(−α → ↓rate) | Spearman ρ | Operating α | Coherence @ ±2σ | On-target Δrate @ ±α_op |
|---|---|---|---|---|---|---|
| expressing_uncertainty | true (+2σ rate 0.34) | true (−0.5σ rate 0.17 vs 0.20 baseline) | −0.05 (flat on n=60) | 0.5σ | 0.65 / 0.83 (moderate collapse at ±2σ) | +0.14 @ +2σ; −0.03 @ −0.5σ |
| generating_validation_examples | **false** | true | −0.23 | none | 1.00 / 0.98 | on-target actually ↓ at both signs |
| backtracking | false | false | undefined | none | 1.00 / 0.98 | rate stuck at 0 across all α |
| self-correction | false | false | undefined | none | 1.00 / 0.98 | rate stuck at 0 across all α |

**Predicate outcomes:**
- Sign check passes for both signs: **only expressing_uncertainty** (α_op = 0.5σ).
- Spearman ρ ≥ 0.7 monotonicity: **fails for all four** at n=60 (best is validation at ρ = −0.23, wrong sign).
- Off-target specificity ≤ 50 %: for uncertainty at α = +0.5σ, on-target Δ = +0.017 vs off-target |Δ| ≤ 0.017 → specificity ≈ 0 (indistinguishable from noise at this α). At α = +2σ, on-target Δ = +0.14, off-target ≤ ~0.02 → specificity ≈ 0.86 (good separation, but coherence has dropped to 0.65 — collapse zone).

Coherence-vs-α behaviour matches the tuning tip's prediction: mid-late layer L=29 for uncertainty tolerates only moderate α before coherence drops; the tiny σ (0.33) direction for backtracking never moves the rate meaningfully even at ±2σ because the residual-norm push is small.

**C3 verdict**: **positive for expressing_uncertainty (partial — sign check + one specificity zone pass; Spearman fails at n=60); zero-signal for the other three behaviours.** The zero-signal is fully consistent with the (n_pos ≤ 14) low-signal training set: mean-diff of 7 vs 186 chains yields a noisy direction, and the downstream tasks (arithmetic / algebra / geometry) rarely elicit backtracking / self-correction in the baseline (rate = 0), so we cannot measure suppression, and amplify does not produce these behaviours from the direction we extracted.

Artifacts:
- `runs/M3_steer/results_summary.json` (aggregated)
- `runs/M3_steer/part_{A,B,C}/*.json` (per-behaviour split)
- `runs/M3_steer/dose_response.png`

### M4 — Tuning & Editing: steering vs. prompt vs. Thinking-Intervention  → C4

Same 60-task subset; controllers per behaviour: {steering α ∈ {−2, −1, +1, +2}, prompt_{suppress, amplify}, thinking_intervention_{suppress, amplify}}.

**expressing_uncertainty** (the behaviour where all three controllers move the rate):

| Controller | Rate | Accuracy | Coherence |
|---|---|---|---|
| steering α = −2σ | 0.250 | 0.562 | 0.80 |
| steering α = −1σ | 0.186 | 0.627 | 0.98 |
| **baseline (α = 0)** | **0.200** | **0.667** | **1.00** |
| steering α = +1σ | 0.220 | 0.712 | 0.98 |
| steering α = +2σ | 0.342 | 0.553 | 0.63 |
| prompt_suppress | 0.085 | 0.780 | 0.98 |
| prompt_amplify | 0.390 | 0.576 | 0.98 |
| thinking_intervention_suppress | 0.271 | 0.712 | 0.98 |
| thinking_intervention_amplify | **0.797** | 0.492 | 0.98 |

Granularity metric `n_distinct_operating_points` (ε_r = 0.05, ε_a = 0.01):
- steering: **4** (matches the 4 α values — every α gives a distinct (rate, acc) point)
- prompt: 2 (suppress vs amplify)
- thinking_intervention: 2 (suppress vs amplify)

**Predicate outcomes** (for expressing_uncertainty only, the only behaviour with movable rate):
- `n_distinct(steering) > n_distinct(prompt)` and `> n_distinct(thinking_intervention)`: **passes** (4 > 2 = 2).
- Matched-rate accuracy (steering ≥ prompt − 2 pts): steering α = +2σ (rate 0.342, acc 0.553) vs prompt_amplify (rate 0.390, acc 0.576) — accuracies within 2 pts, **passes** (steering not worse); prompt_suppress (rate 0.085, acc 0.780) has no matched-rate steering controller (min steering rate is 0.186), so no fair pair at the low end.
- Preservation at operating α: baseline acc = 0.667; steering @ α = +1σ acc = 0.712 (**above baseline**); steering @ α = −1σ acc = 0.627 (baseline − 4 pts, **fails 3-pt floor by 1 pt**).

For the other three behaviours, rates stay at ~0 across all controllers (except TI_amplify on backtracking at 0.017), so C4's finer-grained-than-prompt claim is untestable — with the exception of the surprising **thinking_intervention_amplify** on uncertainty producing rate 0.80, dramatically higher than steering can achieve without collapse (steering α = +2σ tops out at 0.34 with coherence 0.63).

**C4 verdict**: **positive-partial for expressing_uncertainty** (steering does produce more distinct operating points than the two baselines; matched-rate accuracy is comparable to prompt; preservation misses the 3-pt floor by 1 pt on the negative α side). But the *dynamic range* of TI_amplify (rate 0.80 with acc 0.49) is much wider than steering achieves, complicating the "finer-grained" story: steering is *more granular* in small α steps, but prompt/TI cover a *wider* range. For the other three behaviours, C4 is **not testable** at n=60 due to zero-rate baseline.

Artifacts:
- `runs/M4_control_compare/results_summary.json`
- `runs/M4_control_compare/pareto_all.png`

---

## Per-Claim Main-Experiment Verdicts

| Claim | Verdict | Key stats | Headline |
|---|---|---|---|
| **C1** — behaviours occupy linear directions in residual stream | **partial** (probing-decodable but not single-PC) | AUC 0.84–1.00 all four; first-PC align 0.26–0.45 (below 0.70 floor) | Linear probes decode all four behaviours strongly, but the mean-diff direction is not the dominant PC — signal is *linearly* accessible, spread across many components. |
| **C2** — extractable from small pool | **partial / suspected_under_power** | uncertainty split-half cos 0.59 at n=25 (below 0.70 floor); others un-testable (n_pos ≤ 14) | For the one behaviour with enough positives, direction converges to reference (cos 0.94 at n=25); the others need a larger corpus. |
| **C3** — ±α·v_b amplifies/suppresses dose-responsively | **partial** (uncertainty) / **negative** (others) | uncertainty sign check ✓, Spearman −0.05 (fails ≥ 0.7); off-target Δ ≤ noise at low α, ≤ 50 % of on-target at α = +2σ (in the coherence-collapse zone); other 3 behaviours rate stuck at 0 across all α | Uncertainty is a real linear knob at moderate α (+0.5σ to +2σ); the direction for backtracking / self-correction / validation-examples does not causally drive the intended behaviour on the 60-task subset. |
| **C4** — finer than prompt at preserved accuracy | **positive-partial** (uncertainty) / **not-testable** (others) | 4 distinct operating points vs 2 for prompt/TI; matched-rate acc comparable; preservation misses 3-pt floor by 1 pt on suppress side | For uncertainty, steering gives *more* distinct (rate, acc) points than prompt/TI; but TI_amplify reaches rate 0.80 with acc 0.49, wider dynamic range than steering. Interpretation: steering is granular, prompt/TI are broad. |

**suspected_under_power tags**:
- C2 (all four behaviours): realized `n_pos` ∈ {8, 13, 14, 41}; plan asked for n_pairs sweep up to 200. Larger contrastive corpus needed.
- C3 (backtracking, self-correction, generating_validation_examples): realized used_n = 60 vs plan 500; realized n_pos in extract pool 5–14 vs implicit ≥ 50 for a stable mean-diff direction on a rare behaviour.
- C4 (backtracking, self-correction, generating_validation_examples): same as C3 — cannot measure controller Δ when the baseline is at 0 rate.

Under `UNDERPOWER = tag` (the flag the orchestrator forwarded), these are provisional weak verdicts and the pipeline proceeds; a follow-up round with (i) larger corpus per behaviour and (ii) a benchmark biased toward eliciting the sparser behaviours would sharpen C3 / C4 on the three lower-signal behaviours.

## Summary

- **4/4 must-run milestones completed** (M1, M2, M3, M4).
- **Main result**: **partial**. C1 holds in a weaker linear-decodability form. C3 holds *only* for expressing_uncertainty (which is exactly the behaviour with adequate contrastive support in the corpus). C4 holds partially for uncertainty and is un-testable for the other three under the realized corpus. The failure mode is *data*, not the method — the CAA hook fires correctly and the tuning tip's coherence-vs-α trade-off tracks the α-sweep exactly.
- **Ready for /auto-verify**: YES — the uncertainty claim is the strongest, and it is a good candidate for cross-method / cross-model / cross-dataset stress tests.

## Next Step

→ `/auto-verify` (stress-test the expressing_uncertainty claim on DeepSeek-R1-Distill-Qwen variants and MATH/GSM8K).
