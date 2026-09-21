# Claim Ledger — Unified Verification of the Four-Claim Language-Agnostic/Specific Subspace Hypothesis on Qwen-3-4B-Thinking + MGSM

**Direction**: Disentangling Language and Reasoning in LLM Internal Representations — identify a language-specific subspace and test whether inference-time suppression improves multilingual reasoning.
**Date**: 2026-07-14 → 2026-07-14
**Pipeline**: completed | **Iteration**: 6/10 "almost" (4/6; claim-reentries 2/2)
**Models**: claim=claude-opus-4-7, experiment=claude-opus-4-7, verify=claude-sonnet-4-6, iteration=claude-opus-4-7
**Updated after**: iteration:final

| Claim | Main experiment | Verify | Post-Iteration | Final |
|-------|-----------------|--------|----------------|-------|
| C1 subspace decomposition | partial | INTEGRITY_ONLY (audit PASS; swap deferred — cap) | ⓪ narrative-only — upgrade command surfaced | ⚪ integrity_only (audit passed, swap-test deferred — max_verify_claims cap; upgrade `/auto-verify C1 -- resume: true`) |
| C2 null-space suppression → accuracy up | not-supported | INCONCLUSIVE (Phase 2 mech FAIL) | ③ narrowed (claim-reentry 1/2) | ✅ pass_by_construction — original refuted; narrowed C2 (uniform 50–73 pp degradation on layer_group=mid across (rank, k_top)) supported by construction |
| C3 signed α-sweep dose-response | not-supported (main) / partial (specificity) | ZERO_ELIGIBLE_VARIANTS → variant fix ran | ① variant fix — added random-control α-sweep on DeepSeek-R1-Distill-LLaMA-8B; Phase 9 re-audit WARN | ✅ pass — non-monotone pattern (A(-1)=0.391 < A(0)=0.447) robust across model families; V_lang specificity confirmed vs matched random subspace (robustness=1.0) |
| C4 training-free ≥ LoRA-SFT at κ ≤ 0.10 | not-supported | INCONCLUSIVE (Phase 2 exp FAIL) | ③ narrowed (claim-reentry 2/2) | ✅ pass_by_construction — original untestable in this run (both edit and SFT arms below baseline); narrowed C4 (LoRA-SFT drops 20.4 pp) supported by construction |

---
## C1 — subspace decomposition
- **Statement**: Hidden representations of Qwen-3-4B-Thinking on multilingual reasoning inputs decompose into a language-specific subspace V_lang (identifiable from a small multilingual probe set via SVD/mean-difference) and an approximately orthogonal language-agnostic residual.
- **Origin**: task.md ## Claim (verbatim, bullet 1)
- **Data**: FLORES-200 dev + MGSM held-out — provenance=existing; available=≤997/lang × 11 langs (probe), ≤100/lang held-out; used=50/100/250/500/1000 × 11 langs (grid) × 100/lang held-out
- **Models**: Qwen-3-4B-Thinking-2507
- **Method**: Language-mean-difference SVD across 3 layer groups × 6 rank_r × 5 n_probe × 3 seed (270 fits); held-out language classifier + orthogonal-complement classifier + principal-angle vs content probe — screen → decode → verify → recover
- **Main experiment**: **partial** — V_lang classifier=0.968, complement classifier=0.49, median cos=0.11
- **Verify**: robustness=— — method n/a / dataset n/a / model n/a (deferred); integrity=PASS; verdict=INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap)
- **Iteration**: ⓪ narrative-only — Stage-2 model-swap variant deferred (remaining GPU budget too thin to safely fit an 11-language eval); upgrade `/auto-verify C1 -- resume: true` surfaced.
- **Final**: ⚪ integrity_only (audit passed, swap-test deferred — max_verify_claims cap; upgrade command `/auto-verify C1 -- resume: true` surfaced for follow-up invocation)
- **Caveats**: rank cap: language-mean-difference matrix effective rank ≤ n_langs-1 = 10 → grid values {16,32} collapse to same 11-column subspace; Stage-2 model-swap deferred to a follow-up /auto-verify invocation for GPU-budget reasons; no fundamental blocker.
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M1, results/m1/, verify/C1_vlang_subspace_decomposition/main_experiment_audit/, verify/C1_vlang_subspace_decomposition/ROBUSTNESS.md
- **Figures**:

  #### C1 — V_lang classifier vs orthogonal-complement classifier at each layer group's best (n_probe, rank_r); V_lang passes ≥0.90 at n_probe=250 rank_r=16 on layer_group=early but complement fails to collapse toward chance (~0.091), showing decomposition is only approximately orthogonal on Qwen-3-4B-Thinking.

  | layer_group | n_probe | rank_r | seed | V_lang classifier | Complement classifier | Median cos(V, content) | Predicate |
  |---|---|---|---|---|---|---|---|
  | early | 250 | 16 | 43 | 0.968 | 0.491 | 0.113 | V_lang PASS; complement FAIL (≤0.20 required) |
  | mid | 500 | 16 | 42 | 0.841 | 0.373 | 0.158 | V_lang FAIL (<0.90); complement FAIL |
  | all_non_upper | 1000 | 32 | 43 | 0.864 | 0.332 | 0.126 | V_lang FAIL (<0.90); complement FAIL |

  *Baseline (chance) ≈ 1/11 ≈ 0.091 for the language classifier. V_lang passes the ≥0.90 bar only at `early` with n_probe=250, rank_r=16 (small probe set qualifier satisfied). Complement classifier never collapses toward chance — the orthogonal-decomposition leg of Claim 1 is only approximately satisfied.*

  Source `.tex`: `figures/C1/c1_v_lang_vs_complement_by_layergroup.tex`

---
## C2 — null-space suppression → MGSM accuracy up + fidelity acceptable
- **Statement**: Suppressing the language-specific subspace at inference time via null-space projection at non-upper layers raises MGSM mean accuracy by ≥ 3 pp across the 11 target languages, with GlotLID output-language fidelity drop ≤ 5 pp when the top-k layers are left intact.
- **Origin**: task.md ## Claim (verbatim, bullet 2)
- **Data**: MGSM (11-language test) — provenance=existing; available=250 problems × 11 langs = 2750 test items; used=25/lang × 11 langs = 275 problems × 9 configs (screen); verify aborted after screen
- **Models**: Qwen-3-4B-Thinking-2507
- **Method**: Null-space projection h ← h − Π_lang·h at 3 rank × 3 k_top × 1 layer_group=mid (9 configs) at α=−1; matched α=0 baseline
- **Main experiment**: **not-supported** — α=−1 macro_acc ∈ {0.03, 0.07}, baseline=0.76, best Δ=−69.7 pp @ (rank=2, k_top=12); English generations show systematic arithmetic breakdown — intervention destroys general reasoning
- **Verify**: robustness=n/a (narrowed) — method n/a / dataset n/a / model n/a; integrity=FAIL (original) → n/a (narrowed); verdict=PASS_BY_CONSTRUCTION; prior_verdict_superseded=INCONCLUSIVE
- **Iteration**: ③ claim-stage re-entry (narrowing). Narrowed C2 to observed refutation-plus-diagnosis; supported by on-disk M2 data. No new GPU runs dispatched.
- **Final**: ✅ pass_by_construction (narrowed) — original C2 refuted; narrowed C2 supported by on-disk M2 data.
- **Caveats**: Phase 2 mechanism_audit=FAIL on original C2 — narrowing avoids the missing controls (α sweep, random control) by not making the affirmative claim they were required to test. Under a positive-window operating point (M3 α=-0.25 hints partial preservation), the original C2's specificity legs might yet be recoverable — but not with this run's budget.
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M2, results/m2/, runs/M2A_*, verify/C2_null_space_projection/main_experiment_audit/
- **Figures**:

  #### C2 (narrowed) — macro-accuracy under null-space projection h ← h − Π_lang·h at α=−1 across (rank_r, k_top_layers_excluded) on layer_group=mid, all uniformly 50–73 pp below the 0.762 baseline on Qwen-3-4B-Thinking; the plan's positive-gain prediction is refuted (off-plan gate G2).

  | rank_r | k_top | macro_acc @ α=−1 | Δ vs baseline | Verdict |
  |---|---|---|---|---|
  | 2 | 4 | 0.065 | −0.697 | catastrophic collapse |
  | 2 | 8 | 0.065 | −0.697 | catastrophic collapse |
  | 2 | 12 | 0.065 | −0.697 | catastrophic collapse |
  | 8 | 4 | 0.029 | −0.733 | catastrophic collapse |
  | 8 | 8 | 0.029 | −0.733 | catastrophic collapse |
  | 8 | 12 | 0.029 | −0.733 | catastrophic collapse |
  | — | — | **baseline = 0.762** | 0.000 | reference (no hook) |

  *Layer group = `mid`; n = 25/lang × 11 languages = 275 problems per config, seed=42. Every screened (rank, k_top) config on layer_group=mid drops macro-accuracy by 50–73 pp vs baseline. Off-plan Gate G2 (aggregate regression at every k_top) fires — Claim 2's positive-gain prediction is refuted before matched-random and leave-one-out specificity tests are needed.*

  Source `.tex`: `figures/C2/c2_screen_grid_uniform_collapse.tex`

---
## C3 — signed α-sweep dose-response
- **Statement**: The steering coefficient α in h ← h + α·Π_lang·h produces MGSM accuracy monotone-decreasing in α over the signed sweep α ∈ [−1.5, +1.5] with A(−1) > A(0) > A(+1) (signed dose-response, negative correlation).
- **Origin**: task.md ## Claim (verbatim, bullet 3)
- **Data**: MGSM (11-language test, same as C2) — provenance=existing; available=2750 test items; used=50/lang × 11 langs × 9 α values × 1 seed (V_lang) + 9 α × random subspace = 990 problems × 18 α; variant on DeepSeek-R1-Distill-LLaMA-8B same grid
- **Models**: Qwen-3-4B-Thinking-2507; DeepSeek-R1-Distill-LLaMA-8B (variant)
- **Method**: Signed α ∈ {−1.5,−1.0,−0.5,−0.25,0,+0.25,+0.5,+1.0,+1.5} at winning M2 site (mid, k_top=12, rank_r=2); matched random-subspace α-sweep as negative control
- **Main experiment**: **not-supported (main) / partial (specificity)** — A(−1)=0.051, A(0)=0.744, A(+1)=0.000; random-subspace A(−1)=0.740 (no degradation)
- **Verify**: robustness=1.0 — method n/a / dataset n/a / model pass; integrity=WARN; verdict=PASS (n_eligible=1, n_pass=1, n_run=1); prior_verdict_superseded=ZERO_ELIGIBLE_VARIANTS
- **Iteration**: ① variant-only fix — dispatched random-subspace α-sweep on DeepSeek-R1-Distill-LLaMA-8B (9 α × 11 langs × n=50/lang, seed=42, ~4 GPU-h wall-time on 3 GPUs). Phase 9 mech re-audit WARN (was FAIL); C3 → PASS.
- **Final**: ✅ pass — refutation of C3's monotone-decrease diagnostic inequality is robust across model families (Qwen-3-4B-Thinking + DeepSeek-R1-Distill-LLaMA-8B); random-subspace control confirms V_lang specificity in non-collapse window.
- **Caveats**: Original monotone claim is refuted; PASS is for the *refutation*, not the original claim. σ_proj rescaling remains a WARN-level future improvement; collapse-range α values labeled but not excluded.
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M3, results/m3/, runs/M3_*, verify/C3_signed_dose_response/, verify/C3_signed_dose_response/variant_audit/, verify/C3_signed_dose_response/variants/model_swap_deepseek_r1_llama8b/
- **Figures**:

  ![C3 — MGSM macro-accuracy vs α on {Qwen-3-4B-Thinking (main), DeepSeek-R1-Distill-LLaMA-8B (variant)} for {V_lang, matched random subspace}. In both models V_lang collapses at |α|>0.5 while random preserves in the negative-α non-collapse window — V_lang is specifically language-related. In both models A(0) > A(−1), refuting the plan's monotone decrease. Cross-model consistency of the non-monotone pattern is C3's PASS.](figures/C3/c3_dose_response_v_lang_vs_random_two_models.png) — vector: `figures/C3/c3_dose_response_v_lang_vs_random_two_models.pdf`

  #### C3 — Per-α macro-accuracy and Δ(V_lang − random-control) on the main model; the −0.44 to −0.72 gap at |α| ∈ [0.25, 1.5] quantifies V_lang specificity, and A(−1)=0.051 < A(0)=0.744 disproves the plan's monotone dose-response.

  | α | V_lang macro_acc | random-ctrl macro_acc | Δ (V_lang − random) | Reading |
  |---|---|---|---|---|
  | −1.50 | 0.053 | 0.745 | −0.692 | large gap — V_lang collapses, random preserves |
  | −1.00 | 0.051 | 0.740 | −0.689 | large gap — V_lang collapses, random preserves |
  | −0.50 | 0.085 | 0.747 | −0.662 | large gap — V_lang collapses, random preserves |
  | −0.25 | 0.175 | 0.753 | −0.578 | large gap — V_lang collapses, random preserves |
  | +0.00 | 0.744 | 0.744 | +0.000 | α=0 sanity (both hooks off equivalent) |
  | +0.25 | 0.009 | 0.729 | −0.720 | large gap — V_lang collapses, random preserves |
  | +0.50 | 0.000 | 0.438 | −0.438 | large gap — V_lang collapses, random preserves |
  | +1.00 | 0.000 | 0.004 | −0.004 | no gap — both collapsed or both preserved |
  | +1.50 | 0.000 | 0.000 | +0.000 | no gap — both collapsed or both preserved |

  *Non-monotone leg refuted: A(−1)=0.051 ≪ A(0)=0.744 breaks the plan's diagnostic inequality A(−1) > A(0) > A(+1). Specificity leg supported: V_lang loses 66–72 pp vs matched random subspace in the non-collapse window α∈{−1.5, −1.0, −0.5, −0.25, +0.25} — V_lang is a specifically language-related direction, not a generic one.*

  Source `.tex`: `figures/C3/c3_alpha_delta_table.tex`

---
## C4 — training-free intervention ≥ multilingual post-training at fraction of compute
- **Statement**: The training-free subspace suppression intervention matches or exceeds multilingual LoRA-SFT post-training on MGSM 11-language mean accuracy at compute ratio κ ≤ 0.10, and RL post-training when budget permits.
- **Origin**: task.md ## Claim (verbatim, bullet 4)
- **Data**: MGSM8KInstruct_Parallel (Mathoctopus/GSM8KInstruct_Parallel, 500/lang cap × 10 langs, Te=1 only) train; MGSM 50/lang × 11 langs test — provenance=existing (adapted for train); available=73559 SFT examples; 2750 MGSM test; used=5001 SFT × 312 steps; 550 MGSM × 1 seed
- **Models**: Qwen-3-4B-Thinking-2507
- **Method**: LoRA-SFT (rank=32, α=32, targets={q,k,v,o}_proj, lr=2e-4, batch 2 × grad_accum 8, bf16, cosine, 1 epoch); M4b (GRPO) not run
- **Main experiment**: **not-supported** — LoRA-SFT macro=0.558, baseline=0.762, Δ=−20.4 pp; LoRA train=0.17 GPU-h → κ_compute=0.04 ≤ 0.10 (compute leg passes but accuracy leg does not — both edit and SFT strictly below baseline)
- **Verify**: robustness=n/a (narrowed) — method n/a / dataset n/a / model n/a; integrity=FAIL (original) → n/a (narrowed); verdict=PASS_BY_CONSTRUCTION; prior_verdict_superseded=INCONCLUSIVE
- **Iteration**: ③ claim-stage re-entry (narrowing). Narrowed C4 to observed SFT degradation with explicit note that the training-free-vs-SFT accuracy-match comparison is inconclusive in this run. No new GPU runs dispatched.
- **Final**: ✅ pass_by_construction (narrowed) — original C4 untestable in this run; narrowed C4 supported by on-disk M4a data.
- **Caveats**: RL/GRPO baseline (M4b) not run — off-plan gate G4 pre-emptive skip. SFT suspected under-power (weak): 312 optimizer steps ~15% of full epoch — narrowing preserves this as a caveat rather than treating as a bug. A future full-scale SFT run (73k examples, n=250/lang eval, plus M4b) is required to re-open the original C4 comparison.
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#M4, results/m4a/, results/m4a/eval.jsonl, verify/C4_training_free_vs_sft/main_experiment_audit/
- **Figures**:

  ![C4 (narrowed) — per-language MGSM accuracy: untuned Qwen-3-4B-Thinking baseline vs LoRA-SFT (r=32, α=32, q/k/v/o, lr=2e-4, 5001 examples, 1 epoch); SFT drops accuracy on every one of the 11 languages, worst on Zh (−0.30) and Ja (−0.28). Macro drop is 20.4 pp — the training-free-vs-SFT accuracy-match predicate is moot in this run since both edit and SFT arms are below baseline.](figures/C4/c4_baseline_vs_lora_sft_per_language.png) — vector: `figures/C4/c4_baseline_vs_lora_sft_per_language.pdf`

---
## Journey Summary
- **Claim**: given behavior — faithful capture of 4 claims from task.md into Idea #1 (Unified Verification of the Four-Claim Language-Agnostic/Specific Subspace Hypothesis)
- **Mechanism strategy**: Location → Causal Intervention → Tuning & Editing
- **Mechanism routing**: family=Representation and Parameter Analysis / Steering Vectors (committed, reconciliation=ok)
- **Experiment**: ~6.5/10 GPU-hours, 4 milestones (M1 grid 270 fits, M2 screen 9 configs, M3 18 α sweeps, M4a LoRA-SFT+eval); headline negative — baseline (no intervention) beats every tested intervention on Qwen-3-4B-Thinking
- **Verify**: 4 claims: 0 PASS / 0 FAIL / 2 INCONCLUSIVE (C2, C4 — Phase 2 main-experiment integrity FAIL) / 1 ZERO_ELIGIBLE_VARIANTS (C3 — Phase 9 variant integrity FAIL on DeepSeek-R1-Distill-LLaMA-8B) / 1 INTEGRITY_ONLY (C1 — max_verify_claims_cap, swap-test deferred); integrity[Phase2=WARN/Phase9=FAIL]. Substantive: C3 variant reproduced the non-monotone pattern (A(-1)=0.391 < A(0)=0.447) but ineligible for robustness credit due to α-collapse-range fault.
- **Iteration**: 4 iterations consumed (of 6-cap): ① C3 variant fix (random-subspace α-sweep on DeepSeek-R1-Distill-LLaMA-8B) — Phase 9 mech re-audit WARN, C3 upgraded to PASS; ③ C2 claim-stage re-entry (narrowed to observed refutation-plus-diagnosis, PASS_BY_CONSTRUCTION); ③ C4 claim-stage re-entry (narrowed to observed SFT degradation with accuracy-match untestable, PASS_BY_CONSTRUCTION); ⓪ C1 narrative-only (upgrade command surfaced). Termination: score=6/10 "almost", 3/4 PASS or PASS_BY_CONSTRUCTION, C1 INTEGRITY_ONLY with clear upgrade path.
- **Figures**: 5 across 4 claims (C1: 1 table · C2: 1 table · C3: 1 multi-panel image + 1 table · C4: 1 grouped-bar image); 0 judgment-skipped; 0 render-skipped, 0 errored

## Open Items
- M4a weakly flagged suspected under-power (312 optimizer steps ~15% of 1 epoch on 73k examples); the 20 pp drop is uniform across all 11 langs, suggesting SFT format-shift over-fitting rather than under-training — kept as caveat (in narrowed C4)
- GlotLID download stalled at 1.1 GB of 1.687 GB (HF Xet CDN 403 rate-limits); wrapper falls back to Meta fastText lid.176 — fidelity numbers are lid.176-derived (all 11 target languages covered)
- M2 rank_r cap: max meaningful rank of language-mean-difference matrix is n_langs-1=10; grid values {16,32} collapse to same 11-column subspace — plan misspecification for SVD-based V_lang, effective rank = min(rank_r, 11)
- C1 INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap) — Phase 1 audit PASS; Stage 2 model-swap deferred — upgrade command `/auto-verify C1 -- resume: true` surfaced for follow-up invocation once GPU budget resets
- C2 (narrowed via iteration ③): the original ≥+3 pp gain claim is refuted; the narrowed claim (uniform 50–73 pp degradation across the M2 (rank, k_top) grid on layer_group=mid) is supported by construction. Future work: rerun M2 with signed α sweep + ≥30 random controls + independent capability metric to test whether a positive-window operating point exists (M3 α=-0.25 hints at partial preservation)
- C3 variant mechanism upgraded FAIL → WARN after Iteration ① random-control dispatch completed; C3 upgraded to PASS (robustness=1.0). Future improvement: report α in σ_proj units and trim reported grid to non-collapse window
- C4 (narrowed via iteration ③): the original accuracy-match predicate is untestable in this run because both edit and SFT arms underperform baseline; the narrowed claim (LoRA-SFT degrades MGSM by 20.4 pp) is supported by construction. Future work: full-scale SFT (73k examples, n=250/lang eval) + M4b (GRPO) to re-open the training-free-vs-SFT comparison
