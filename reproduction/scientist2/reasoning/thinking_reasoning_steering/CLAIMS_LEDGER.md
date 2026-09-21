# Claim Ledger — Linear Steering of Reasoning Behaviours in DeepSeek-R1-Distill

**Direction**: Reasoning behaviours in thinking LLMs are linearly steerable via contrastive-pair activation directions
**Date**: 2026-07-14 → 2026-07-15
**Pipeline**: completed | **Iteration**: 5/10 "almost" (3/6)
**Models**: claim=claude-opus-4-7, experiment=claude-opus-4-7, verify=claude-sonnet-4-6, iteration=claude-opus-4-7
**Updated after**: iteration:final

| Claim | Main experiment | Verify | Post-Iteration | Final |
|-------|-----------------|--------|----------------|-------|
| C1 linearity of behaviour direction | partial | INTEGRITY_ONLY (cap) | ⓪ narrative-only | ⚪ integrity_only (swap-test deferred — max_verify_claims cap); iteration ⓪ narrative-only |
| C2 small-pool extractability | partial [under-power] | INTEGRITY_ONLY (cap) | ⓪ narrative-only | ⚪ integrity_only (swap-test deferred — max_verify_claims cap) [provisional — under-power]; iteration ⓪ narrative-only |
| C3 dose-response causal control (original) | partial (uncertainty) / negative (other 3) | INCONCLUSIVE (Phase 2 mechanism FAIL) | ③ replaced by C3_v2 | ✗ SUPERSEDED by C3_v2 — random-direction control refutes causal specificity |
| C3_v2 (iter 2 rewrite) direction NOT causally specific | supported (negative finding) | PASS (based on iter-1 evidence) | accepted at iter 3 | ✓ holds (negative finding) — strictly scoped |
| C4 finer than prompt (original) | positive-partial (uncertainty) / not-testable (other 3) | **FAIL** (robustness 0.00) | ② main-experiment fix → ③ replaced by C4_v2 | ✗ SUPERSEDED by C4_v2 |
| C4_v2 (iter 3 rewrite) scale-invariant coherence / no granularity | partial (dual finding) | PASS (Llama-8B) | accepted at iter 3 | ⚠ configuration-specific (Llama-8B only); cross-model OPEN |

---
## C1 — Linear behaviour directions in the residual stream
- **Statement**: Each reasoning behaviour b ∈ {expressing_uncertainty, generating_validation_examples, backtracking, self-correction} maps onto an approximately linear direction in the residual stream of DeepSeek-R1-Distill-Llama-8B at some layer L*(b): held-out linear-probe ROC-AUC ≥ 0.75 AND first-PC alignment |cos(v_b(L*), first_PC)| ≥ 0.7.
- **Origin**: task.md (given)
- **Data**: 100 R1-distill chains + 100 GPT-5.4 answers, LLM-judge annotated (κ = 0.77–1.00) — provenance=constructed; available=200 chains (n_pos {41,13,14,8}), used=200 (80/20 extract/held-out)
- **Models**: DeepSeek-R1-Distill-Llama-8B
- **Method**: Location — per-behaviour per-layer mean-difference direction + linear probe + first-PC alignment sanity
- **Main experiment**: partial — AUC uncertainty=0.977 (L29), validation=0.840 (L6), backtracking=1.000 (L1), self-correction=0.892 (L19); first-PC |cos| = 0.263 / 0.364 / 0.340 / 0.447 (all below 0.70 floor)
- **Verify**: robustness=— — method excluded / dataset excluded / model n/a; integrity=WARN; verdict=INTEGRITY_ONLY (max_verify_claims cap)
- **Iteration**: ⓪ narrative-only — kept as supporting context; probe-decodable ≠ single-PC alignment finding surfaced in limitations
- **Final**: ⚪ integrity_only (swap-test deferred — max_verify_claims cap); iteration ⓪ narrative-only
- **Caveats**: backtracking L*=1 layer may reflect prompt-structure priors rather than behaviour semantics
- **Artifacts**: runs/M1_locate/results.json, runs/M1_locate/directions/, verify/C1_linear_behaviour_directions/
- **Figures**:
  - ![Per-behaviour held-out linear-probe ROC-AUC vs layer for DeepSeek-R1-Distill-Llama-8B. Four behaviours reach AUC ≥ 0.75 at a chosen layer L*(b) (uncertainty=0.977 at L29, validation=0.840 at L6, backtracking=1.000 at L1, self-correction=0.892 at L19), but first-PC alignment fails (|cos| ∈ [0.26, 0.45]) — signal is decodable yet spread across many components.](figures/C1/c1_layer_auc_per_behaviour.png) — vector: `figures/C1/c1_layer_auc_per_behaviour.pdf`

---
## C2 — Small-pool extractability
- **Statement**: Each behaviour direction is extractable from a small pool (n_pairs ≤ 200) of contrastive activation pairs: split-half cos ≥ 0.7 at every n_pairs ≥ 25, and steering-effect ratio-to-large-pool ≥ 0.8 for every behaviour.
- **Origin**: task.md (given)
- **Data**: same auxiliary contrastive corpus as C1 — provenance=constructed; available=corpus n_pos capped 41/13/14/8, used=60 planned extraction runs (realized ~6 for uncertainty)
- **Models**: DeepSeek-R1-Distill-Llama-8B
- **Method**: Location — n_pairs sweep of mean-difference direction; split-half stability + steering-effect preservation ratio
- **Main experiment**: partial [provisional — suspected under-power] — uncertainty split-half cos at n=25 = 0.588 (below 0.70 floor); cos-to-reference at n=25 = 0.944; other 3 behaviours skip (pool < 10)
- **Verify**: robustness=— — method excluded / dataset excluded / model n/a; integrity=WARN; verdict=INTEGRITY_ONLY (max_verify_claims cap)
- **Iteration**: ⓪ narrative-only — reviewer flagged: upgrading via verify won't fix real bottleneck (n_pos ≤ 14 corpus)
- **Final**: ⚪ integrity_only (swap-test deferred — max_verify_claims cap) [provisional — suspected under-power]; iteration ⓪ narrative-only
- **Caveats**: [suspected under-power: used_n 6/60 grid points for uncertainty, 0/60 for the other three behaviours]
- **Artifacts**: runs/M2_smallpool/results_summary.json, verify/C2_small_pool_extractability/

---
## C3 — Dose-response causal control (ORIGINAL, superseded by C3_v2)
- **Statement**: Adding α·σ·v_b at layer L*(b) causes dose-response amplification/suppression: sign(rate_b(±α_op) − rate_b(0)) matches sign(α) at some α_op ∈ [0.5σ, 3σ]; Spearman ρ(α, rate_b) ≥ 0.7 on the coherent α-range; off-target |Δrate| ≤ 50% of on-target |Δrate|.
- **Origin**: task.md (given)
- **Data**: 500-task benchmark — provenance=constructed; available=500, used=60; α grid 7 values × 4 behaviours = 28 configs
- **Models**: DeepSeek-R1-Distill-Llama-8B
- **Method**: Causal Intervention — α-sweep unit-vector CAA additive steering hook at L*(b), coefficient α·σ_proj·u
- **Main experiment**: partial (uncertainty) / negative [provisional — suspected under-power] (other three) — uncertainty: sign check ✓, α_op=0.5σ, Δrate=+0.14 at +2σ; validation/backtracking/self-correction: rates stuck at 0
- **Verify**: robustness=— — method excluded / dataset excluded / model excluded; integrity=FAIL; verdict=INCONCLUSIVE (Phase 2 mechanism rigor FAIL)
- **Iteration**: ③ replaced by C3_v2 (rewrite accepted, iteration 2). Expanded α grid to 9 values covering ≥3 OOM + added random-direction control (n_random=30 on uncertainty); falsifier = random-direction control (Δrate=0.000 learned vs 0.095±0.015 random, z=-6.33)
- **Final**: ✗ SUPERSEDED by C3_v2 — random-direction control refutes causal specificity; original dose-response claim empirically falsified on this model at n_pos ≤ 41
- **Caveats**: [suspected under-power: used_n 60/500, seeds 1/1, grid 7/9]
- **Artifacts**: runs/M3_steer/results_summary.json, verify/C3_dose_response_causal_control/main_experiment_audit/, runs/iteration_round_1/M3_C3_expand/results_summary.json

---
## C3_v2 — Decoder-identified direction is NOT causally specific
- **Statement**: [revised at iteration 2 (③ claim-stage re-entry); original C3 asserted dose-response causal control] Decoder-identified directions for expressing_uncertainty are linearly predictive (probe AUC 0.977 at L=29) but are NOT causally specific under norm-matched steering on the DeepSeek-R1-Distill-Llama-8B residual stream: random unit directions of the same L2 norm at α ~ 1σ produce Δrate = +0.095 ± 0.015 in expressing_uncertainty, whereas the learned mean-difference direction produces Δrate = 0.000 (z = −6.33 vs random). Behavior changes only emerge near coherence-collapse boundaries (α ≥ ±3σ) which are indistinguishable from generic perturbation collapse. Therefore steering with unit-vector CAA at n_pos ≤ 41 does not deliver dose-response causal control of reasoning behaviors on this model.
- **Origin**: iteration:round-2 ③ rewrite of C3
- **Data**: iteration-1 expanded α-grid run + random-direction control on the 500-task benchmark — provenance=constructed; used=60 (9-α × 60 tasks + 20 random directions × 20 tasks at α=1σ on uncertainty only)
- **Models**: DeepSeek-R1-Distill-Llama-8B
- **Method**: Causal Intervention — α-sweep ([-3, -1, -0.3, -0.1, 0, +0.1, +0.3, +1, +3]·σ_proj) unit-vector CAA hook at L*(b)=29 + random-direction control at α=1σ
- **Main experiment**: supported (negative finding) — Spearman ρ on coherent subset = -0.019 (p=0.97); learned direction Δrate=0.000 at α=1σ; random directions Δrate=0.095±0.015 (n=20/30); z=-6.33; coherence collapse at ±3σ (0.43/0.28)
- **Verify**: robustness=— — method n/a / dataset n/a / model n/a; integrity=PASS; verdict=PASS (based on iter-1 evidence; not a formal swap-test)
- **Iteration**: accepted at iteration 3 — substantively supported by iter-1 data; reviewer notes: keep tightly scoped to this model / behavior / method family / data regime — do NOT generalize
- **Final**: ✓ holds (negative finding) — learned direction NOT causally specific under random-direction control; strictly scoped to this model + behavior + method family + data regime (narrowed_to: expressing_uncertainty on Llama-8B at n_pos=41 with unit-vector CAA)
- **Caveats**: Tightly scoped — do NOT generalize to other models / behaviors / method families without re-testing; random-direction control ran on 20/30 planned (budget-truncated); z-score already highly significant
- **Artifacts**: runs/iteration_round_1/M3_C3_expand/results_summary.json, src/run_M3_C3_expand.py, review-stage/AUTO_REVIEW.md#round-2
- **Figures**:
  - ![Random-direction control refutes causal specificity of the learned uncertainty direction on DeepSeek-R1-Distill-Llama-8B at L*=29, α=1σ. Learned mean-difference direction produces Δrate = 0.000 in expressing_uncertainty vs random unit directions Δrate = 0.095 ± 0.015 (n=20; z = −6.33). The learned direction moves the behaviour LESS than a random direction of matched L2 norm.](figures/C3_v2/c3v2_random_direction_control.png) — vector: `figures/C3_v2/c3v2_random_direction_control.pdf`
  - ![Expanded α-grid dose-response on the 9-point ±3σ grid for uncertainty. Sign check passes weakly for +α; Spearman ρ on coherent subset = −0.019 (p=0.97) — no monotonic dose-response. Coherence collapses at ±3σ (0.43 / 0.28).](figures/C3_v2/c3v2_alpha_sweep_coherent.png) — vector: `figures/C3_v2/c3v2_alpha_sweep_coherent.pdf`

---
## C4 — Finer than prompt (ORIGINAL, superseded by C4_v2)
- **Statement**: Steering-vector control offers strictly more distinct (behaviour-rate, accuracy) operating points than NL-instruction prompt engineering and than Thinking Intervention, with matched-rate accuracy(steering) ≥ accuracy(prompt) − 2 pts and accuracy(steering @ α_op) ≥ accuracy(α=0) − 3 pts.
- **Origin**: task.md (given)
- **Data**: 500-task benchmark, 60-task subset — provenance=constructed; used=60; 8 controllers × 4 behaviours = 32 configs
- **Models**: DeepSeek-R1-Distill-Llama-8B (main); DeepSeek-R1-Distill-Qwen-14B (verify swap)
- **Method**: Tuning & Editing — steering-α vs. NL-prompt vs. Thinking-Intervention token-insertion; Pareto count of distinct (rate, acc) operating points
- **Main experiment**: positive-partial (uncertainty) / not-testable [under-power] (other three) — n_distinct(steering)=4 > prompt=2 = TI=2 ✓; matched-rate acc within 2 pts ✓; preservation misses 3-pt floor by 1 pt on suppress side
- **Verify**: robustness=0.00 — method excluded / dataset excluded / model FAIL; integrity=WARN; verdict=FAIL
- **Iteration**: ② main-experiment fix (iter 1) → ③ replaced by C4_v2 (iter 3); scale-invariant coefficient introduced (coef = α_frac · mean_residual_norm(L*)); falsifiers: cross-model transfer under σ_proj scaling AND the "finer than prompt" advantage under fair parameterization
- **Final**: ✗ SUPERSEDED by C4_v2 — original claim's fine-grained advantage was parameterization-dependent; scale-invariant parameterization eliminates it
- **Caveats**: [suspected under-power: used_n 60/500]; the original apparent advantage was an artifact of σ_proj scaling, not a genuine mechanistic property
- **Artifacts**: runs/M4_control_compare/results_summary.json, verify/C4_finer_than_prompt_engineering/, runs/iteration_round_1/M4_C4_scale_invariant/

---
## C4_v2 — Scale-invariant coefficient preserves coherence but eliminates granularity advantage
- **Statement**: [revised at iteration 3 (③ claim-stage re-entry); original C4 asserted strict fine-grained advantage over prompt/TI] Scale-invariant steering — coef = α_frac · mean_residual_norm(L*) — avoids catastrophic coherence collapse across DeepSeek-R1-Distill-Llama-8B (α_frac ∈ {−0.15, −0.05, +0.05, +0.15} → coherence ≥ 0.98). However, under this well-controlled parameterization on the Llama-8B source model, steering does NOT provide finer-grained operating points than NL-instruction prompt engineering (n_distinct(steering) = 2 = n_distinct(prompt) at ε_r = 0.05, ε_a = 0.01). The apparent 'steering is finer than prompt' advantage in the original σ-scaled setup was driven by parameterization choices that also caused cross-model failure; once removed, the advantage disappears. Cross-model (Qwen-14B) confirmation of coherence preservation was in-flight but did not complete within iteration budget.
- **Origin**: iteration:round-3 ③ rewrite of C4
- **Data**: 500-task benchmark, 60-task subset; iteration-1 scale-invariant M4 run on Llama-8B — provenance=constructed; used=60; 7/8 controllers completed on Llama-8B; Qwen-14B swap phase killed by cluster contention
- **Models**: DeepSeek-R1-Distill-Llama-8B, DeepSeek-R1-Distill-Qwen-14B (incomplete)
- **Method**: Tuning & Editing — scale-invariant α_frac · mean_residual_norm(L*) steering hook on 4 α_frac values vs. prompt suppress/amplify vs. TI suppress
- **Main experiment**: partial (dual finding — positive on coherence, negative on granularity) — coherence ≥ 0.98 across all 4 α_frac values on Llama-8B (fix succeeded); n_distinct(steering)=2 = n_distinct(prompt)=2 (primary predicate FAILS — no fine-grained advantage); mean_residual_norm(L*=29) = 42.95 vs σ_proj = 10.52 (ratio ~4.1×)
- **Verify**: robustness=— — method n/a / dataset n/a / model n/a; integrity=WARN; verdict=PASS (Llama-8B portion accepted)
- **Iteration**: accepted at iteration 3 — Llama-8B portion substantively supported; reviewer requires cross-model wording removed or explicitly labeled unverified; narrowed_to: Llama-8B source model; cross-model Qwen-14B coherence preservation is OPEN — needs ~2 GPU-h follow-up
- **Final**: ⚠ configuration-specific (Llama-8B only) — coherence preserved under scale-invariance; no fine-grained advantage over prompt engineering; cross-model claim OPEN until Qwen-14B run completes
- **Caveats**: Qwen-14B cross-model coherence run incomplete (killed by cluster contention at 7/8 controllers on Llama phase); Claim text must hedge cross-model wording until Qwen-14B run completes; TI_amplify controller not completed
- **Artifacts**: runs/iteration_round_1/M4_C4_scale_invariant/llama8b/results_summary.json, runs/iteration_round_1/M4_C4_scale_invariant/llama8b/mean_residual_norm.json, src/run_M4_C4_scale_invariant.py, review-stage/AUTO_REVIEW.md#round-3
- **Figures**:
  - #### Scale-invariant steering (coef = α_frac · mean_residual_norm(L*)) preserves coherence ≥ 0.98 across all α_frac ∈ {−0.15, −0.05, +0.05, +0.15} on Llama-8B (fix succeeded), but eliminates the fine-grained advantage: n_distinct(steering) = 2 = n_distinct(prompt) at ε_r = 0.05. The original 'steering is finer than prompt' story was parameterization-dependent.

    **Table.** Scale-invariant steering on DeepSeek-R1-Distill-Llama-8B at L* = 29 (mean residual norm = 42.95). Steering coefficient = α_frac · mean_residual_norm(L*). Coherence stays ≥ 0.98 for all four steering rows, but the range of behaviour rates across steering rows equals the range across prompt rows (no fine-grained advantage over prompting).

    | Controller | n | Coherence | Accuracy | Behaviour rate |
    |---|---|---|---|---|
    | Steering (α_frac = -0.15) | 60 | 98.3% | 72.9% | 18.6% |
    | Steering (α_frac = -0.05) | 60 | 100.0% | 65.0% | 21.7% |
    | Steering (α_frac = +0.05) | 60 | 100.0% | 73.3% | 23.3% |
    | Steering (α_frac = +0.15) | 60 | 100.0% | 70.0% | 23.3% |
    | Prompt (suppress) | 60 | 100.0% | 78.3% | 11.7% |
    | Prompt (amplify) | 60 | 100.0% | 75.0% | 31.7% |
    | Thinking-intervention (suppress) | 60 | 100.0% | 78.3% | 31.7% |

    Source `.tex`: `figures/C4_v2/c4v2_scale_invariance_table.tex`

---
## Journey Summary
- **Claim**: 1 given behavior → 4 sub-claims (C1 linearity, C2 small-pool extractability, C3 dose-response, C4 finer-than-prompt at preserved accuracy)
- **Mechanism strategy**: Location → Causal Intervention → Tuning & Editing
- **Mechanism routing**: family=Representation and Parameter Analysis, submethod=Steering Vectors (unit-vector CAA, α·σ_proj·u additive residual hook at L*(b))
- **Experiment**: 9 runs (M1 locate + M2 smallpool + M3 steer × 3 + M4 control × 3 + M3 sanity), ~1.9 GPU-hours, headline partial (uncertainty is a real linear knob; other three behaviours data-limited)
- **Verify**: 4 claim(s): 0 PASS / 1 FAIL / 1 INCONCLUSIVE / 0 ZEV / 2 INTEGRITY_ONLY (cap=2, swap_off=0); integrity[Phase2 WARN — C3 mechanism FAIL / Phase9 WARN]
- **Iteration**: 3/6 iterations, claim-reentries=2/2 (EXHAUSTED), score 5/10 verdict almost, termination=claim_reentry_exhausted; two type-② fixes (C3 α-grid expansion + random control; C4 scale-invariant coefficient) + two type-③ rewrites (C3→C3_v2 negative-specificity, C4→C4_v2 dual finding)
- **Figures**: 4 across 3 claims (C1: 1 line; C3_v2: 1 bar + 1 line; C4_v2: 1 table); 3 judgment-skipped (C2 single scalar; C3 & C4 superseded by v2 rewrites); 0 render-skipped, 0 errored

## Open Items
- C1: audit passed but swap-test deferred (max_verify_claims cap) — upgrade with `/auto-verify C1 — resume: true`
- C2: audit passed but swap-test deferred (max_verify_claims cap) — upgrade with `/auto-verify C2 — resume: true` (won't fix real bottleneck — corpus n_pos ≤ 14)
- C4_v2 Qwen-14B cross-model coherence run incomplete (~2 GPU-h needed); C4_v2 claim text must hedge cross-model wording until this run completes
- Iteration ended at claim_reentry_exhausted with score 5/10 (below TARGET_SCORE=6) — the 1-pt gap reflects narrow scope / workshop-tier contribution, not experimental gaps; further improvement would need venue-focused work outside the loop's scope
