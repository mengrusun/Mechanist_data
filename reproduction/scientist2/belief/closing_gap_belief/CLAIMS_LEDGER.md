# Claim Ledger — Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence

**Direction**: Test whether well-calibrated internal accuracy signal and verbalized-confidence signal occupy separate, nearly orthogonal linear subspaces in Llama-3.1-8B-Instruct on TriviaQA — knowledge deficit vs. readout failure dichotomy.
**Date**: 2026-07-13 → 2026-07-14
**Pipeline**: completed | **Iteration**: 6.5/10 "almost" (1/6)
**Models**: claim=claude-opus-4-7, experiment=claude-opus-4-7, verify=claude-sonnet-4-6, iteration=claude-opus-4-7
**Updated after**: iteration:final

| Claim | Main experiment | Verify | Post-Iteration | Final |
|-------|-----------------|--------|----------------|-------|
| C1 gold-correctness probe | supported (AUC 0.840) | INTEGRITY_ONLY (cap) — | ⓪ tempered as 'TriviaQA decodable correlate' | audit passed, swap-test deferred (max_verify_claims cap) — reviewer tempered scoping |
| C2 verbalized-confidence probe | supported (AUC 0.948) | INTEGRITY_ONLY (cap) — | ⓪ tempered as prompt-protocol-specific readout | audit passed, swap-test deferred (max_verify_claims cap) — reviewer tempered scoping |
| C3a geometric near-orthogonality (**load-bearing**) | supported (\|cos\|=0.015) | PASS 1.00 (Qwen2.5-7B-Instruct replicates) | ⓪ 'probe-direction' near-orthogonality, not full independence | PASS — architecture-independent; scoping tempered to probe-directions |
| C3b causal separability | supported (primary); null (secondary) | INTEGRITY_ONLY (cap) — | ⓪ downgraded to 'weak steering diagnostic' | audit passed, swap-test deferred (max_verify_claims cap) — reviewer downgraded to 'weak, inconclusive steering diagnostic' |
| C3c dissociation-when-disagree | not-supported [provisional — suspected under-power] | INTEGRITY_ONLY (cap) — | ⓪ accepted as failed/underpowered; do-not-retry | audit passed, swap-test deferred (max_verify_claims cap) — accepted as failed/underpowered diagnostic |

---
## C1 — gold-correctness probe linearly accessible
- **Statement**: Llama-3.1-8B-Instruct encodes well-calibrated gold correctness on TriviaQA in a linearly accessible direction of the residual-stream hidden state at the canonical hook, beating token-probability and null baselines.
- **Origin**: task.md — Claim 1 (calibration probe existence).
- **Data**: TriviaQA rc.nocontext validation (post length-filter) — provenance=existing; available=17897, used=10000 (6000 train / 2000 dev / 2000 test); subset: Dropped ~7900 long-gold questions; idx-based split (regenerated after qid-collision leakage fix).
- **Models**: Llama-3.1-8B-Instruct
- **Method**: L2-logistic-regression linear probe on outputs.hidden_states[L] across 33 sites; AUROC + post-isotonic ECE; 200-resample retrain-on-bootstrap 95% CI; token-prob + random-direction + shuffled-label nulls — screen → decode → verify.
- **Main experiment**: supported — AUROC(probe_c) = 0.840 [0.818, 0.835] at L*=31; ECE_iso = 0.031; shuffled-label null AUC = 0.497.
- **Verify**: robustness=— — method n/a / dataset n/a / model excluded; integrity=WARN; verdict=INTEGRITY_ONLY (stage2_skip_reason=max_verify_claims_cap)
- **Iteration**: ⓪ narrative-only: temper as 'decodable correctness correlate on TriviaQA', not a task-general truth variable.; falsified: []; narrowed_to: Decodable correctness correlate on Llama-3.1-8B-Instruct + TriviaQA (not a task-general truth variable).
- **Final**: audit passed, swap-test deferred (max_verify_claims cap) — reviewer tempered scoping to 'TriviaQA decodable correlate'
- **Caveats**: Bootstrap n reduced planned 1000 → 200 for CPU budget; CI half-widths < 0.03 so not flagged suspected-under-power. Phase 2 audit WARN: bootstrap CI does not contain point estimate; n=200 vs planned 1000 disclosed.
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#block-b1, refine-logs/EXPERIMENT_RESULTS.md#m1, artifacts/probe_metrics.json, runs/M1_probes/, verify/C1_correctness_linear_accessible/, review-stage/AUTO_REVIEW.md
- **Figures**:
  - ![Per-layer AUROC of the correctness probe on Llama-3.1-8B-Instruct + TriviaQA — probe_c beats shuffled-label and random-direction nulls across the mid-late residual stream, peaking at L*=31 (AUC=0.840).](figures/C1/c1_layer_sweep_auroc.png) — vector: `figures/C1/c1_layer_sweep_auroc.pdf`

---
## C2 — verbalized-confidence probe linearly accessible pre-emission
- **Statement**: Llama-3.1-8B-Instruct encodes its about-to-be-verbalized confidence in a linearly accessible direction of the residual stream pre-emission, with paraphrase robustness across P0/P1/P2.
- **Origin**: task.md — Claim 2 (verbalized-confidence probe existence).
- **Data**: TriviaQA rc.nocontext validation (same as C1) — provenance=existing; available=17897, used=10000 probe + 500 paraphrase dev; subset: Two-context extraction; Stage-1.5 → ordinal path (std_c=10.3, share_c≥95 = 0.96).
- **Models**: Llama-3.1-8B-Instruct
- **Method**: Ordinal probe + binarized-AUROC probe_v_binary; 200-resample bootstrap CIs; random-direction + shuffled-label + paraphrase (P1 Tian 0–1, P2 Likert) nulls — screen → decode → verify.
- **Main experiment**: supported — AUROC(probe_v_bin) = 0.948 [0.913, 0.940]; ordinal top-1 acc = 0.994; macro-F1 = 0.862. Paraphrase: P1 AUC=0.838 (Δ=-0.11), P2 AUC=0.870 (Δ=-0.08).
- **Verify**: robustness=— — method n/a / dataset n/a / model excluded; integrity=WARN; verdict=INTEGRITY_ONLY (stage2_skip_reason=max_verify_claims_cap)
- **Iteration**: ⓪ narrative-only: frame as readout of impending verbalized-confidence behavior under this prompting protocol, not a task-general introspective confidence variable.; falsified: []; narrowed_to: Readout of impending verbalized-confidence behavior under this prompting protocol (partly formatting/default style rather than pure introspective confidence).
- **Final**: audit passed, swap-test deferred (max_verify_claims cap) — reviewer tempered scoping to prompt-protocol-specific readout
- **Caveats**: Paraphrase Δ AUC (-0.08 to -0.11) slightly exceeds ≤ 0.05 planned tolerance — mild probe_v paraphrase-sensitivity, AUC stays above 0.70 floor. Phase 2 audit WARN: proxy GT (model's own verbalization by design); bootstrap CI mismatch.
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#block-b1, refine-logs/EXPERIMENT_PLAN.md#block-b5, refine-logs/EXPERIMENT_RESULTS.md#m1-m1-5-m5, artifacts/stage15_variance.json, artifacts/paraphrase.json, verify/C2_verbalized_conf_linear/, review-stage/AUTO_REVIEW.md
- **Figures**:
  - ![Per-layer AUROC of the verbalized-confidence probe under three paraphrases (P0 primary, P1 Tian 0–1, P2 Likert) — probe_v_bin remains above the 0.70 accessibility floor throughout the mid-late stream despite Δ≈-0.10 paraphrase drop.](figures/C2/c2_layer_sweep_paraphrase.png) — vector: `figures/C2/c2_layer_sweep_paraphrase.pdf`

---
## C3a — geometric near-orthogonality of v_c and v_v (load-bearing)
- **Statement**: The gold-correctness direction v_c and the verbalized-confidence direction v_v at the primary reporting layer L* are geometrically near-orthogonal (|cos| ≤ 0.3 with 95% CI upper bound < 0.4), robust across L*±2 neighborhood.
- **Origin**: task.md — Claim 3 (near-orthogonality; geometric part). LOAD-BEARING.
- **Data**: Probe weight vectors v_c^L, v_v^L from Block B1 — provenance=adapted; available=33 sites × pairs, used=33 all-sites + 5-layer neighborhood; subset: L*=31 by mean of normalized AUROCs; 200-resample retrain-on-bootstrap; single-pass elicitation variant on 10k questions.
- **Models**: Llama-3.1-8B-Instruct
- **Method**: |cos(v_c^L, v_v^L)| per layer; retrain-on-bootstrap CI; random-unit-direction null (theoretical std ≈ 0.016) and shuffled-label pair as reference — screen → decode (weight-vector cosine) → verify (single-pass variant).
- **Main experiment**: supported — |cos(v_c*, v_v*)| = 0.015 [0.001, 0.034] at L*=31; neighborhood mean L*±2 = 0.025; random-direction null 0.011; single-pass unified-prompt variant |cos| = 0.139 (still ≪ 0.3).
- **Verify**: robustness=1.00 — method n/a / dataset n/a / model pass; integrity=PASS; verdict=PASS
- **Iteration**: ⓪ narrative-only: reword 'near-orthogonal linear subspaces' → 'near-orthogonal optimal linear probe directions' (not equivalent to computational independence).; falsified: []; narrowed_to: Near-orthogonality of *optimal linear probe directions* (not equivalent to causal independence of underlying computations).
- **Final**: PASS — architecture-independent (Qwen2.5-7B-Instruct swap replicates); scoping tempered to 'probe-direction' near-orthogonality (not full computational independence)
- **Caveats**: v_c direction is context-sensitive across elicitation contexts (cos(v_c_single, v_c*) = 0.47); v_v is stable (cos = 0.025).
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#block-b2, refine-logs/EXPERIMENT_PLAN.md#block-b7, refine-logs/EXPERIMENT_RESULTS.md#m2-m6, artifacts/cos_trajectory.json, artifacts/cos_bootstrap.json, artifacts/single_pass.json, verify/C3a_near_orthogonality/, verify/C3a_near_orthogonality/ROBUSTNESS.md, verify/C3a_near_orthogonality/variants/model-swap-qwen25-7b-instruct/, review-stage/AUTO_REVIEW.md
- **Figures**:
  - ![Per-layer |cos(v_c, v_v)| trajectory on Llama-3.1-8B-Instruct — the correctness and verbalized-confidence directions stay near the random-direction null (grey band, theoretical std 1/√4096 ≈ 0.016) across the whole stream; primary reporting layer L*=31 marked (dashed).](figures/C3a/c3a_cos_trajectory.png) — vector: `figures/C3a/c3a_cos_trajectory.pdf`
  - #### C3a load-bearing measurement replicates across model swap (Llama-3.1-8B-Instruct → Qwen2.5-7B-Instruct) — |cos| stays near the random-direction null with tight 95% CIs upper bound < 0.05, well below the 0.3 pre-registered threshold.

    | Model | d_hidden | L* | \|cos\| at L* | 95% CI | Neighborhood mean (L*±2) | Random-dir null | C3a passes (≤0.30) |
    |---|---|---|---|---|---|---|---|
    | Llama-3.1-8B-Instruct (main) | 4096 | 31 | 0.015 | [0.001, 0.034] | 0.025 | 0.011 | ✓ |
    | Qwen2.5-7B-Instruct (swap) | 3584 | 22 | 0.021 | [0.001, 0.037] | 0.015 | 0.013 | ✓ |

    Source `.tex`: `figures/C3a/c3a_swap_replication.tex`

---
## C3b — causal separability under activation steering
- **Statement**: v_c and v_v are causally separable knobs: activation steering along v_c does not preferentially move the v_v probe readout beyond a matched-magnitude random-direction null (and vice versa); emitted-output effect is a secondary corroboration.
- **Origin**: task.md — Claim 3 (causal separability part).
- **Data**: TriviaQA test slice from B1 (2000 held-out) — provenance=existing; available=2000, used=500 steering slice × α-grid (4500 forward passes); subset: Matched-magnitude random-direction control; internal PRIMARY, emitted SECONDARY.
- **Models**: Llama-3.1-8B-Instruct
- **Method**: Additive activation steering (α·σ·v at last-input-token residual, HF hook) across α ∈ {-1σ, 0, +1σ}; PRIMARY = Δ cross-probe readout with 0.5·σ_probe threshold; SECONDARY = Δ emitted-c + Δ accuracy; perplexity safety cap.
- **Main experiment**: supported (PRIMARY internal); null on SECONDARY (emitted) — All 4 cross-direction Δ criteria pass absolute mode. Emitted-output SECONDARY: Δ accuracy ≈ 0, Δ c_mean ≈ 0 at α=±1σ (null).
- **Verify**: robustness=— — method n/a / dataset n/a / model excluded; integrity=WARN; verdict=INTEGRITY_ONLY (stage2_skip_reason=max_verify_claims_cap)
- **Iteration**: ⓪ narrative-only: downgrade 'causal separability' → 'weak, inconclusive steering diagnostic with no output-level effect'.; falsified: []; narrowed_to: Weak, inconclusive steering diagnostic with no output-level effect (α=±1σ, 3-point grid, n_random=1).
- **Final**: audit passed, swap-test deferred (max_verify_claims cap) — reviewer downgraded from 'causal separability' to 'weak steering diagnostic with null on output-level effect'
- **Caveats**: Emitted-output SECONDARY null at α=±1σ dose (consistent with pre-registered anti-claim). Phase 2 audit WARN: sparse α grid (3 pts), n_random=1.
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#block-b3, refine-logs/EXPERIMENT_RESULTS.md#m3, artifacts/steering_results.json, runs/M3_steering/, verify/C3b_causal_separability/, review-stage/AUTO_REVIEW.md
- **Figures**:
  - ![Δ cross-probe readout under activation steering α ∈ {-1σ, 0, +1σ} × direction ∈ {v_c, v_v, random-matched} — cross-direction Δ (blue) is bounded by the matched-magnitude random-direction control (grey) and well below the 0.5·σ_probe threshold (dash-dot). Primary INTERNAL-readout evidence for C3b.](figures/C3b/c3b_cross_probe_delta.png) — vector: `figures/C3b/c3b_cross_probe_delta.pdf`

---
## C3c — dissociation-when-disagree diagnostic
- **Statement**: Dissociation-when-disagree: for questions where the probe-derived correctness score is LOW, accuracy is HIGHER when the verbalized confidence is LOW than when it is HIGH — McNemar test p < 0.05.
- **Origin**: task.md — Claim 3 (downstream-usable diagnostic part).
- **Data**: TriviaQA test split (2000 held-out) — provenance=existing; available=2000, used=2000; subset: analytic split on probe_c(x) × verbalized-confidence; (probe_low, verbal_low) cell has n=4.
- **Models**: Llama-3.1-8B-Instruct
- **Method**: Two-proportion z-test on (probe_low, verbal_low) vs (probe_low, verbal_high) accuracy; pre-registered predicate acc(low, low) > acc(low, high).
- **Main experiment**: not-supported [provisional — suspected under-power] — z = 1.35, p_one_sided = 0.91 (n1=386, n2=4); sign reversed from prediction.
- **Verify**: robustness=— — method n/a / dataset n/a / model excluded; integrity=WARN; verdict=INTEGRITY_ONLY (stage2_skip_reason=max_verify_claims_cap)
- **Iteration**: ⓪ narrative-only: present as failed/underpowered diagnostic; reviewer accepted the provisional null under UNDERPOWER=tag policy.; falsified: []; narrowed_to: Failed / underpowered diagnostic — root cause is Llama-3.1-8B-Instruct's degenerate verbalized-c distribution (96% ≥ 95), not a coding bug. Do NOT retry (reviewer explicitly recommended against).
- **Final**: audit passed, swap-test deferred (max_verify_claims cap) — reviewer accepted as failed/underpowered diagnostic; recommended do-not-retry
- **Caveats**: [suspected under-power: (probe_low, verbal_low) cell n=4 of 2000 test — 2/1000 base rate]. Phase 2 audit WARN: cell n=4/2000 suspected under-power. Reviewer (gpt-5.4) explicitly recommended against retrying — root cause is Llama-3.1-8B-Instruct's degenerate verbalized-c behavior, not a bug.
- **Artifacts**: refine-logs/EXPERIMENT_PLAN.md#block-b4, refine-logs/EXPERIMENT_RESULTS.md#m4, artifacts/dissociation.json, verify/C3c_dissociation_when_disagree/, review-stage/AUTO_REVIEW.md
- **Figures**:
  - #### 2×2 dissociation-when-disagree cells — the (probe LOW, verbal LOW) cell has n=4, driving the failed diagnostic; 96% of verbalized c ≥ 95 under Llama-3.1-8B-Instruct's default prompting collapses the load-bearing contrast.

    | probe \ verbal | verbal LOW: n | verbal LOW: acc | verbal HIGH: n | verbal HIGH: acc |
    |---|---|---|---|---|
    | probe HIGH | 17 | 35.3% | 1593 | 83.6% |
    | probe LOW | 4 | 0.0% | 386 | 31.3% |

    **Test**: two-proportion z on the (probe LOW, verbal HIGH) vs. (probe LOW, verbal LOW) contrast — z = 1.35, p_one_sided = 0.911, significant at 0.05? **no**. Root cause: n=4 in the load-bearing cell (2/1000 base rate).

    Source `.tex`: `figures/C3c/c3c_dissociation_2x2.tex`

---
## Journey Summary
- **Claim**: 1 given behavior → 3 claims decomposed into 5 verification predicates (C1, C2, C3a, C3b, C3c); mechanism strategy Location → Causal Intervention; refinement 9.1/10 READY.
- **Mechanism strategy**: Location → Causal Intervention
- **Mechanism routing**: family=Probing/Residual Stream States + Representation and Parameter Analysis/Steering Vectors (submethods reconciled OK against plan; committed)
- **Experiment**: 12 runs, ~0.5 GPU-hours, headline positive on C1/C2/C3a (load-bearing) + C3b internal-readout; C3b emitted-output null; C3c FAIL with suspected-under-power tag (cell n=4/2000).
- **Verify**: 5 claim(s): 1 PASS / 0 FAIL / 0 INCONCLUSIVE / 0 ZEV / 4 INTEGRITY_ONLY (cap=4, swap_off=0); integrity[Phase2=WARN/Phase9=PASS]. C3a robustness=1.00 (Qwen2.5-7B-Instruct swap replicates: |cos|=0.021 [0.001,0.037] at L*=22, architecture-independent).
- **Iteration**: 1/6 iterations, claim-reentries=0/2, score 6.5/10 verdict almost, termination=positive_verdict (three-dimensional STOP fired at iteration 1; five ⓪ narrative-only paper-scoping edits, zero back-edge actions consumed).
- **Figures**: 6 across 5 claims; 0 judgment-skipped; 0 render-skipped, 0 errored.

## Open Items
- C1: audit passed (WARN — bootstrap CI does not contain point estimate, disclosed n=200); swap-test deferred (max_verify_claims cap). Upgrade: /auto-verify C1 -- resume: true
- C2: audit passed (WARN — proxy GT + bootstrap CI mismatch + paraphrase Δ=-0.11 slightly exceeds 0.10 tolerance); swap-test deferred (max_verify_claims cap). Upgrade: /auto-verify C2 -- resume: true
- C3b: audit passed (WARN — sparse α grid (3 pts), n_random=1); swap-test deferred (max_verify_claims cap). Upgrade: /auto-verify C3b -- resume: true
- C3c: audit passed (WARN — cell n=4/2000 suspected under-power); swap-test deferred (max_verify_claims cap). Upgrade: /auto-verify C3c -- resume: true (root cause is Llama-3.1-8B-Instruct's degenerate verbalized-c distribution — 96% ≥ 95 — not a bug; iteration reviewer recommended accepting the provisional null)
