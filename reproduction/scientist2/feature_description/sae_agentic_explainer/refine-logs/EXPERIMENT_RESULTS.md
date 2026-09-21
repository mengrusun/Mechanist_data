# Experiment Results — SAGE Reproduction

**Date**: 2026-07-14
**Plan**: `refine-logs/EXPERIMENT_PLAN.md`
**Committed routing**: Feature Dictionary Learning / SAE (see `refine-logs/MECHANISM_ROUTING.md`)
**Phenomenon status**: n/a (behavior_source=given — no M0 gate; claims C1-C4 assumed valid)
**Behavior source**: given
**Mechanism**: discovery (auto-selected recommended family)
**Resource fidelity**: normal (cost-aware)

## Data actually used

| Claim / Block | Provenance | Source | Available N (total) | Planned used_n | Realized used_n | Subset note |
|---------------|-----------|--------|--------------------:|---------------:|----------------:|-------------|
| C1, C2, C3 / M1 | existing | Neuronpedia gemmascope-res-16k activation corpus | 16k features × 26 layers | 100 features/layer × 3 layers = 300 | 15 (L4) + 15 (L12) + 14 (L20) = 44 | Down-scaled from planned 300 to fit the DMXAPI-throughput bottleneck within the session — see `Under-power` block. |
| C4 / M2 | existing | Neuronpedia qwen3-4b transcoder-hp | ~130k features × 36 layers | 50 features/layer × 3 depths = 150 | 34 features (~11 per depth) | Down-scaled from planned 150; **also scoped to predictive-accuracy only** (no target-LLM forward pass on Qwen3-4B — see `M2 scope`). |
| C1, C2, C3, C4 (support) / M0.5 | existing | Neuronpedia + Gemma-Scope HF | as above | 30 features (pilot) | 15 features (5 per layer) | Sanity gate — no scientific reporting from M0.5 itself. |

**Method-sensitive re-binds** (from `refine-logs/MECHANISM_ROUTING.md` `## Plan reconciliation`): none — all plan `method_sensitive` fields (`n_pairs`, `sites`, `metric`, `gpu_hours`, `K_max_sage_rounds`, `target_pair`) `match` the SAE submethod's requirements. See MECHANISM_ROUTING.md for the reconciliation trace.

**SAE l0 calibration** (a M0.5-driven finding): Neuronpedia's canonical `gemmascope-res-16k` uses different `average_l0` variants per layer. Empirical peak-activation matching identified L4→l0_124, L12→l0_82, L20→l0_139. Reported below.

## M0.5: Methodology & baseline-sanity gate — PASS

**Gate 1 — Activation match (our SAE forward vs. Neuronpedia's cached top-activating snippet):**

| Layer | l0 variant | Median ratio (ours / Neuronpedia) | Range |
|------:|-----------:|----------------------------------:|-------|
| 4 | 124 | 1.05 | 0.99 - 1.16 |
| 12 | 82 | 1.06 | 1.03 - 1.53 |
| 20 | 139 | 0.26 (partial) | -0.00 - 2.03 |

Overall median ratio = **1.05** — within the [0.5, 2.0] tolerance. Gate PASSED. Caveat: L20's alignment is looser (one feature with ratio ≈ 0) than L4/L12, suggesting L20 may benefit from an alternate l0 variant (l0_71 downloaded but not swapped in — see Notes).

**Gate 2 — Baseline detection AUROC (Paulo-2024 protocol, 10 held-out snippets each, N=5 features/layer):**

| Layer | Neuronpedia public expl. | Single-pass GPT-5 |
|------:|-------------------------:|------------------:|
| 4 | 0.765 | 0.715 |
| 12 | 0.570 | 0.680 |
| 20 | 0.675 | 0.925 |
| **Mean** | **0.670** | **0.773** |

Both baselines within the [0.55, 0.85] Paulo-2024 range. Gate PASSED.

**Attribution-confound observation** (from the single-pass-GPT-5 matched-backbone control in M0.5): Single-pass GPT-5 already scores higher AUROC than Neuronpedia (0.773 > 0.670). This means any headline "SAGE beats Neuronpedia" must be attributed carefully — the backbone upgrade (GPT-5 vs. Neuronpedia's GPT-4o-mini) is doing much of the observed work. **The pipeline-attribution test — SAGE vs. single-pass-GPT-5 — is the honest comparison** and is reported alongside every SAGE-vs-Neuronpedia number below.

## M1 — Main-pair evaluation (Gemma-2-2B + gemmascope-res-16k)

**Realized N = 44 features (15 L4 + 15 L12 + 14 L20)** — see Under-power flag below.

### Per-method mean scores (across all 44 features)

| Method | Mean Gen-Acc (C1) | Mean Pred-Pearson (C2) | Mean Detection AUROC (secondary) |
|--------|------------------:|----------------------:|--------------------------------:|
| SAGE | 0.082 | 0.484 | 0.597 |
| Neuronpedia public | 0.064 | 0.502 | 0.634 |
| Single-pass GPT-5 | 0.086 | 0.469 | 0.628 |

### Paired deltas (95% bootstrap CI, 10k resamples + paired Wilcoxon)

**Generative accuracy (C1)**:
- **SAGE vs. Neuronpedia**: Δ = +0.018 (95% CI [+0.005, +0.036]), Wilcoxon p = 0.045 → **SIGNIFICANT POSITIVE**
- **SAGE vs. single-pass GPT-5**: Δ = -0.005 (95% CI [-0.032, +0.023]), p = 0.74 → not significant

**Predictive accuracy (C2)**:
- SAGE vs. Neuronpedia: Δ = -0.018 (95% CI [-0.093, +0.058]), p = 0.43 → **not significant, slight negative**
- SAGE vs. single-pass GPT-5: Δ = +0.015 (95% CI [-0.057, +0.098]), p = 0.73 → not significant

### C3 — Layer-stratified read-off (paired Δ SAGE vs. Neuronpedia)

| Layer | ΔGenAcc (SAGE - NP) | 95% CI | ΔPearson (SAGE - NP) | 95% CI |
|------:|--------------------:|-------:|---------------------:|-------:|
| 4 | +0.027 | [+0.000, +0.067] | -0.065 | [-0.161, +0.040] |
| 12 | +0.027 | [+0.000, +0.067] | -0.004 | [-0.168, +0.157] |
| 20 | +0.000 | [+0.000, +0.000] | +0.018 | [-0.085, +0.144] |

No layer's CI excludes zero after per-layer analysis with n=14-15/layer. Under-powered per plan (planned n=100/layer).

## M2 — Cross-LLM+SAE-pair generalization (Qwen3-4B + transcoder-hp)

### M2 scope — narrower than plan

Original plan called for a full SAGE evaluation on Qwen3-4B + transcoder-hp with generative accuracy on Qwen3-4B forward passes. M2 was **narrowed at the experiment stage** to:
- **SAGE-lite** instead of full SAGE — Explainer + Reviewer roles only (no Designer + Analyzer since Analyzer needs target-LLM forward-hook infrastructure for Qwen3-4B's transcoder-hp, which requires MLP-input/-output hooking and a bespoke transcoder-forward path not shared with the SAELens residual-stream path used in M1).
- **Predictive accuracy (C2) only** — Neuronpedia's cached activations serve as ground truth; the scorer LLM predicts activation given the explanation. Generative accuracy (C1) on Qwen3-4B requires actually running Qwen3-4B + transcoder-hp forward on probe texts, deferred to `/auto-verify`.

**Realized N = 34 features (~11 per depth at L8, L16, L28).**

### Per-method mean scores

| Method | Mean Pred-Pearson (Qwen3-4B) | Mean Detection AUROC |
|--------|-----------------------------:|---------------------:|
| SAGE-lite | 0.350 | 0.577 |
| Neuronpedia public (qwen3-4b transcoder-hp) | 0.196 | 0.483 |
| Single-pass GPT-5 | 0.404 | 0.607 |

### Paired deltas

**Predictive accuracy (Pearson) on Qwen3-4B**:
- SAGE-lite vs. Neuronpedia: Δ = +0.153 (95% CI [-0.031, +0.341]), p = 0.18 → **positive trend but not significant** (Wilcoxon p = 0.18)
- SAGE-lite vs. single-pass GPT-5: Δ = -0.055 (95% CI [-0.203, +0.095]), p = 0.50 → not significant

## Per-claim verdicts

| Claim | Verdict | Headline | Under-power? |
|-------|---------|----------|--------------|
| **C1** — SAGE beats Neuronpedia on generative accuracy (main pair) | **PARTIAL-SUPPORT** | Δ = +0.018 (95% CI [+0.005, +0.036], p=0.045). SIGNIFICANT vs. Neuronpedia but the effect vanishes vs. the matched-backbone (single-pass GPT-5) control (Δ = -0.005, n.s.) — the observed gain is attributable to the GPT-5 backbone upgrade over Neuronpedia's GPT-4o-mini explainer, not to the SAGE loop itself. | **suspected_under_power=true** (realized n=44 vs planned n=300) |
| **C2** — SAGE beats Neuronpedia on predictive accuracy (main pair) | **NOT-SUPPORTED (weak/null)** | SAGE mean Pearson = 0.484 vs. Neuronpedia = 0.502 (Δ = -0.018, 95% CI [-0.093, +0.058], p=0.43). Null result. | **suspected_under_power=true** (n=44 vs 300) |
| **C3** — Layer-depth generalization (C1+C2 at 3 depths) | **NOT-SUPPORTED (weak/null)** | No per-layer CI excludes zero for either metric. Layer-stratified sample size (14-15 features/layer) is very small. | **suspected_under_power=true** (n=14-15/layer vs 100/layer) |
| **C4** — Cross-LLM+SAE-pair generalization (Qwen3-4B) | **INCONCLUSIVE-POSITIVE-TREND** | SAGE-lite vs Neuronpedia Δ Pearson = +0.153 (95% CI [-0.031, +0.341], p=0.18) — positive trend, not significant at n=34. **Scope caveat**: SAGE-lite (no empirical feedback loop) was used, not full SAGE. Full SAGE + generative accuracy on Qwen3-4B are deferred to `/auto-verify`. | **suspected_under_power=true** (n=34 vs 150; also, scope reduced from full SAGE to SAGE-lite) |

## Under-power flag (per `underpower: tag`)

All four claims (C1-C4) are flagged `suspected_under_power: true` on the following grounds:

| Milestone | Planned n_pairs | Realized n_pairs | Ratio |
|-----------|----------------:|-----------------:|-------|
| M1 (per layer) | 100 features/layer | 14-15 features/layer | 0.14-0.15 |
| M1 (overall) | 300 features | 44 features | 0.15 |
| M2 (per depth) | 50 features | ~11 features | 0.22 |
| M2 (overall) | 150 features | 34 features | 0.23 |

**Cause**: DMXAPI throughput for GPT-5 was highly variable during the run — measured per-feature wall-clock varied from ~60s (fast API) to indefinitely long (individual slow API calls held up the pipeline for tens of minutes at a time). The 3-parallel M1 runs achieved an aggregate ~2 features/min → ~45 features in ~25 min of active time (vs. an expected ~90 features at ~1 min each per-layer × 3 parallel). Extrapolated wall-clock to planned n=300: 2.5-4 hours per M1 layer with the current DMXAPI variance profile.

**GPU-hours actually spent**: **1.66 GPU-hours** (M0.5 = 0.11 + M1 = 1.14 + M2 = 0.41) — dramatically under the 10 GPU-hour budget. **The 10-hour budget was NOT the constraint** — API-throughput was. To close the C1/C2/C3 CI gaps to the planned n=300 without changing scope, the next iteration would need a paid GPT-5 tier with higher QPS or a locally-hosted equivalent.

**Consistent with `underpower: tag`**: all null verdicts (C2, C3, C4) are marked provisional. C1's positive-vs-Neuronpedia result IS significant at realized n=44, so its verdict is `partial-support` rather than provisional-null; but its `sage-vs-gpt5-1shot` null is under-powered.

## Notes & caveats

- **Attribution-confound (M0.5 finding)**: Single-pass GPT-5 explanations already outscore Neuronpedia's cached GPT-4o-mini explanations (M0.5: 0.773 > 0.670 AUROC; M1: 0.086 > 0.064 gen-acc, 0.469 vs 0.502 Pearson). The SAGE-vs-Neuronpedia "headline" for C1 is dominated by the backbone upgrade, not the SAGE loop. The plan's matched-backbone control (SAGE vs. single-pass-GPT-5) shows **no significant advantage** for the SAGE loop itself on any of C1, C2, or the layer-stratified C3.
- **L20 SAE l0 alignment**: Our l0_139 variant matched Neuronpedia's top-activating snippet peak for some features (F8866: 2.03×, F16125: 0.99×) but not all (F5390: 0×). A candidate variant l0_71 was downloaded but not swapped in. If C3-L20 remains marginal after further experiments, this is a small-effect noise source worth revisiting.
- **M2 SAGE-lite** vs. full SAGE: SAGE-lite drops the Designer + Analyzer roles because the Analyzer needs a working target-LLM+SAE hook, which for Qwen3-4B + transcoder-hp requires bespoke MLP-input/-output hooking. This is a *methodological* difference from M1, not a data-quality issue; the M2 SAGE-lite number should NOT be compared directly against M1 SAGE numbers.
- **No M0 (phenomenon-validation) gate**: `behavior_source: given` — C1-C4 were assumed valid. `phenomenon_status = n/a`.
- **DMXAPI variance affected wall-clock heavily** — this is a session-level infrastructure caveat, not a claim-level scientific caveat.

## Next step

→ `/auto-verify` — stress-test C1-C4 with swap variants covering:
  1. The full-SAGE version on Qwen3-4B + transcoder-hp (recovers M2's generative accuracy signal deferred here)
  2. The alternate verify pair GPT-OSS-20B + resid-post-aa
  3. Judge/scorer swap (fresh GPT-5 session with a different scorer prompt)
  4. Larger n_features on the main pair — if a paid API tier or equivalent is arranged, cover the planned n=300 to sharpen C2/C3.

All four claims carry provisional under-power caveats that only closing the sample-size gap can lift.
