# Experiment Plan

**Problem**: Measure the geometric alignment + causal cross-coupling between the internal gold-correctness direction and the internal verbalized-confidence direction in Llama-3.1-8B-Instruct on TriviaQA, testing the three fixed claims C1/C2/C3 from task.md.
**Method Thesis**: Two paired linear probes on residual-stream activations at a canonical hook, per-layer AUROC/Spearman/ECE, single primary reporting layer L\*, cosine similarity with neighborhood-robustness, cross-direction activation steering with matched-magnitude random-direction control (internal readout primary, emitted output secondary).
**Date**: 2026-07-13

---

## Top Metadata (machine markers)

```yaml
behavior_source: given
mechanism: discovery
mechanism_strategy:
  directions: [Location, Causal Intervention]
  rejected:
    - Tuning & Editing — the claims ask whether v_c and v_v are separate, not how to use them to improve calibration; that is a downstream engineering step handled by Kossen 2025 / Zhang 2025.
    - Formation Tracing — training-time genesis of the split is not part of C1/C2/C3 and would double the compute budget.
    - Unit Interpretation — SAE / ICA decomposition would characterize what each direction encodes; C3 is a direction-level geometric claim, testable from probe vectors directly.
    - Decision Auditing — asks whether decisions rest on the right features; C3 is a representation-geometry claim, not a per-decision audit.
  note: >
    Location (linear probing) supports C1 and C2 by locating each signal's direction at a canonical hook location.
    Causal Intervention (activation steering) supports C3b by testing that steering along one direction does not preferentially move the other's probe readout beyond a matched-magnitude random-direction null. The Location output vectors v_c^L*, v_v^L* feed directly into the Intervention milestone.
# Note: resource_fidelity is NOT stamped strict — this is BEHAVIOR_SOURCE=given + MECHANISM=discovery, the cost-aware combination (not the given+given reproduction combo).
```

---

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|-----------------|-----------------------------|---------------|
| C1 (gold correctness linearly accessible) | Establishes that Llama-3.1-8B-Instruct internally knows when it's correct on TriviaQA — necessary precondition for the readout-vs-knowledge dichotomy | AUROC(L_c\*) ≥ 0.70 with retrain-on-bootstrap 95% CI lower bound above 0.65; ECE(L_c\*) ≤ 0.10 post-isotonic; ECE(probe) < ECE(token_prob) — all beating random-direction and shuffled-label nulls | B1 (probing), B6 (nulls) |
| C2 (verbalized confidence linearly accessible pre-emission) | Establishes that the verbalization channel has its own linearly-decodable representation before the number is emitted | Stage-1.5 chosen path meets its threshold (Spearman ρ ≥ 0.5 continuous OR top-1 ordinal accuracy ≥ 0.55 + macro-F1 ≥ 0.4); binarized AUROC(L_v\*) ≥ 0.70; Δ ≤ 0.1 across paraphrase P0/P1/P2 | B1 (probing), B6 (nulls + paraphrase) |
| C3a (geometric near-orthogonality) | The load-bearing claim: quantifies dissociation as an angle, not just a scalar correlation | |cos(v_c^L\*, v_v^L\*)| ≤ 0.3 with 95% bootstrap CI upper bound < 0.4, gated on C1 & C2 passing; neighborhood-robust across L\*±2; consistent (or explicitly compared) in single-pass unified-prompt robustness variant | B2 (geometric), B7 (single-pass) |
| C3b (causal separability) | Promotes C3a from "geometrically distinct" to "causally distinct knobs" — the mechanism-level test | Cross-steering criterion (ratio OR absolute-effect) satisfied on internal readout (PRIMARY) in both cross directions v_c→v_v_readout and v_v→v_c_readout; emitted-output effect (SECONDARY corroboration) consistent in sign and privileged v_v→confidence-number direction | B3 (steering) |
| C3c (dissociation-when-disagree) | Turns C3 from static structural claim into a downstream-usable diagnostic | Accuracy(low probe, high verbal) < Accuracy(low probe, low verbal) with McNemar test p < 0.05 | B4 (dissociation) |

**Anti-claims explicitly ruled out**:
- "The result is due to probe overfitting" → shuffled-label null + retrain-on-bootstrap CIs.
- "The result is because both probes are weak" → C1 & C2 AUROC gates fire before C3a is interpretable.
- "The near-orthogonality is trivially expected under any two probes" → random-direction cosine null shows the *scale* of random near-orthogonality (std √(1/4096) ≈ 0.016); the claim is that BOTH probes independently pass AUROC gates AND their |cos| is not larger than a shuffled-label pair.
- "The result depends on the two-context extraction" → single-pass unified-prompt variant (B7) reports |cos| under a single elicitation context.

---

## Paper Storyline

- **Main paper must prove**: C1, C2, C3a, C3b (internal-readout primary).
- **Appendix can support**: C3b emitted-output corroboration, C3c dissociation-when-disagree, single-pass robustness variant, paraphrase P1/P2 robustness, per-layer trajectories with neighborhood-robustness figure.
- **Experiments intentionally cut**: SAE decomposition of v_c and v_v, formation-tracing of the split across training checkpoints, fine-tuning-based direct-confidence-alignment intervention, cross-dataset angle generalization (would confound C3 per Kim et al. 2025), circuit-level mechanistic breakdown.

---

## Experiment Blocks

### Block B1: Linear-probe layer sweep — locates v_c and v_v (covers C1 & C2)
- **Claim tested**: C1 (correctness probe accessibility), C2 (verbalization probe accessibility), extracts v_c^L and v_v^L for C3.
- **Why this block exists**: without B1 there are no directions to compare. This is the Location stage of the mechanism strategy.
- **Dataset / split / task**:
  - Provenance: **existing**
  - Source: **TriviaQA** (`rc.web` subset, validation split)
  - Available N: 10,833 questions in `rc.web` validation
  - Planned used N: **10,000** — dropped ~800 questions with post-normalization gold answer > 5 tokens (out-of-scope for closed-book QA setting; not a cost cut)
  - Split: stratified by post-normalization gold-answer length quartile → **6,000 train probe / 2,000 dev / 2,000 test**
- **Compared systems**:
  - probe_c_binary vs. token-probability confidence vs. P(True) self-elicitation vs. random-direction probe null vs. shuffled-label null (for C1)
  - probe_v_primary (continuous or ordinal; Stage-1.5-selected) vs. random-direction null vs. shuffled-label null (for C2)
- **Metrics**:
  - C1: AUROC (primary), ECE post-isotonic on dev, retrain-on-bootstrap 95% CI (1000 resamples)
  - C2 primary path: Spearman ρ or ordinal top-1 accuracy + macro-F1
  - C2 secondary: binarized AUROC of probe_v_binary
- **Setup details**:
  - Canonical hook: `outputs.hidden_states[L]` (post-block residual) at last-input-token, all 32 layers
  - L2-regularized logistic regression (probe_c_binary, probe_v_binary) with C ∈ {0.001, 0.01, 0.1, 1, 10} tuned on dev; L2-regularized linear regression for probe_v_continuous; ordinal cross-entropy for probe_v_ordinal
  - vllm for forward-pass 1 answer generation (batch 64, max_new_tokens 20, greedy); HuggingFace transformers with forward-hook for hidden-state extraction (batch 32, no generation)
- **Success criterion**: C1 AUROC(L_c\*) ≥ 0.70; C2 primary threshold met at L_v\*; both above random-direction and shuffled-label nulls with non-overlapping 95% CIs.
- **Failure interpretation**: If C1 fails, the correctness signal is not linearly accessible → the anchor's dichotomy is refuted (both signals absent). If C2 fails, either the model does not internally commit to a specific confidence before emission (a *finding*), or the pre-emission probe target is under-specified.
- **Table / figure target**: Table 1 (main paper) — per-layer AUROC / Spearman / ECE for probe_c and probe_v_primary with bootstrap CIs; Fig 1 (main paper) — per-layer AUROC curves for both probes.
- **Priority**: MUST-RUN
- **Estimated GPU-hours**: 2.5 h (2 h collection + 0.5 h probe re-fits for bootstrap)
- **method_sensitive**: `[n_pairs, sites, metric, gpu_hours]` (subsymptom candidate probe families at /auto-experiment Phase 1.5 may adjust concrete probe-fitting protocol / evaluation metric within the "linear probing" family)

### Block B2: Geometric measurement — cosine similarity + per-layer trajectory (covers C3a)
- **Claim tested**: C3a (near-orthogonality of v_c and v_v at the primary reporting layer + neighborhood).
- **Why this block exists**: this is the direct measurement no prior published paper reports.
- **Dataset / split / task**:
  - Provenance: **existing** (reuses B1's probe weights, no additional data)
  - Source: probe weight vectors from B1
  - Available N: 32 layer pairs (one per layer)
  - Planned used N: 32 (all)
- **Compared systems**:
  - cos(v_c^L, v_v^L) primary
  - cos(v_c^L, random_unit_direction) null (theoretical std √(1/4096) ≈ 0.016)
  - cos(v_c^L, v_v^L_shuffled_labels) null (upper reference — a probe trained on random labels should give the same near-zero cosine as random directions)
- **Metrics**: |cos| with retrain-on-bootstrap 95% CI at L\* (primary); |cos| per layer with eval-only 95% CI (secondary per-layer curve)
- **Setup details**:
  - `L* = argmax_L [ AUROC_c(L) / max_L' AUROC_c(L') + AUROC_v_binary(L) / max_L' AUROC_v_binary(L') ]` — the mean of normalized AUROCs across the two binary probes
  - Neighborhood window: L\*±2 layers (5-layer band)
  - Bootstrap procedure for primary CI at L\*: 1000 resamples of the 6k probe-training set, re-fit both probes on each resample, re-extract v_c^{L\*} and v_v^{L\*}, compute cos
- **Success criterion**: |cos(v_c^{L\*}, v_v^{L\*})| ≤ 0.3 with 95% CI upper bound < 0.4; the mean |cos| across L\*±2 layers ≤ 0.3 (neighborhood-robustness).
- **Failure interpretation**:
  - If |cos| ≈ 1 → knowledge-deficit interpretation: v_c and v_v are the same channel; the model's correctness knowledge IS what it verbalizes (miscalibration is at the encoding level, not the readout level).
  - If |cos| ≈ 0.5 → intermediate; report honestly; neither pure interpretation supported.
  - If |cos| passes but neighborhood-robustness fails → L\* got lucky; C3a not supported.
- **Table / figure target**: Table 2 (main paper) — |cos| at L\* + neighborhood, with nulls; Fig 2 (main paper) — per-layer |cos| trajectory with CI band + probe AUROC curves overlaid.
- **Priority**: MUST-RUN
- **Estimated GPU-hours**: 0.2 h (pure CPU + probe re-fits; only counted as GPU for consistency)
- **method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`
- **depends_on**: [B1]

### Block B3: Causal-intervention steering — cross-direction with matched-magnitude random control (covers C3b)
- **Claim tested**: C3b (causal cross-coupling below matched-random baseline on internal readout primary, emitted output secondary).
- **Why this block exists**: promotes C3a from geometry to causal mechanism — the Causal Intervention direction of the mechanism strategy.
- **Dataset / split / task**:
  - Provenance: **existing**
  - Source: TriviaQA `rc.web` validation, held-out test split from B1
  - Available N: 2,000 test samples from B1
  - Planned used N: **500** (primary steering slice) + 200 (optional 6-α robustness slice)
  - Subset note: 500-sample slice chosen for compute — 4,500 total steered forward passes fit in ~1h GPU on Llama-3.1-8B; 500 is sufficient for the effect-size CIs the C3b test requires (SE for a proportion or mean at N=500 is 1/√500 ≈ 0.045).
- **Compared systems**:
  - Steer along v_c^{L\*} at layer L\* (measure Δ in probe_v_binary readout and in emitted confidence number under forward pass 2)
  - Steer along v_v^{L\*} at layer L\* (measure Δ in probe_c_binary readout and in emitted correctness under forward pass 1)
  - Steer along random_unit^{L\*} of matched σ_L\* norm (matched-magnitude control)
- **Metrics**:
  - Primary (internal readout): Δ probe readout under cross-direction vs. random-direction steering; C3b pass criterion (ratio OR absolute-effect) evaluated on this.
  - Secondary corroboration (emitted output): Δ verbalized confidence number (for v_v-steering pass 2) and Δ correctness rate (for v_c-steering pass 1) vs. random-direction control.
  - Reported with 95% bootstrap CIs over the 500 held-out samples + full distribution (violin/box).
- **Setup details**:
  - Grid: `α ∈ {−1σ_L*, 0, +1σ_L*} × direction ∈ {v_c^L*, v_v^L*, random_unit^L*} × 500 samples = 4500 forward passes`
  - σ_L\* = std of residual-stream activation magnitude at layer L\* on the training set
  - σ_probe_readout = std of unsteered held-out probe readout distribution at L\*
  - Steering site: HuggingFace forward hook on residual-stream output of block L\*; add `α × σ_L* × v` at last-input-token position
  - Perplexity safety cap: if steered generations show > 3× baseline mean token perplexity, halve α and re-run
  - No vllm (KV-cache reuse conflicts with per-token hooks); use HF `generate` with batch_size 32, `torch.compile`
- **Success criterion (C3b)**: EITHER `|Δ_v_other_steer| / |Δ_random_direction_steer| ≤ 1.5` when `|Δ_random| ≥ 0.5σ_probe_readout` OR `|Δ_v_other_steer| ≤ 0.5σ_probe_readout` when the ratio is undefined. Passes on internal readout PRIMARY in both cross directions; emitted-output SECONDARY corroboration where feasible.
- **Failure interpretation**:
  - If v_c steering DOES move probe_v readout by more than random → v_c and v_v are causally coupled → C3 refuted at the causal level even if geometry is orthogonal.
  - Asymmetry (v_v-steering moves emitted confidence strongly but v_c-steering does not shift correctness much) is expected and reported honestly: emitted-confidence is easier to shift than downstream correctness.
- **Table / figure target**: Table 3 (main paper) — cross-steering readout Δ + CI + pass/fail; Fig 3 — steering dose-response curves.
- **Priority**: MUST-RUN
- **Estimated GPU-hours**: 1.0 h (4500 steered passes at 20 max_new_tokens with batched HF generation)
- **method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`
- **depends_on**: [B1, B2]

### Block B4: Dissociation-when-disagree accuracy analysis (covers C3c)
- **Claim tested**: C3c (when probe_c and verbalized confidence disagree, verbalization loses reliability).
- **Why this block exists**: turns the static geometric claim into a downstream-actionable diagnostic. Supporting analysis, secondary priority.
- **Dataset / split / task**:
  - Provenance: **existing** (reuses B1's test predictions + verbalized confidences)
  - Source: TriviaQA `rc.web` held-out test 2k samples from B1
  - Available N: 2,000; Planned used N: 2,000
- **Compared systems**: N/A — descriptive analysis binning by (probe_c_calibrated_output, verbalized_conf)
- **Metrics**: Correctness rate per cell; McNemar test on the (low probe, high verbal) vs. (low probe, low verbal) contrast.
- **Setup details**: bins on probe_c_calibrated_output at median (or path-dependent threshold); bins on c at 50 (or Likert-equivalent).
- **Success criterion**: Accuracy(low probe, high verbal) < Accuracy(low probe, low verbal), McNemar p < 0.05.
- **Failure interpretation**: If no gap, C3c is not supported; C3 stands only on geometric+causal evidence.
- **Table / figure target**: Table 4 (appendix or main) — 2x2 accuracy table + McNemar stat.
- **Priority**: MUST-RUN (needed to complete the C3 chain in the ledger)
- **Estimated GPU-hours**: 0 h (analysis-only on cached data)
- **method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`
- **depends_on**: [B1]

### Block B5: Verbalized-confidence variance diagnostic + Stage-1.5 path decision
- **Claim tested**: Not a direct claim test — a gating diagnostic that determines the C2 primary probe path (continuous vs. ordinal) and feeds C2's operational threshold.
- **Why this block exists**: pre-registered contingency for the case where verbalized confidence clusters near 100 (which the task.md anchor itself predicts).
- **Dataset / split / task**:
  - Provenance: **existing** (reuses B1's collected c values)
  - Source: 10k TriviaQA questions, verbalized-confidence numbers c from forward pass 2
  - Available N: 10,000; Planned used N: 10,000
- **Compared systems**: N/A — descriptive statistics.
- **Metrics**: mean(c), std(c), entropy(c), quantiles, share(c≥95), share(c≤5), parseable-rate, parseable-vs-unparseable correctness gap.
- **Setup details**: computed on the 6k training slice; decision applies to all downstream C2 evidence.
- **Success criterion**: decision rule fires and selects continuous OR ordinal path (both valid, per Stage 1.5 spec).
- **Failure interpretation**: unparseable-rate ≥ 10% → switch to fallback P1 prompt (Tian 2023 style); parseable-vs-unparseable correctness gap > 5 percentage points → flag selection bias and report in paper.
- **Table / figure target**: Table 0 / early main paper — c distribution + Stage-1.5 decision.
- **Priority**: MUST-RUN (gates C2's operational metric)
- **Estimated GPU-hours**: 0 h (analysis-only on B1 cached outputs)
- **method_sensitive**: `[metric]`
- **depends_on**: [B1]

### Block B6: Robustness ablations — nulls + paraphrase (covers C1, C2, C3a)
- **Claim tested**: robustness of C1 and C2 to random-direction nulls, shuffled-label nulls, and paraphrase variants (P1, P2). Also feeds C3a stability.
- **Why this block exists**: reviewer-required nulls per round-1 and round-2 review to defend "the result is not overfitting or prompt-specific".
- **Dataset / split / task**:
  - Provenance: **existing / adapted** (existing TriviaQA data with re-elicited c under two paraphrase prompts)
  - Source: TriviaQA `rc.web` dev split from B1
  - Available N: 2,000 dev
  - Planned used N: **500** dev samples × 2 alternate prompts = 1,000 additional forward-pass-2 generations (paraphrase); 0 additional generations for nulls (analytic).
  - Subset note: 500 is sufficient for ΔSpearman/AUROC CIs at required precision, and keeps the block under 0.5h GPU.
- **Compared systems**:
  - Nulls (analytic): random-direction probe, shuffled-label probe (each C=probe_c_binary, C=probe_v_binary)
  - Paraphrase: P0 vs. P1 (Tian 2023: 0.0-1.0 scale) vs. P2 (Likert mapped to 20/40/60/80/100)
- **Metrics**:
  - Nulls: AUROC and cos vs. random-direction baseline expectations
  - Paraphrase: ΔSpearman ρ ≤ 0.1, Δbinarized AUROC ≤ 0.05, Δ|cos| ≤ 0.1
- **Setup details**: re-run forward pass 2 with P1 and P2 on 500 dev samples; extract H_2^L\*; re-fit probe_v_binary; compute cos with the original v_c^{L\*} from P0.
- **Success criterion**: nulls show random performance and cos ≈ theoretical null; paraphrase Δs within tolerances above.
- **Failure interpretation**: null failures = probe overfitting; paraphrase failures = C2's linear direction is prompt-specific (still a meaningful finding — reduces the generality of the C2/C3 claim to the tested prompt).
- **Table / figure target**: Appendix Table A1 — nulls; Appendix Table A2 — paraphrase Δs.
- **Priority**: MUST-RUN
- **Estimated GPU-hours**: 0.5 h (paraphrase forward passes)
- **method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`
- **depends_on**: [B1]

### Block B7: Single-pass unified-prompt robustness variant (covers C3a two-context concern)
- **Claim tested**: robustness of C3a to the two-context objection — does |cos| stay low under a single unified elicitation prompt?
- **Why this block exists**: pre-registered response to round-2 reviewer's "two-context integration point" concern; directly addresses the strongest representation-space objection.
- **Dataset / split / task**:
  - Provenance: **existing**
  - Source: TriviaQA `rc.web` validation, same 10k as B1
  - Available N: 10,000; Planned used N: 10,000
- **Compared systems**: probe_c_single + probe_v_single trained on H_single^L (unified prompt: "Answer with your best guess then confidence (0-100).") vs. probe_c + probe_v from B1 (two-pass).
- **Metrics**: |cos(v_c_single^{L\*}, v_v_single^{L\*})| with bootstrap CI; AUROC of each single-pass probe.
- **Setup details**: run one additional forward pass with the unified prompt for all 10k questions; extract H_single^L at last-input-token; re-fit probes; compute cos.
- **Success criterion (pre-registered)**:
  - If two-pass |cos| ≤ 0.3 AND single-pass |cos| ≤ 0.3 → strengthens the dissociation interpretation.
  - If two-pass |cos| ≤ 0.3 AND single-pass |cos| > 0.5 → main claim is restricted to two-context; no single-shared-state conclusion.
  - Both values reported honestly.
- **Failure interpretation**: both |cos| high → C3a not supported. Only two-context |cos| high → the paired-sample design gives a lower cosine than a joint elicitation would.
- **Table / figure target**: Main paper Table 2 additional row OR Appendix Table A3.
- **Priority**: MUST-RUN
- **Estimated GPU-hours**: 1.0 h (10k forward passes + hook extraction + probe re-fits)
- **method_sensitive**: `[n_pairs, sites, metric, gpu_hours]`
- **depends_on**: [B1]

---

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost (GPU-h) | Risk |
|-----------|------|------|---------------|--------------|------|
| M1 (B1)   | Hidden-state + label collection (both forward passes) + probe training | R001 (pass 1 gen), R002 (pass 1 hook), R003 (pass 2 gen), R004 (pass 2 hook), R005 (probe fits + bootstrap) | Probe_c AUROC(L_c*) ≥ 0.70 AND probe_v_primary threshold met — else STOP and report negative C1/C2 | 2.5 | Answer scoring quality; probe_v trivial-readout; parseable-rate |
| M1.5 (B5) | Stage-1.5 variance diagnostic + path selection | R006 (analysis) | Continuous or ordinal path selected; parseable-rate ≥ 90% — else switch to fallback prompt P1 | 0 | Confidence values may cluster at 100 |
| M2 (B2)   | Cosine + per-layer trajectory + bootstrap CIs | R007 (probe re-fits for bootstrap cos), R008 (per-layer curves) | \|cos(v_c^L*, v_v^L*)\| computed with CI; if C1 & C2 gates failed at M1, C3a is not interpretable (report as such) | 0.2 | L\* selection sensitivity |
| M3 (B3)   | Cross-direction activation steering | R009 (steering: 4500 passes) | Cross-direction Δ vs. random-direction Δ meets ratio-or-absolute criterion on internal readout | 1.0 | Perplexity blow-up; ratio instability (handled by absolute-effect fallback) |
| M4 (B4)   | Dissociation-when-disagree analysis | R010 (2x2 accuracy table + McNemar) | Statistical significance of the (low-probe, high-verbal) cell being least accurate | 0 | Cell counts may be small if verbalized c skews high |
| M5 (B6)   | Robustness ablations — nulls + paraphrase | R011 (null probes analytic), R012 (P1 forward pass 2), R013 (P2 forward pass 2), R014 (probe re-fits for P1/P2) | Nulls at chance; paraphrase Δ within tolerance | 0.5 | Model may refuse paraphrase P2 Likert format |
| M6 (B7)   | Single-pass unified-prompt robustness | R015 (unified-prompt forward pass), R016 (hook extraction), R017 (single-pass probe fits + cos) | Report both two-pass and single-pass \|cos\|; predetermined interpretation applies | 1.0 | Model may not naturally emit the "confidence: X" tail |

**Total estimated GPU-hours (main experiment)**: 5.2 h (M1 + M2 + M3 + M4 + M5 + M6). Buffer under 10h HARD budget: ~4.8 h remaining for verify + iteration.

**Sanity stage** before M1: implement, dry-run on 100 samples, sanity-check answer scoring against 20 manual labels, sanity-check confidence-parsing on 50 samples. (~15 min human + ~0.05 h GPU; not counted separately.)

---

## Compute and Data Budget

- **Total estimated GPU-hours (main experiment)**: ~5.2 h
- **Verify stage (see Verify Suggestions below)**: expected ~2 h
- **Buffer for iteration / re-runs / integrity fixes**: ~2.8 h
- **HARD budget**: 10 h — plan fits comfortably
- **Data preparation needs**: TriviaQA already local (`/data/zhenqian/data`); model already local (`/data/zhenqian/models/Llama-3.1-8B-Instruct`); 100-sample manual label-quality verification (1 person-hour)
- **Human evaluation needs**: 100-sample correctness-scoring audit only
- **Biggest bottleneck**: steering pass (M3) — 4500 generations at ~1s each with HF batched generation

---

## Verify Suggestions (feeds `/auto-verify` Workflow 1.75)

Do NOT preempt `/auto-verify`'s swap selection — the plan below is a suggestion pool for it to draw from. `/auto-verify` may run 0, 1, 2, or 3 variants based on its `MAX_VERIFY_CLAIMS` and per-claim importance ranking.

- **Highest-priority claim to stress-test**: C3a (the load-bearing geometric claim).
- **Suggested swap variants (candidate pool, use as needed, not necessarily all)**:
  1. **Model swap — Llama-3.1-8B base (non-Instruct)**: strongest test — task.md's readout-failure story predicts C3 will WEAKEN or invert on the base model (which does not have the RLHF-induced verbalized-confidence-inflation channel). Priority for /auto-verify's admission set.
  2. **Model swap — Qwen2.5-7B-Instruct**: cross-family generalization of the C3 result within RLHF-tuned models.
  3. **Model swap — Mistral-7B-Instruct-v0.1**: second cross-family generalization.
  4. **Dataset swap — TruthfulQA**: within-domain (factual QA with known confidence gaps); a natural test of whether the calibration/verbalization directions are TriviaQA-specific or broadly hold.
  5. **Dataset swap — MMLU**: multiple-choice format changes the verbalization channel (choose-a-letter vs. free-form answer + confidence); tests whether the C3 result generalizes to a structurally different QA format.
- **NOT recommended for a first swap round**: MATH (arithmetic answers may not have well-defined verbalized-confidence semantics in Llama-3.1-8B-Instruct without additional prompting engineering; leave for a later round).
- **Model swap candidate pool (from task.md)**: {Llama-3.1-8B, Qwen2.5-7B, Qwen2.5-7B-Instruct, Mistral-7B-v0.1, Mistral-7B-Instruct-v0.1} — 5 candidates.
- **Dataset swap candidate pool (from task.md)**: {MATH, MMLU, TruthfulQA} — 3 candidates.

---

## Risks and Mitigations

- **Risk 1: Verbalized confidence clusters at 100 → C2 signal too weak.**
  - Mitigation: Stage 1.5 pre-registered decision rule (M1.5) automatically switches to ordinal probe on 4 bins with binarize-at-30th-percentile for v_v extraction.
- **Risk 2: Model refuses / fails to emit a parseable number in ≥ 10% of samples.**
  - Mitigation: fallback to Tian 2023 prompt style (P1, 0.0-1.0 scale) — detected and swapped in M1.5.
- **Risk 3: probe_v_primary AUROC ≥ 0.95 → probe trivially reads the pre-emission committed number.**
  - Mitigation: report as a finding, not a failure. C3 can still hold geometrically.
- **Risk 4: L\* is a lucky isolated layer; |cos| oscillates broadly across neighborhood.**
  - Mitigation: neighborhood-robustness guardrail — C3a is considered supported only if the mean |cos| across L\*±2 layers ≤ 0.3. Failing this narrows the C3a claim to "at this specific layer" with reduced confidence.
- **Risk 5: Cross-steering ratio criterion unstable when random-direction Δ is near zero.**
  - Mitigation: absolute-effect companion criterion (|Δ_v_other| ≤ 0.5σ_probe_readout).
- **Risk 6: Steering catastrophically breaks generation (perplexity blows up).**
  - Mitigation: perplexity safety cap at 3× baseline; halve α and re-run.
- **Risk 7: TriviaQA alias scoring diverges from human judgment.**
  - Mitigation: 100-sample manual audit; if agreement < 90%, expand alias set or add a second-pass string-matching heuristic.
- **Risk 8: Two-context cosine numerically differs materially from single-pass cosine.**
  - Mitigation: pre-registered interpretation (block B7) — main claim restricted to two-context; single-pass reported alongside honestly.
- **Risk 9: GPU budget overrun (steering + probe re-fits exceed estimate).**
  - Mitigation: 4.8 h buffer; if exceeded on steering, reduce dose grid to {−1σ, +1σ} (skip 0-α baseline reuse), or reduce N=500→N=300.

---

## Final Checklist

- [x] Main paper tables covered: Table 1 (probe accessibility), Table 2 (geometry), Table 3 (causal steering), Table 4 (dissociation)
- [x] Novelty isolated: matched-pair canonical-hook framing + neighborhood-robustness + two-context robustness variant
- [x] Simplicity defended: only two probe types + linear steering; SAE / formation tracing / fine-tuning explicitly rejected
- [x] Frontier contribution justified: linear probes + steering are field-standard; the innovation is the matched-pair characterization, not the primitives
- [x] Nice-to-have separated from must-run: token-position ablation, calibration-method comparison, question-only baseline, post-emission upper bound, 6-α robustness slice → all appendix/optional
- [x] Every mechanism/intervention milestone carries `method_sensitive` (routing-stage rebinding allowed)
- [x] All milestones tagged by claim(s) they cover
- [x] Verify suggestions provided as a candidate pool, not a preempted plan
- [x] Top metadata: `behavior_source: given`, `mechanism: discovery`, `mechanism_strategy:` block, NO `resource_fidelity: strict`
- [x] Plan opens directly with mechanism milestones — NO M0 gate (behavior_source=given)
