# Research Proposal (round 0): Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence in Llama-3.1-8B-Instruct

## Problem Anchor

- **Bottom-line problem**: RLHF-tuned LLMs verbalize confidence scores clustered near 100% regardless of actual correctness, misleading downstream users. Prior work has separately shown (a) that internal states linearly encode correctness/truthfulness (Marks-Tegmark 2023, Azaria-Mitchell 2023, Orgad 2024) and (b) that verbalized confidence is a controllable signal (Kadavath 2022, Tian 2023). What remains uncharacterized is the **geometric relationship** between the internal calibration direction and the internal verbalization direction — is the model's poor verbalized calibration a *knowledge deficit* (both signals collapsed onto one distorted direction) or a *readout failure* (calibration is intact but written on a direction orthogonal to the one that produces the verbal number)?
- **Must-solve bottleneck**: We must produce a single-model, single-dataset **matched-pair** measurement that quantifies (i) the linear accessibility of gold correctness (C1), (ii) the linear accessibility of verbalized confidence (C2), and (iii) the geometric angle *and* causal separability between the two directions (C3), on Llama-3.1-8B-Instruct evaluated on TriviaQA. No prior published paper reports this angle with matched controls.
- **Non-goals**: (i) proposing a new probing algorithm — vanilla logistic regression on residual-stream activations is deliberately used, since the field's consensus is that linear probes suffice; (ii) training a new mechanism family (no SAE decomposition, no formation-tracing, no fine-tuning) — the geometric-angle claim is fully testable at inference time with existing techniques; (iii) fixing miscalibration — this is a *characterization* paper, not an intervention paper; (iv) generalizing to other domains — we deliberately restrict to TriviaQA to keep C3's cosine-similarity number interpretable (per Kim et al. 2025, truth directions are task-specific, so cross-task measurements would confound the C3 test).
- **Constraints**: 10h total GPU budget across the whole project (probing + generating answers + collecting hidden states + steering runs). GPU ids ∈ {1, 2, 3, 5, 6}. Filesystem restricted to work_dir + `/data/zhenqian/data` + `/data/zhenqian/models`. Conda env `belief`. vllm allowed for batched generation. Verify swaps within {Llama-3.1-8B (base), Qwen2.5-7B{-Instruct}, Mistral-7B-v0.1{-Instruct-v0.1}} × {MATH, MMLU, TruthfulQA} — use as needed, not all.
- **Success condition**: We can report a paper whose main table contains a matched (best_layer_C1, best_layer_C2) pair for Llama-3.1-8B-Instruct on TriviaQA with (i) AUROC_C1 ≥ 0.70 with ECE ≤ 0.10 beating the token-probability baseline, (ii) Spearman ρ_C2 ≥ 0.5 with binarized AUROC_C2 ≥ 0.70 on pre-emission hidden states, and (iii) `|cos(v_c, v_v)| ≤ 0.3` at those matched layers *with* a causal-steering test showing that steering along v_c does not move v_v's readout beyond a random-direction null and vice versa, *plus* a dissociation-when-disagree accuracy drop. If all three land, the paper contains the first direct evidence for the readout-failure interpretation of RLHF-induced overconfidence.

## Technical Gap

Two separate lines exist. Neither directly measures the angle:

1. **Correctness-probing line** — Marks-Tegmark 2023 shows linear T/F structure in Llama activations, Azaria-Mitchell 2023 (SAPLMA) trains classifiers on hidden states with 71-83% AUROC, Orgad 2024 shows the signal is token-concentrated but dataset-fragile, Liu et al. 2024 (Universal Truthfulness Hyperplane) shows a shared direction only emerges with dataset diversity, and The Confidence Manifold 2026 shows the correctness subspace is 3-8D. None of these papers measure the alignment with the verbalization channel.

2. **Verbalized-confidence line** — Kadavath 2022 (P(True)), Lin 2022 (train to verbalize), Tian 2023 (better-than-token-prob calibration via prompt), Yang 2024 (verbalized confidence is prompt-sensitive), Kossen 2025 (verbal-uncertainty is a *single* linear feature, only moderately correlated with semantic uncertainty), Zhang 2025 (DPO can partially align verbal ↔ token-prob confidence). Kossen 2025 comes closest — it establishes that verbal uncertainty is linear — but reports only a scalar Pearson correlation between two output-space quantities (semantic vs. verbal uncertainty scores), not a geometric angle between two probe direction vectors, and does not compare the *pre-emission* verbal-uncertainty direction against a *gold-correctness* probe direction on the same hidden states.

The gap that determines whether verbalized miscalibration is *knowledge deficit* vs. *readout failure* is the pairwise **geometric and causal** relationship between (v_c, v_v). If |cos(v_c, v_v)| ≈ 1 and steering v_c moves v_v's readout in the same way, that's a knowledge-deficit story (the two are the same channel, and the channel is miscalibrated). If |cos(v_c, v_v)| ≈ 0 and steering v_c does not move v_v's readout beyond a random-direction control, that's a readout-failure story (the model has calibration information available, but that information does not reach the verbalization channel). This distinction has direct downstream consequences: readout-failure implies a geometric fix (project v_v onto v_c's subspace at inference time, or DPO on verbal-confidence tokens as in Zhang 2025) suffices, while knowledge-deficit implies re-training is required.

## Method Thesis

- **One-sentence thesis**: On Llama-3.1-8B-Instruct evaluated on TriviaQA, we train two independent linear probes on paired residual-stream hidden states — one predicting gold correctness of the model's free-form answer, one predicting the numeric confidence the model will subsequently verbalize — and measure their per-layer AUROC / ECE, their pairwise cosine similarity, and their causal separability under matched-magnitude activation steering, thereby producing the first direct evidence that gold calibration and verbalized confidence occupy separate, nearly orthogonal linear subspaces.
- **Why this is the smallest adequate intervention**: linear probes + linear steering are the field's simplest tools for both directions. No new architecture. No new training regime. No SAE. The only novelty is the **matched-pair paired-sample design** and the **direct measurement of the angle** (both geometric via cosine and causal via cross-steering), which is a bookkeeping and controls story, not a modeling story.
- **Why this route is timely**: the 2025 wave (Kossen, Zhang, HACK, Calibration-Across-Layers) has just started asking the dissociation question but has not measured its geometry. This paper crystallizes the dissociation into a single geometric number and its per-layer trajectory — a claim the field can build on (e.g., for readout-only interventions).

## Contribution Focus

- **Dominant contribution**: A matched-pair, controls-heavy characterization of the geometric angle and causal separability between the gold-correctness and verbalized-confidence directions in one model on one dataset — the direct measurement no prior paper reports.
- **Optional supporting contribution**: The **dissociation-when-disagree** accuracy analysis — when the internal correctness probe and the verbalized confidence disagree, the verbalization is materially less reliable than when they agree. This turns C3 from a static geometric claim into an actionable *usable signal* for downstream trust calibration.
- **Explicit non-contributions**:
  - No new probing method (deliberate: field uses logistic regression, so should we)
  - No new steering algorithm (deliberate: ITI-style steering, per Li et al. 2023)
  - No fine-tuning / DPO / RLHF (Zhang 2025 already does that)
  - No SAE decomposition (would double the work with no additional evidence for C3)
  - No formation tracing (would double the work; genesis is a follow-up paper)
  - No cross-dataset generalization of the angle (Kim et al. 2025 established truth directions are task-specific; cross-task would confound C3)
  - No mechanistic circuit discovery (too broad; C3 is about direction-level structure, not circuit-level)

## Proposed Method

### Complexity Budget

- **Frozen / reused backbone**: Llama-3.1-8B-Instruct — never touched, only forward-passed and (in the steering runs) residual-stream-activation-hooked.
- **New trainable components**: two linear probes (each a logistic regression: 4096-dim input → 1 scalar), per layer swept. No new modules. Well under `MAX_NEW_TRAINABLE_COMPONENTS = 2`.
- **Tempting additions intentionally not used**:
  - Nonlinear probes (2-layer MLP) — reject: field consensus is linear ≥ nonlinear for these signals (Confidence Manifold 2026 shows linear ≥ nonlinear on correctness).
  - SAE decomposition — reject: the geometric-angle claim is fully testable at direction level.
  - Multi-token-position probing — reject: use last-input-token position (before answer generation for C1's forward-pass; before "confidence: X" token position for C2), plus a robustness ablation on token position.
  - PCA-selected multi-dimensional subspaces beyond a single direction — reject in main experiment; include as one sensitivity ablation.

### System Overview

```
                     ┌─── forward pass 1 (answer generation) ────┐
                     │                                            │
TriviaQA question ──► Llama-3.1-8B-Instruct                       │
                     │  (a) collect residual-stream hidden state  │
                     │      at last input token, all 32 layers    │
                     │  (b) generate the free-form answer         │
                     │  (c) score answer against gold aliases     │──► correctness label y_c ∈ {0, 1}
                     │                                            │
                     └────────────────────────────────────────────┘

                     ┌─── forward pass 2 (verbalized confidence) ─┐
                     │                                            │
question + answer ──► Llama-3.1-8B-Instruct                       │
   + "How confident │  (a) collect residual-stream hidden state  │
   are you? Give a  │      at last input token (before conf.),   │
   number 0-100"    │      all 32 layers                          │
                     │  (b) generate a number 0-100               │──► verbalized confidence c ∈ [0, 100]
                     │  (c) parse the number                      │
                     └────────────────────────────────────────────┘

Paired-sample dataset:  (H_1^L, y_c, H_2^L, c)_i for each sample i, at each layer L.

┌─── Location: two linear probes per layer, cross-validated ────┐
│  probe_c(H_1^L) → predicted correctness  (target = y_c)       │
│  probe_v(H_2^L) → predicted verbalized confidence (target = c)│
│  Report AUROC_c(L), ECE_c(L), Spearman ρ_v(L), AUROC_v(L)     │
│  Extract direction vectors v_c^L and v_v^L (probe weights,    │
│  L2-normalized)                                                │
└────────────────────────────────────────────────────────────────┘

┌─── Geometric orthogonality (C3a) ─────────────────────────────┐
│  cos_sim(v_c^L, v_v^L) at every layer L                       │
│  Report at matched best-AUROC layers                          │
│  Null: cos_sim(v_c^L, random_direction_of_same_norm)          │
└────────────────────────────────────────────────────────────────┘

┌─── Causal separability (C3b) — cross-steering ────────────────┐
│  For α ∈ {-2σ, -1σ, -0.5σ, 0, +0.5σ, +1σ, +2σ} of the         │
│  direction's activation-magnitude distribution:                │
│    (i) Steer forward pass 2 along v_c at best-C1 layer;       │
│        measure Δ(probe_v readout) at best-C2 layer            │
│    (ii) Steer forward pass 2 along v_v at best-C2 layer;      │
│        measure Δ(probe_c readout) at best-C1 layer            │
│    (iii) Same but with a random unit direction of matched norm │
│  C3b holds iff |Δ_cross_steer_v_c| ≤ 1.5 × |Δ_random|         │
│  and |Δ_cross_steer_v_v| ≤ 1.5 × |Δ_random|                    │
└────────────────────────────────────────────────────────────────┘

┌─── Dissociation-when-disagree (C3c) ──────────────────────────┐
│  Bin samples by (probe_c_output, verbalized_confidence)       │
│  Compare accuracy in "probe says low, verbalization says high"│
│  vs. "both agree low" vs. "both agree high"                   │
└────────────────────────────────────────────────────────────────┘
```

### Core Mechanism

- **Input / output**:
  - **Forward pass 1** input: TriviaQA question. Output: hidden states H_1^L ∈ R^{4096} at each of the 32 residual-stream layer positions, at the last-input-token position (before answer generation). Also: the model's free-form answer (max 20 tokens, greedy decode) → scored via TriviaQA aliases → correctness label y_c ∈ {0,1}.
  - **Forward pass 2** input: same question + model's own answer + `"How confident are you that this answer is correct? Give a probability from 0 to 100 as a single number.\nConfidence:"`. Output: hidden states H_2^L at last-input-token (immediately before the confidence number is emitted), plus the emitted number c ∈ [0,100].
- **Architecture**: two independent logistic regressions with L2 regularization. Weights extracted as direction vectors v_c^L (from probe_c weights, L2-normalized) and v_v^L (from probe_v_binarized weights, L2-normalized). For probe_v we also fit a *continuous* linear regression to get a Spearman ρ estimator; the *direction vector* used for C3 comes from a binarized-at-median logistic regression, so both direction vectors are unit-length and directly comparable.
- **Training signal / loss**: logistic-regression cross-entropy for probe_c (y_c ∈ {0,1}) and for probe_v_binarized (indicator [c > median(c_train_set)]); mean-squared error for the auxiliary continuous probe_v used for the Spearman ρ report.
- **Why this is the main novelty**: the paired-sample design — same question, forward pass 1 for correctness, forward pass 2 for verbalization — ensures we measure two directions on activations from *the same input distribution*, so the cosine similarity between the direction vectors is a well-defined geometric quantity, not confounded by different data distributions.

### Optional Supporting Component

- **Dissociation-when-disagree analysis**: bin the held-out set on (probe_c_output_calibrated, verbalized_confidence) — four cells: (low, low), (low, high), (high, low), (high, high). Report accuracy per cell. Prediction: (low, high) accuracy < (low, low) accuracy < (high, low) accuracy < (high, high) accuracy. This is a small analysis on top of the main table, not a separate contribution — but it turns the geometric claim into a downstream-usable signal.

### Modern Primitive Usage

- **vllm** — batched generation (forward pass 1 + forward pass 2 answers + confidence numbers) for compute efficiency. vllm serves as an *inference-time throughput* primitive; it does not enter the paper's claim.
- **HuggingFace transformers** for hidden-state extraction (vllm does not expose per-layer residual-stream states cleanly). One-pass extraction + parallel logistic-regression training on cached activations.
- **No LLM-as-judge / no external LLM in the loop** — TriviaQA correctness is a well-defined normalized-alias exact-match; using an LLM judge would introduce a second noisy signal.

### Integration into Base Generator / Downstream Pipeline

- The Llama-3.1-8B-Instruct model is used strictly at inference time. Two forward passes per sample. Hidden states cached to disk (approx `n_samples × 32 layers × 4096 dim × 2 passes × float16 ≈ 8k × 32 × 8KB × 2 ≈ 4 GB`, well within budget).
- Probes trained CPU-side or on any spare GPU (cost negligible).
- Steering runs: hook the residual stream at layer L_c or L_v, add α × v to the last-input-token position, run to completion, log the emitted confidence number and the pre-emission hidden state.

### Training Plan

- **Data construction**: 10k TriviaQA `rc.web` validation-set questions (the standard closed-book QA slice). Split 6k train probe / 2k dev probe / 2k test probe. Only questions Llama can complete in ≤ 20 tokens (drop long-answer questions). Aliases from TriviaQA release for scoring.
- **Stage 1 — Hidden-state + label collection**: run forward passes 1 & 2 on all 10k samples, cache H_1^L and H_2^L for all 32 layers, y_c, c. Cost: ~2 h wall on one GPU with vllm-batched generation (the hidden-state extraction requires HF transformers for the extraction pass but generation uses vllm).
- **Stage 2 — Probe training**: 32 layers × 2 probes = 64 logistic regressions on 6k train examples, evaluated on 2k dev + 2k test. Cost: minutes on CPU.
- **Stage 3 — Geometric measurement**: compute cos(v_c^L, v_v^L) at every layer. Cost: negligible.
- **Stage 4 — Causal steering**: at (L_c*, L_v*) the argmax-AUROC layers, run steering forward passes on the 2k test split at 6 α values × 3 directions (v_c, v_v, random) = 18 conditions × 2k = 36k forward passes. Cost: ~3 h with vllm.
- **Stage 5 — Analyses & robustness ablations**: paraphrase-robustness of the confidence prompt, shuffled-label null, random-direction null, token-position ablation, per-layer trajectory of |cos|.
- Total main-experiment cost estimate: ~6 h GPU. Leaves ~4 h for verify-stage swaps.

### Failure Modes and Diagnostics

- **Failure mode 1: Probe_v trivially reads back the emitted confidence token from the pre-emission hidden state (i.e., the model has already committed internally to a specific number before emission).** *Detect*: probe_v AUROC binarized-at-median ≥ 0.95 would be a red flag that "confidence" is just a shallow copy of an internal commitment. *Fallback*: report the finding (still novel: even if the pre-emission state contains the exact number, C3 still holds if v_v ⊥ v_c geometrically). No mitigation needed; this is a *finding*, not a failure.
- **Failure mode 2: The model refuses to give a confidence number.** *Detect*: parseable-number rate < 90% on validation. *Fallback*: switch to the Tian-2023 prompt style ("Give a probability from 0.0 to 1.0"), the top-k logit distribution ("What is P(True)?"), or a discrete Likert prompt ("choose one: very low / low / medium / high / very high" → map to 20/40/60/80/100). Fallback prompts are pre-registered before running.
- **Failure mode 3: |cos| is near 0 only because both probes are weak.** *Detect*: enforce C1/C2 AUROC thresholds as gates before reporting C3a. If either probe fails to hit its AUROC target, we report the negative finding for that claim but do *not* interpret |cos| as evidence of orthogonality.
- **Failure mode 4: TriviaQA gold-correctness ambiguity on open-form answers.** *Detect*: manual inspection of a random 100 disagreements. *Fallback*: use TriviaQA-provided aliases + Wikidata alias union + normalized string match (lowercase, strip punctuation, remove leading article). Report per-sample pairwise-annotator agreement (κ) via 2 independent normalizations for a 500-sample subset.
- **Failure mode 5: Steering catastrophically breaks the model (post-steer generations become nonsense).** *Detect*: on the 0-α baseline, average token-level perplexity should not increase > 10× under steering. *Fallback*: reduce max |α| to the largest coefficient at which perplexity increase is < 3×.

### Novelty and Elegance Argument

- **Closest work**: Kossen 2025 ("Calibrating Verbal Uncertainty as a Linear Feature") — measures a single linear feature for verbal uncertainty and reports only a *scalar Pearson r* between semantic and verbal uncertainty. **Exact difference**: (a) Kossen 2025 compares two *output-space* scalars (semantic uncertainty vs. verbal uncertainty); we compare two *representation-space* direction vectors (v_c gold-correctness direction vs. v_v verbalized-confidence direction) and report their cosine similarity, i.e., the underlying *geometric* structure that Kossen's scalar correlation is a downstream projection of. (b) Kossen 2025 does not report per-layer trajectories; we do. (c) Kossen 2025 does not report cross-direction steering with a random-direction null; we do. (d) Kossen 2025 uses *semantic uncertainty* (a sampling-based measure) as the "internal" side; we use *gold correctness* (the ground truth), which cleanly separates the *knowledge* signal from the *sampling variance* signal.
- **Why focused, not module-piled**: the paper contains one experimental table (per-layer AUROC/ECE/cos for C1 + C2 + C3a), one causal table (cross-steering + dissociation-when-disagree for C3b + C3c), and standard ablations. No new components. The dominant contribution is a matched-pair *measurement*, elegant enough to explain in a figure.

## Claim-Driven Validation Sketch

### Claim 1 (C1): Gold correctness is linearly accessible in Llama-3.1-8B-Instruct hidden states on TriviaQA
- **Minimal experiment**: layer-swept logistic regression on H_1^L predicting y_c, 6k train / 2k dev / 2k test.
- **Baselines / ablations**: (a) token-probability of the answer span (model's own default calibration signal), (b) P(True) self-elicitation prompt, (c) random-direction probe null, (d) shuffled-label null.
- **Metric**: AUROC + ECE (with isotonic post-calibration on dev, evaluated on test) at the best-AUROC layer.
- **Expected evidence**: AUROC ≥ 0.70 at some layer, ECE ≤ 0.10 post-calibration, ECE(probe) < ECE(token_prob).

### Claim 2 (C2): Verbalized confidence is linearly accessible in Llama-3.1-8B-Instruct hidden states on TriviaQA (pre-emission)
- **Minimal experiment**: layer-swept logistic regression on H_2^L (pre-emission) predicting binarized-at-median [c > median]; auxiliary linear regression for Spearman ρ.
- **Baselines / ablations**: (a) post-emission last-token hidden state (should be near-perfect readout — trivial upper bound), (b) input-token-only hidden state before question (should be near-chance — lower bound), (c) paraphrased confidence-elicitation prompt (Tian 2023 style), (d) shuffled-label null.
- **Metric**: Spearman ρ (continuous), binarized AUROC.
- **Expected evidence**: Spearman ρ ≥ 0.5, binarized AUROC ≥ 0.70 at some layer, robust to paraphrase (Δρ ≤ 0.1 across prompt variants).

### Claim 3 (C3): The two directions are separate (cos ≈ 0), causally separable (cross-steer null), and dissociate-when-disagree
- **C3a Minimal experiment**: compute cos(v_c^L, v_v^L) at every layer; report at layers where C1 and C2 achieve their best AUROC.
- **C3a Baselines / ablations**: cos(v_c^L, random_unit_direction) — should be ≈ 0 with std √(1/4096) ≈ 0.016 (Gaussian direction concentration); cos(v_c^L, v_v^L) shuffled-labels — should also concentrate near 0. The claim is that the *actual* cos is near 0 while the two probes each independently achieve their AUROC thresholds.
- **C3a Metric**: |cos| at best-layer pair. Test: `|cos(v_c^{L_c*}, v_v^{L_v*})| ≤ 0.3` AND C1 AUROC-gate passed AND C2 AUROC-gate passed.
- **C3b Minimal experiment**: cross-steering at 6 α values × 3 direction types (v_c, v_v, random) × 2k test samples.
- **C3b Baselines / ablations**: random-direction steering of matched norm.
- **C3b Metric**: `|Δ(probe_readout under v_other steering)| / |Δ(probe_readout under matched random-direction steering)| ≤ 1.5`.
- **C3c Minimal experiment**: bin held-out samples into (probe_c high/low) × (verbalized_conf high/low), report accuracy.
- **C3c Metric**: accuracy(probe_c low ∧ verbalized_conf high) < accuracy(probe_c low ∧ verbalized_conf low), with a paired-sample statistical test.

## Experiment Handoff Inputs

- **Must-prove claims**: C1, C2, C3a, C3b, C3c (all three sub-clauses of C3).
- **Must-run ablations**:
  1. Random-direction probe null (for both C1 and C2)
  2. Shuffled-label probe null (for both C1 and C2)
  3. Paraphrase robustness of verbalized-confidence elicitation prompt (for C2 and C3)
  4. Per-layer trajectory of |cos(v_c^L, v_v^L)| (for C3a)
  5. Matched-magnitude random-direction steering control (for C3b)
  6. Token-position ablation (last-input-token vs. answer-first-token vs. answer-last-token) — for probe stability
  7. Post-hoc calibration ablation (isotonic vs. Platt vs. temperature scaling) — for C1's ECE claim
  8. Best-layer-window robustness (report cos not just at argmax but at top-3-AUROC layers of each probe)
- **Critical datasets / metrics**: TriviaQA `rc.web` validation, AUROC, ECE, Spearman ρ, cosine similarity, dissociation-cell accuracy.
- **Highest-risk assumptions**:
  1. TriviaQA correctness scoring reflects the model's "actual correctness" (mitigation: alias-list + manual agreement audit on 100 samples).
  2. The pre-emission hidden state does not trivially encode the exact upcoming confidence number (mitigation: report if AUROC is > 0.95 as a positive finding rather than a failure).
  3. The 6k / 2k / 2k split is representative of TriviaQA (mitigation: stratified sampling by question type / answer length).

## Compute & Timeline Estimate

- **Total GPU-hours**: ~6 h main experiment (2 h collection + 3 h steering + 1 h ablations), + ~3 h verify swaps = ~9 h total, within the 10h HARD budget.
- **Data / annotation cost**: none new; TriviaQA + aliases already in the dataset. ~1 hour of human review to sanity-check 100 correctness scores.
- **Timeline**: 1 day implementation + 1 day main experiment + 1 day verify + iteration buffer = <1 week wall time.
