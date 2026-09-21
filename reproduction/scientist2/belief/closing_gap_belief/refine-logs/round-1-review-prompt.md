You are a senior ML reviewer for a top venue (NeurIPS/ICML/ICLR). This is an early-stage, method-first research proposal.

**IMPORTANT MODE NOTE**: this is a *given-claim verification* project — the three claims (C1, C2, C3) are FIXED by task.md and MUST NOT be modified, weakened, strengthened, or re-scoped. Your review should assess whether the **testing method** cleanly and rigorously tests those specific claims — not whether the claims themselves are worth pursuing (that decision is upstream).

Your job is NOT to reward extra modules, contribution sprawl, or a giant benchmark checklist.
Your job IS to stress-test whether the proposed testing method:
(1) still tests the original three anchored claims exactly as stated (no drift, no re-scoping),
(2) is concrete enough to implement in ~10 hours of GPU compute,
(3) presents a focused, elegant characterization (matched-pair measurement, not a new algorithm),
(4) uses appropriate techniques (linear probes + steering are field-standard for this claim structure).

Review principles:
- Prefer the smallest adequate mechanism over a larger system.
- Penalize parallel contributions that make the paper feel unfocused.
- Do NOT propose adding SAEs, formation tracing, or fine-tuning — those are explicitly rejected as non-goals in the anchor.
- Do NOT ask for extra experiments unless they are needed to prove the three specific claims.
- Do not ask to change the claims — they are given.

=== PROPOSAL ===

# Research Proposal (round 0): Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence in Llama-3.1-8B-Instruct

## Problem Anchor

- **Bottom-line problem**: RLHF-tuned LLMs verbalize confidence scores clustered near 100% regardless of actual correctness. Prior work has separately shown (a) internal states linearly encode correctness/truthfulness (Marks-Tegmark 2023, Azaria-Mitchell 2023, Orgad 2024) and (b) verbalized confidence is a controllable signal (Kadavath 2022, Tian 2023). What remains uncharacterized is the geometric relationship between the internal calibration direction and the internal verbalization direction — is verbalized miscalibration a knowledge deficit or a readout failure?
- **Must-solve bottleneck**: A single-model, single-dataset matched-pair measurement quantifying (i) linear accessibility of gold correctness (C1), (ii) linear accessibility of verbalized confidence (C2), and (iii) the geometric angle + causal separability between the two (C3), on Llama-3.1-8B-Instruct + TriviaQA. No prior paper reports this angle.
- **Non-goals**: no new probing algorithm, no SAE, no formation tracing, no fine-tuning, no cross-dataset generalization of the angle. This is a characterization paper, not an intervention paper.
- **Constraints**: 10h total GPU budget, GPU ids ∈ {1,2,3,5,6}, filesystem-restricted, Llama-3.1-8B-Instruct + TriviaQA mandatory, verify swaps bounded.
- **Success condition**: matched (best_layer_C1, best_layer_C2) with C1 AUROC≥0.70 + ECE≤0.10 + ECE(probe)<ECE(token_prob); C2 Spearman ρ≥0.5 + binarized AUROC≥0.70 on pre-emission; C3 |cos|≤0.3 + cross-steering null + dissociation-when-disagree.

## Method Thesis

On Llama-3.1-8B-Instruct evaluated on TriviaQA, train two independent linear probes on paired residual-stream hidden states — one predicting gold correctness of the model's free-form answer, one predicting the numeric confidence the model will subsequently verbalize — and measure their per-layer AUROC/ECE, pairwise cosine similarity, and causal separability under matched-magnitude activation steering, producing the first direct evidence that gold calibration and verbalized confidence occupy separate, nearly orthogonal linear subspaces.

## Contribution Focus

- Dominant contribution: matched-pair, controls-heavy characterization of the geometric angle and causal separability between the gold-correctness and verbalized-confidence directions in one model on one dataset.
- Optional supporting contribution: dissociation-when-disagree accuracy analysis — when the two disagree, the verbalization is materially less reliable.
- Non-contributions: no new probe method, no new steering algorithm, no fine-tuning, no SAE, no formation tracing, no cross-dataset generalization.

## Proposed Method

### System Overview

Forward pass 1: TriviaQA question → Llama-3.1-8B-Instruct → collect H_1^L at last-input-token, all 32 layers → generate free-form answer → score against TriviaQA aliases → correctness y_c ∈ {0,1}.

Forward pass 2: question + model's answer + "How confident are you that this answer is correct? Give a probability from 0 to 100 as a single number.\nConfidence:" → Llama-3.1-8B-Instruct → collect H_2^L at last-input-token (pre-emission) → generate the number c ∈ [0,100].

Paired-sample dataset: (H_1^L, y_c, H_2^L, c)_i for each sample i at each layer L.

Location: two linear probes (logistic regression) per layer.
- probe_c(H_1^L) → correctness (target y_c)
- probe_v(H_2^L) → verbalized confidence (target: binarized-at-median [c > median])
- Extract L2-normalized direction vectors v_c^L, v_v^L (probe weights)

Geometric orthogonality (C3a): cos(v_c^L, v_v^L) at every layer, reported at matched best-AUROC layers.

Causal separability (C3b): cross-steering — steer forward pass 2 along v_c at best-C1 layer, measure Δ(probe_v readout) at best-C2 layer, and vice versa. Random-direction null of matched norm.

Dissociation-when-disagree (C3c): bin (probe_c high/low) × (verbalized_conf high/low), compare accuracy per cell.

### Complexity Budget

- Frozen: Llama-3.1-8B-Instruct.
- New trainable: two linear probes (logistic regressions, 4096-dim → 1) per layer.
- Rejected: nonlinear probes, SAE decomposition, multi-token-position primary probing (kept as ablation), PCA multi-D subspaces primary (kept as ablation).

### Training Plan

- 10k TriviaQA rc.web validation questions, drop long-answer, split 6k train / 2k dev / 2k test.
- Stage 1: hidden-state + label collection with vllm+HF hooks. ~2h GPU.
- Stage 2: 32 layers × 2 probes = 64 logistic regressions. CPU minutes.
- Stage 3: cos measurement. Negligible.
- Stage 4: cross-steering, 6 α × 3 directions × 2k = 36k forward passes. ~3h GPU.
- Stage 5: ablations. ~1h.
- Verify swaps: ~3h. Total ~9h within 10h HARD budget.

### Failure Modes

1. Probe_v trivially reads emitted confidence token → detect via AUROC ≥ 0.95 red-flag; if so, still novel because C3 can hold geometrically regardless.
2. Model refuses to give a number → detect via <90% parseable-number rate; fallback to Tian 2023 prompt / P(True) / Likert.
3. |cos| ≈ 0 only because both probes are weak → enforce C1/C2 AUROC gates before reporting C3a.
4. TriviaQA gold-correctness ambiguity → alias-list + Wikidata + manual audit on 100 samples.
5. Steering breaks the model → cap |α| at ≤3× perplexity ratio.

### Claim-Driven Validation Sketch

C1: layer-swept logistic regression on H_1^L predicting y_c, 6k/2k/2k. Baselines: token-probability of answer span, P(True), random-direction null, shuffled-label null. Metric: AUROC + ECE (isotonic-calibrated). Expected: AUROC≥0.70, ECE≤0.10, ECE(probe)<ECE(token_prob).

C2: layer-swept logistic regression on H_2^L (pre-emission) predicting binarized [c > median]. Auxiliary linear regression for Spearman ρ. Baselines: post-emission last-token hidden state (trivial upper bound), question-only hidden state (chance lower bound), paraphrased prompt, shuffled-label null. Metric: Spearman ρ, binarized AUROC. Expected: ρ≥0.5, binarized AUROC≥0.70, Δρ across paraphrase≤0.1.

C3a: cos(v_c^L, v_v^L) at every layer. Null: cos with random unit direction. Test: |cos at matched best layers|≤0.3 gated on C1 and C2 passing.
C3b: cross-steering 6 α × 3 directions × 2k. Null: random-direction of matched norm. Test: |Δ_cross_direction| / |Δ_random_direction| ≤ 1.5.
C3c: bin (probe_c, verbalized_conf), compare accuracy per cell. Test: paired-sample stat test on (low probe, high verbal) cell showing lower accuracy than (low probe, low verbal).

### Must-Run Ablations

1. Random-direction probe null
2. Shuffled-label probe null
3. Paraphrase robustness of verbalized-confidence prompt
4. Per-layer trajectory of |cos|
5. Matched-magnitude random-direction steering control
6. Token-position ablation
7. Post-hoc calibration ablation (isotonic vs Platt vs temperature)
8. Best-layer-window robustness (top-3-AUROC layers)

=== END PROPOSAL ===

Score these 7 dimensions from 1-10:

1. **Problem Fidelity**: Does the method still test the three anchored claims (C1, C2, C3a/b/c) EXACTLY as stated, or has it drifted?
2. **Method Specificity**: Are the interfaces, probes, steering protocol, and metrics concrete enough that an engineer could implement them in ~1 day?
3. **Contribution Quality**: Is the matched-pair characterization framed as one dominant contribution, or is it fragmented?
4. **Frontier Leverage**: Is the choice of techniques (logistic-regression probes + linear steering) appropriate for these claims, or is a modern primitive missing / forced?
5. **Feasibility**: Can this be executed in 10h of GPU compute?
6. **Validation Focus**: Are the ablations necessary and sufficient to make C1/C2/C3 defensible against a top-venue reviewer?
7. **Venue Readiness**: If executed well, would the result be publishable at a top venue?

**OVERALL SCORE** (1-10): Weighting: Problem Fidelity 20%, Method Specificity 25%, Contribution Quality 15%, Frontier Leverage 5%, Feasibility 15%, Validation Focus 15%, Venue Readiness 5%.

For each dimension scoring < 7, provide:
- The specific weakness
- A concrete fix at the method level (interface / loss / experimental protocol / deletion of unnecessary parts). Do NOT ask to change the three fixed claims.
- Priority: CRITICAL / IMPORTANT / MINOR

Then add:
- **Simplification Opportunities**: 1-3 concrete ways to delete, merge, or reuse components while preserving the three fixed claims. Write "NONE" if already tight.
- **Modernization Opportunities**: 1-3 concrete ways to replace old-school pieces with more natural foundation-model-era primitives if genuinely better. Write "NONE" if already modern enough.
- **Drift Warning**: "NONE" if the proposal still tests the three claims as stated; otherwise explain the drift clearly.
- **Verdict**: READY / REVISE / RETHINK
  - READY: overall ≥ 9, no drift, one focused dominant contribution, no complexity bloat.
  - REVISE: direction promising but not yet at READY bar.
  - RETHINK: core mechanism or framing is still fundamentally off.
