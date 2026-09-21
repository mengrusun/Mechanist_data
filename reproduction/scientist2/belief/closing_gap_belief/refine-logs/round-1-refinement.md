# Round 1 Refinement

## Problem Anchor (verbatim from round 0)

- **Bottom-line problem**: RLHF-tuned LLMs verbalize confidence scores clustered near 100% regardless of actual correctness. Prior work has separately shown (a) internal states linearly encode correctness/truthfulness and (b) verbalized confidence is a controllable signal. What remains uncharacterized is the geometric relationship between the internal calibration direction and the internal verbalization direction — is verbalized miscalibration a knowledge deficit or a readout failure?
- **Must-solve bottleneck**: single-model, single-dataset matched-pair measurement quantifying C1, C2, and C3 on Llama-3.1-8B-Instruct + TriviaQA.
- **Non-goals**: no new probe algorithm, no SAE, no formation tracing, no fine-tuning, no cross-dataset angle generalization.
- **Constraints**: 10h GPU, GPUs {1,2,3,5,6}, filesystem-restricted, Llama-3.1-8B-Instruct + TriviaQA mandatory, verify swaps bounded.
- **Success condition**: matched (best_layer_C1, best_layer_C2) with C1 AUROC≥0.70 + ECE≤0.10; C2 Spearman ρ≥0.5 + binarized AUROC≥0.70; C3 |cos|≤0.3 + cross-steering null + dissociation-when-disagree.

## Anchor Check

- **Original bottleneck**: measure the geometric angle and causal separability between v_c (gold correctness) and v_v (verbalized confidence) on the same model + dataset with matched controls.
- **Why the revised method still addresses it**: the revised plan (a) pins one canonical residual-stream hook so v_c and v_v are unit vectors in the same 4096-dim residual space (fixing the reviewer's C3 identifiability concern), (b) keeps AUROC / Spearman / ECE / cos / cross-steering metrics on their originally-defined targets, (c) preserves all five must-run ablations that support the C1/C2/C3 chain, (d) preserves the dissociation-when-disagree analysis. No claim is weakened, re-scoped, or re-worded.
- **Reviewer suggestions rejected as drift**: none. Reviewer's recommendations are all method-level tightening, not claim-level changes.

## Simplicity Check

- **Dominant contribution after revision**: matched-pair, canonical-hook geometric+causal characterization of two directions in one model on one dataset — same as round 0, now sharper.
- **Components removed or merged**:
  - Ablation set trimmed from 8 items to 6 must-run + 4 optional (moved out of main paper: token-position ablation, calibration-method comparison, question-only lower bound, post-emission upper bound). Rationale: none of these are load-bearing for C1/C2/C3 as stated.
  - Steering grid trimmed from 6 α × 3 directions × 2k = 36k passes → 3 α × 3 directions × 500 = 4500 passes primary (with a 6-α robustness slice on 200 samples if time permits).
  - C2 primary probe changed: linear regression on continuous c, Spearman ρ primary metric; binarized-median logistic still used for the *direction vector* v_v (keeps unit-length comparability with v_c which is also from a binary probe).
- **Reviewer suggestions rejected as unnecessary complexity**: none — every reviewer suggestion accepted.
- **Why the remaining mechanism is still the smallest adequate route**: only three techniques remain (linear probing, cosine similarity, activation steering); everything else is a control.

## Changes Made

### 1. Canonical residual-stream hook (fixes C3 identifiability)
- **Reviewer said**: "If v_c and v_v are extracted from different prompt states, low cosine may partly reflect representation shift rather than true disentangled subspaces. Define a canonical residual hook location and report cosine only there."
- **Action**: pin the hook to the **residual-stream output after the final residual-add of block L** (standard convention shared with Marks-Tegmark 2023; specifically, the tensor exposed as `outputs.hidden_states[L]` from HuggingFace Llama, which corresponds to `x_L = x_{L-1} + attn_out(x_{L-1}) + mlp_out(x_{L-1} + attn_out(x_{L-1}))`). The same hook definition applies verbatim to forward pass 1 and forward pass 2.
- **Impact on core method**: v_c^L and v_v^L are now unit vectors in the *same* 4096-dim vector space at each layer L, so cos(v_c^L, v_v^L) is a well-defined geometric quantity. This is the primary C3a metric.

### 2. Add cross-steering outcome measurement on the observable (fixes C3b hardness)
- **Reviewer said**: "C3b currently measures change in probe readout, not necessarily change in the actual target variable. Report cross-steering results BOTH on probe readout AND on the emitted confidence / correctness-related observable."
- **Action**: at each steering condition, log BOTH:
  - (a) the pre-emission hidden state H_2^{L_v*} and its projection onto v_v (probe readout Δ)
  - (b) the actually emitted confidence number under steering
  - For v_c-steering during forward pass 1 (correctness pass), log the answer generation and score it against gold aliases (change in actual correctness rate)
- **Impact on core method**: C3b now has TWO tests — internal-readout separability (soft) and outcome separability (hard). The claim is defended if the hard test passes; if only the soft test passes we report this honestly as evidence of internal separability without behavioral change (still meaningful given the anchor's readout-failure vs. knowledge-deficit framing).

### 3. C2 primary probe = continuous linear regressor
- **Reviewer said**: "Collapse C2 primary target to one continuous probe + one thresholded report. Continuous confidence signal is more native than binarized."
- **Action**: probe_v_continuous: linear regression on H_2^L → predicted c ∈ ℝ, evaluated via Spearman ρ against actual c. Also fit probe_v_binary: logistic regression on H_2^L → predicted [c > median], evaluated via AUROC. **The direction vector v_v used for C3 comes from probe_v_binary** (unit-length weight vector). Rationale: v_c also comes from a binary probe (logistic on y_c ∈ {0,1}), so both direction vectors are unit-length weight vectors of binary logistic regressions — apples-to-apples for cosine comparison.
- **Impact on core method**: cleaner primary metric for C2 (Spearman ρ ≥ 0.5), same C3a comparability.

### 4. Steering grid trimmed
- **Reviewer said**: "Use 3 alpha values instead of 6 for the primary C3b claim, with matched random-direction null."
- **Action**: primary grid = {−1σ, 0, +1σ} × {v_c, v_v, random_unit} × 500 held-out samples = 4500 forward passes. Optional 6-α robustness on a 200-sample subset if compute permits.
- **Impact**: main-experiment steering cost drops from ~3h → ~45 min.

### 5. Must-run ablations pruned to 6
- **Kept as must-run**:
  1. Random-direction probe null (C1, C2)
  2. Shuffled-label probe null (C1, C2)
  3. Paraphrase robustness of verbalized-confidence prompt on 500-sample dev slice (C2, C3a stability)
  4. Per-layer trajectory of |cos(v_c^L, v_v^L)| (C3a evidence)
  5. Matched-magnitude random-direction steering control (C3b)
  6. Bootstrap CIs on per-layer probe scores + cos values
- **Moved to optional appendix**: token-position ablation, calibration-method comparison, question-only lower bound, post-emission upper bound.
- **Impact**: total main experiment ablation cost drops from ~1h → ~30 min.

### 6. Revised compute budget
- Stage 1 hidden-state + label collection (10k questions × 2 passes with vllm-batched generation + HF hooks for hidden states): ~2h
- Stage 2 probe training (32 layers × 3 probes: probe_c binary, probe_v continuous, probe_v binary): CPU minutes
- Stage 3 cos measurement + bootstrap CIs: negligible
- Stage 4 steering (4500 passes primary + 1000 optional): ~1.5h
- Stage 5 must-run ablations (paraphrase 500 × 2 passes for a second prompt variant): ~30 min
- Verify swaps (2 model × 1 dataset OR 1 model × 2 datasets subset — see EXPERIMENT_PLAN): ~2h
- **Total main + verify**: ~6h, leaving 4h buffer for iteration / re-runs / integrity fixes

## Revised Proposal

# Research Proposal (round 1): Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence in Llama-3.1-8B-Instruct

## Problem Anchor
(verbatim as above)

## Technical Gap
Two mature literatures (correctness probing since Marks-Tegmark 2023; verbalized confidence since Kadavath 2022 / Tian 2023) converge in the 2025 wave (Kossen 2025, Zhang 2025, HACK 2025) on a nascent dissociation view. None of these papers directly measure the **geometric angle** between the gold-correctness probe direction and the verbalized-confidence probe direction on matched hidden states from the same model + dataset, with per-layer trajectory and causal-orthogonality steering. The specific gap: whether verbalized miscalibration is a knowledge deficit (both signals collapsed onto one distorted channel, |cos|≈1) or a readout failure (calibration is intact but written on a channel orthogonal to the verbalization channel, |cos|≈0 with causal separability).

## Method Thesis
On Llama-3.1-8B-Instruct evaluated on TriviaQA, we train two independent linear probes on paired residual-stream hidden states extracted at **one canonical hook location** — one predicting gold correctness of the model's free-form answer, one predicting the numeric confidence the model will subsequently verbalize — and measure their per-layer AUROC / Spearman ρ / ECE, their pairwise cosine similarity at matched best-AUROC layers, and their causal separability under matched-magnitude activation steering evaluated on both internal probe readouts and the actually emitted confidence number, thereby producing the first direct evidence that gold calibration and verbalized confidence occupy separate, nearly orthogonal linear subspaces.

## Contribution Focus
- **Dominant contribution**: matched-pair, canonical-hook, controls-heavy characterization of the geometric angle and causal separability between v_c and v_v in one model on one dataset — the direct measurement no prior paper reports.
- **Optional supporting contribution**: dissociation-when-disagree accuracy analysis.
- **Non-contributions**: no new probe method, no new steering algorithm, no fine-tuning, no SAE, no formation tracing, no cross-dataset generalization of the angle.

## Proposed Method

### Complexity Budget
- **Frozen**: Llama-3.1-8B-Instruct.
- **New trainable**: three logistic-regression / linear-regression probes per layer (probe_c binary; probe_v continuous; probe_v binary — the latter's weights become v_v for C3 comparability).
- **Rejected**: nonlinear probes, SAE, multi-token-position primary probing, PCA multi-D subspaces primary.

### Canonical Hook (NEW)
- **Hook location**: `outputs.hidden_states[L]` from HuggingFace Llama = residual-stream state after the final residual-add of block L. This is the same tensor a "logit-lens" projection would multiply by W_U at the final layer, and is Marks-Tegmark 2023's convention.
- **Same hook definition applies to forward pass 1 and forward pass 2.**
- Token position: last input token *before the next auto-regressive token is emitted*. For forward pass 1, this is the last question token. For forward pass 2, this is the last token of the confidence-elicitation prompt (i.e., the last character of `"...Confidence:"` — the position at which the model is about to emit the number).

### System Overview
```
                     ┌─── forward pass 1 (answer generation) ────────┐
                     │                                                │
TriviaQA question ──► Llama-3.1-8B-Instruct                           │
                     │  (a) canonical hook H_1^L at last-question-tok │
                     │      for all 32 layers                         │
                     │  (b) generate free-form answer                 │──► correctness y_c
                     │  (c) score against TriviaQA aliases            │
                     └────────────────────────────────────────────────┘

                     ┌─── forward pass 2 (verbalized confidence) ─────┐
                     │                                                │
question + answer ──► Llama-3.1-8B-Instruct                           │
   + "How confident │  (a) canonical hook H_2^L at last-prompt-tok    │
   are you? Give a  │      for all 32 layers                         │
   number 0-100"    │  (b) generate the number                       │──► verbalized c
                     │  (c) parse the emitted number                 │
                     └────────────────────────────────────────────────┘

Paired-sample dataset: (H_1^L, y_c, H_2^L, c)_i at each layer L.

┌─── Location: three linear probes per layer (32 layers × 3 = 96 probes) ────┐
│  probe_c_binary(H_1^L)     → correctness            → v_c^L (unit vector)   │
│  probe_v_continuous(H_2^L) → predicted c ∈ ℝ         → Spearman ρ (primary)  │
│  probe_v_binary(H_2^L)     → [c > median(c_train)]  → v_v^L (unit vector),  │
│                                                       binarized AUROC       │
└────────────────────────────────────────────────────────────────────────────┘

┌─── C3a Geometric orthogonality ────────────────────────────────────────────┐
│  cos_sim(v_c^L, v_v^L) at every layer L, with bootstrap CIs                │
│  Primary claim: |cos(v_c^{L_c*}, v_v^{L_v*})| ≤ 0.3 gated on C1, C2 passing│
│  Secondary: |cos(v_c^L, v_v^L)| at the layer where BOTH probes pass their  │
│    AUROC gates jointly (nearest-shared-layer control)                       │
│  Null: cos(v_c, random_unit_direction) — expected ≈ 0 with std √(1/4096)   │
└────────────────────────────────────────────────────────────────────────────┘

┌─── C3b Causal separability — cross-steering ───────────────────────────────┐
│  Grid: α ∈ {−1σ, 0, +1σ} × direction ∈ {v_c, v_v, random_unit} × 500 test  │
│    samples = 4500 forward passes                                            │
│  For each condition, log:                                                   │
│    (i) internal probe readout Δ                                             │
│    (ii) actually emitted output (confidence number for v_v-steering on pass│
│         2; scored correctness for v_c-steering on pass 1)                   │
│  C3b passes iff |Δ_v_other_steer| / |Δ_random_steer| ≤ 1.5 on BOTH internal│
│    readout AND emitted-output measurements                                  │
└────────────────────────────────────────────────────────────────────────────┘

┌─── C3c Dissociation-when-disagree ─────────────────────────────────────────┐
│  Bin held-out samples on (probe_c_output_calibrated, verbalized_conf)      │
│  Report accuracy per cell; paired-sample test on (low probe, high verbal)   │
│    vs. (low probe, low verbal)                                              │
└────────────────────────────────────────────────────────────────────────────┘
```

### Core Mechanism
- **Input / output**: as diagrammed. Canonical hook at each of 32 layers, last-input-token position on both forward passes.
- **Architecture**: L2-regularized logistic regression (probe_c_binary, probe_v_binary) with C hyperparameter tuned on the dev split; L2-regularized linear regression (probe_v_continuous). Direction vectors v_c^L, v_v^L extracted as L2-normalized weight vectors of the two *binary* probes (comparability guarantee).
- **Training signal**: cross-entropy (binary probes) + MSE (continuous probe). y_c = 1 iff the model's free-form answer normalized-matches one of the TriviaQA gold aliases; c = the parsed emitted number in [0, 100].

### Confidence Elicitation Prompt
- **Primary prompt (P0)**:
  ```
  Q: {question}
  A: {model's answer}
  How confident are you that this answer is correct? Give a probability from 0 to 100 as a single number.
  Confidence:
  ```
- **Paraphrase variants (for robustness ablation)**:
  - P1 (Tian 2023 style): `"What is the probability from 0.0 to 1.0 that your answer above is correct? Probability:"`
  - P2 (Likert): `"How confident are you? Answer with a single word: very-low / low / medium / high / very-high. Confidence:"` → mapped to 20/40/60/80/100
- Robustness check: run P1 and P2 on a 500-sample dev slice and report ΔSpearman and ΔAUROC vs. P0. C2 must be robust: Δ ≤ 0.1.

### TriviaQA Correctness Scoring
- Normalized exact-match: lowercase, strip punctuation and articles ("a", "an", "the"), collapse whitespace.
- Match model's generated answer against the union of TriviaQA-provided aliases + a per-question canonical answer.
- 100-sample manual audit to verify the automated score matches human judgment.

### Steering Implementation
- **Hook site**: same canonical residual-stream site (`outputs.hidden_states[L]`), rewritten via HF forward hook by adding `α × v` to the residual state at the target token position before the next block.
- **Steering token position**: apply steering at the last-input-token position only (matches the position at which v was learned).
- **Steering magnitude**: α measured in units of σ_L = the standard deviation of the residual-stream activation magnitude at layer L on the training set. Grid: {−1σ, 0, +1σ}. Random direction has matched L2 norm to v (both are unit vectors, so the intervention magnitudes are identically ±α × σ_L in Euclidean norm).
- **Perplexity safety cap**: if steered generations show > 3× mean token perplexity vs. baseline, halve α and re-run.

### Training Plan
- **Data**: 10k TriviaQA `rc.web` validation questions; drop questions whose gold answer set has no single-word or short-phrase representative (keeps closed-book QA regime). Stratified split: 6k train probe / 2k dev / 2k test.
- **Stage 1 — Hidden-state + label collection**: two forward passes per sample; cache H_1^L and H_2^L for all 32 layers at last-input-token position, y_c, c, and the emitted answer text. Cost: ~2h GPU with vllm-batched generation + HF hooks.
- **Stage 2 — Probe training**: 32 layers × 3 probes = 96 fits on 6k train / 2k dev / 2k test. Cost: minutes.
- **Stage 3 — Geometric measurement**: cos(v_c^L, v_v^L) at every layer with 1000-sample bootstrap CIs on the training set (re-fit probes on bootstrap resamples). Cost: minutes.
- **Stage 4 — Causal steering**: primary 4500 passes (~45 min with vllm-batched); optional 6-α extension on 200 samples if time permits (~15 min).
- **Stage 5 — Ablations**: paraphrase robustness (500 samples × 2 alt prompts × forward pass 2 only + probe re-fit) → ~30 min.
- **Stage 6 — Verify swaps**: two swap variants each on 1k held-out (see EXPERIMENT_PLAN.md for exact selection). ~2h.
- **Total main + verify**: ~6h GPU, leaving 4h buffer.

### Failure Modes and Diagnostics
1. **Probe_v trivially reads emitted confidence**: detect via AUROC ≥ 0.95 flag; report as finding (C3 can still hold).
2. **Model refuses to give a number**: detect via parseable-rate < 90%; fallback to P1 / P2 or logit-based `P(top-integer)`.
3. **|cos| ≈ 0 only because probes are weak**: enforce AUROC gates on C1 & C2 before interpreting C3a.
4. **TriviaQA scoring ambiguity**: alias union + 100-sample manual audit.
5. **Steering breaks the model**: cap α at ≤ 3× perplexity ratio.

### Novelty and Elegance Argument
Closest work: Kossen 2025. Differences: (a) they measure scalar Pearson r between two output-space uncertainty scores, we measure vector cosine between two representation-space probe directions extracted from a canonical hook; (b) they report a best-layer, we report full per-layer trajectories with bootstrap CIs; (c) they don't do cross-direction steering with a random-direction null, we do; (d) they use semantic uncertainty (sampling-based) as the "internal" side, we use gold correctness (unambiguous ground truth). The paper is one main table (per-layer AUROC/Spearman/ECE/cos for C1, C2, C3a), one causal table (cross-steering + dissociation for C3b, C3c), five must-run ablations, all standard.

## Claim-Driven Validation Sketch

### C1: Gold correctness is linearly accessible in Llama-3.1-8B-Instruct hidden states on TriviaQA
- **Experiment**: layer-swept probe_c_binary on H_1^L predicting y_c.
- **Baselines**: token-probability of the answer span, P(True) self-elicitation, random-direction null, shuffled-label null.
- **Metric**: AUROC + ECE (isotonic post-calibration on dev).
- **Expected**: AUROC ≥ 0.70 at best layer, ECE ≤ 0.10, ECE(probe) < ECE(token_prob).

### C2: Verbalized confidence is linearly accessible in Llama-3.1-8B-Instruct hidden states on TriviaQA (pre-emission)
- **Experiment**: layer-swept probe_v_continuous on H_2^L predicting c ∈ ℝ (primary). Auxiliary probe_v_binary for AUROC secondary.
- **Baselines**: random-direction null, shuffled-label null, paraphrase (P1, P2) on 500-sample dev slice.
- **Metric**: Spearman ρ (primary), binarized AUROC (secondary).
- **Expected**: Spearman ρ ≥ 0.5, binarized AUROC ≥ 0.70 at best layer, robust to paraphrase (Δρ ≤ 0.1).

### C3a: Geometric orthogonality
- **Experiment**: cos(v_c^L, v_v^L) at every layer; primary report at matched best-AUROC layers; nearest-shared-layer secondary.
- **Baselines**: cos(v_c, random_unit) null; shuffled-label direction null.
- **Metric**: |cos| with bootstrap 95% CI.
- **Expected**: |cos(v_c^{L_c*}, v_v^{L_v*})| ≤ 0.3 with C1 & C2 gates passed.

### C3b: Causal separability
- **Experiment**: 3 α × 3 directions × 500 samples = 4500 forward passes; log internal probe readout AND emitted output per condition.
- **Baseline**: random-unit-direction steering, matched σ.
- **Metric**: |Δ_v_other_steer| / |Δ_random_steer| on BOTH readout AND output.
- **Expected**: ratio ≤ 1.5 in both directions on both measurements.

### C3c: Dissociation-when-disagree
- **Experiment**: bin held-out on (probe_c_calibrated_output, verbalized_conf), report accuracy per cell.
- **Metric**: accuracy(low probe ∧ high verbal) < accuracy(low probe ∧ low verbal), paired sample test.
- **Expected**: statistically significant gap.

## Experiment Handoff Inputs
- **Must-prove claims**: C1, C2, C3a, C3b, C3c.
- **Must-run ablations** (pruned to 6):
  1. Random-direction probe null (C1, C2)
  2. Shuffled-label probe null (C1, C2)
  3. Paraphrase robustness of verbalized-confidence prompt (500-sample dev; C2, C3 stability)
  4. Per-layer trajectory of |cos| (C3a evidence + bootstrap CIs)
  5. Matched-magnitude random-direction steering control (C3b)
  6. Bootstrap CIs on all headline metrics
- **Optional (appendix if time)**: token-position ablation; calibration-method comparison (isotonic vs Platt vs temperature); question-only lower bound; post-emission upper bound.
- **Critical datasets / metrics**: TriviaQA `rc.web` validation; AUROC, Spearman ρ, ECE, cos, cross-steering readout+output Δ, dissociation-cell accuracy.
- **Highest-risk assumptions**: TriviaQA aliasing quality, pre-emission hidden state not trivially reading emitted number (handled), 6k probe train sufficient (mitigated by bootstrap CIs).

## Compute & Timeline Estimate
- Total GPU-hours: ~6h main + ~2h verify swaps = ~8h, buffer 2h.
- Data cost: none new; TriviaQA + aliases free. ~1 person-hour for 100-sample manual audit.
- Timeline: 1 day implementation + 1 day main + 1 day verify + buffer.
