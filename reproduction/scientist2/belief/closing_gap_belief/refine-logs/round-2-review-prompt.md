[Round 2 re-evaluation — llm-chat is stateless; prior context is included below]

## Prior Round Review (verbatim summary from round 1)

- Previous overall score: **7.1/10**
- Previous verdict: **REVISE**
- Previous drift warning: **NONE**
- Previous dimension scores: Problem Fidelity 8, Method Specificity 8, Contribution Quality 7, Frontier Leverage 8, Feasibility 6, Validation Focus 6, Venue Readiness 6.
- Previous top action items:
  - **CRITICAL — Feasibility**: trim steering grid to 3α (not 6), move token-position + calibration-method + question-only + post-emission ablations to optional, run paraphrase on 500-sample dev slice, cache prompts/states to avoid duplicate generation.
  - **CRITICAL — Validation Focus / C3 identification**: define ONE canonical residual-stream hook location and report cosine primarily at that same hook location on both forward passes; measure cross-steering effect on BOTH probe readout AND emitted output.
  - **IMPORTANT — Venue Readiness**: frame as matched-pair, same-representation-space characterization.
  - **Simplifications**: (1) Collapse C2 primary to continuous linear regressor with Spearman ρ primary + binarized AUROC derived-secondary. (2) Bootstrap CIs on per-layer probe scores + cos.

I revised the proposal based on that feedback. Same three fixed claims (C1, C2, C3). No claim was modified. All reviewer method-level suggestions were accepted.

Key changes:
1. **Canonical residual-stream hook** pinned to `outputs.hidden_states[L]` (post-block residual add) at the last-input-token position on both forward passes. This makes v_c^L and v_v^L unit vectors in the same 4096-dim residual space at each layer, so cos is a well-defined geometric quantity.
2. **Cross-steering measured on BOTH probe readout AND emitted output**: for v_v-steering during forward pass 2, we log the actually emitted confidence number and compare to baseline. For v_c-steering during forward pass 1, we log the generated answer and score its correctness change.
3. **C2 primary probe = continuous linear regressor** (predicts c ∈ ℝ from H_2^L, primary metric = Spearman ρ ≥ 0.5). Direction vector v_v extracted from a *binarized* probe_v_binary (unit-length weight vector for apples-to-apples cosine comparability with v_c which is also from a binary probe).
4. **Steering grid trimmed** to {−1σ, 0, +1σ} × {v_c, v_v, random_unit} × 500 samples = 4500 forward passes (was 36k). Optional 6-α robustness on 200 samples if time.
5. **Must-run ablations pruned to 6** (random-direction null, shuffled-label null, paraphrase robustness on 500-sample dev, per-layer |cos| trajectory, matched-magnitude random-direction steering control, bootstrap CIs). Moved to optional appendix: token-position ablation, calibration-method comparison, question-only lower bound, post-emission upper bound.
6. **Compute budget refit**: ~6h main + ~2h verify = ~8h, 2h buffer under 10h HARD cap.

Please re-score the same 7 dimensions with the same weighting. Same verdict rule: READY only if overall ≥ 9 and no blocking issue remains. Use the same review principles — do not ask to change the three fixed claims; do not add SAEs / formation tracing / fine-tuning; do not ask for cross-dataset angle generalization.

=== REVISED PROPOSAL ===

# Research Proposal (round 1 revised): Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence in Llama-3.1-8B-Instruct

## Problem Anchor (verbatim from round 0)

- Bottom-line problem: RLHF-tuned LLMs verbalize confidence scores clustered near 100% regardless of actual correctness. Prior work has separately shown (a) internal states linearly encode correctness/truthfulness and (b) verbalized confidence is a controllable signal. What remains uncharacterized is the geometric relationship between the internal calibration direction and the internal verbalization direction — is verbalized miscalibration a knowledge deficit or a readout failure?
- Must-solve bottleneck: single-model, single-dataset matched-pair measurement quantifying C1, C2, and C3 on Llama-3.1-8B-Instruct + TriviaQA.
- Non-goals: no new probe algorithm, no SAE, no formation tracing, no fine-tuning, no cross-dataset angle generalization.
- Constraints: 10h GPU, GPUs {1,2,3,5,6}, filesystem-restricted, Llama-3.1-8B-Instruct + TriviaQA mandatory, verify swaps bounded.
- Success condition: matched (best_layer_C1, best_layer_C2) with C1 AUROC≥0.70 + ECE≤0.10; C2 Spearman ρ≥0.5 + binarized AUROC≥0.70; C3 |cos|≤0.3 + cross-steering null (on both readout and output) + dissociation-when-disagree.

## Method Thesis

On Llama-3.1-8B-Instruct evaluated on TriviaQA, train two independent linear probes on paired residual-stream hidden states extracted at ONE canonical hook location (`outputs.hidden_states[L]`, post-block residual add) — one predicting gold correctness of the model's free-form answer, one predicting the numeric confidence the model will subsequently verbalize — and measure their per-layer AUROC / Spearman ρ / ECE with bootstrap CIs, their pairwise cosine similarity at matched best-AUROC layers, and their causal separability under matched-magnitude activation steering evaluated on BOTH internal probe readouts AND the actually emitted output number, thereby producing the first direct evidence that gold calibration and verbalized confidence occupy separate, nearly orthogonal linear subspaces.

## Contribution Focus
- Dominant contribution: matched-pair, canonical-hook, controls-heavy geometric+causal characterization of v_c and v_v in one model on one dataset.
- Supporting: dissociation-when-disagree accuracy analysis.
- Non-contributions: same as before.

## Proposed Method

### Canonical Hook (NEW)
Hook location: `outputs.hidden_states[L]` from HuggingFace Llama = residual-stream state after the final residual-add of block L. Same hook definition applies to forward pass 1 and forward pass 2. Token position: last input token before next auto-regressive token is emitted (last question token for pass 1; last token of "...Confidence:" for pass 2).

### Complexity Budget
- Frozen: Llama-3.1-8B-Instruct.
- New trainable: 32 layers × 3 probes = 96 fits: probe_c_binary (correctness y_c ∈ {0,1}), probe_v_continuous (verbalized c ∈ ℝ, Spearman ρ primary metric for C2), probe_v_binary (binarized [c > median], AUROC secondary metric for C2 AND source of v_v unit vector for C3).
- Rejected: nonlinear probes, SAE, multi-token-position primary probing, PCA multi-D subspaces primary.

### Confidence Elicitation Prompt
Primary P0: `"Q: {question}\nA: {model's answer}\nHow confident are you that this answer is correct? Give a probability from 0 to 100 as a single number.\nConfidence:"`. Paraphrase variants P1 (Tian 2023 0.0-1.0 style) and P2 (Likert mapped to 20/40/60/80/100). Robustness check on 500-sample dev slice: Δρ ≤ 0.1 across variants.

### TriviaQA Correctness Scoring
Normalized exact-match against union of TriviaQA aliases + per-question canonical. 100-sample manual audit for automated-vs-human agreement.

### Steering Implementation
Same canonical residual hook. Add α × v at last-input-token position; α measured in units of σ_L = std of residual-stream activation magnitude on training set. Grid: {−1σ, 0, +1σ}. Random direction has matched L2 norm. Perplexity safety cap: if steered gens show > 3× baseline perplexity, halve α.

### Training Plan
- Data: 10k TriviaQA `rc.web` validation, drop long-answer, 6k/2k/2k split.
- Stage 1: hidden-state + label collection (~2h GPU).
- Stage 2: probe training (CPU minutes).
- Stage 3: cos + bootstrap CIs (minutes).
- Stage 4: cross-steering, 4500 primary passes (~45 min) + optional 6-α on 200 samples (~15 min).
- Stage 5: paraphrase robustness on 500 dev × 2 alt prompts (~30 min).
- Stage 6: verify swaps ~2h.
- Total: ~6h main + ~2h verify, under 10h HARD.

## Claim-Driven Validation Sketch

C1: layer-swept probe_c_binary on H_1^L predicting y_c. Baselines: token-prob, P(True), random-dir null, shuffled-label null. Metric: AUROC + ECE (isotonic). Expected: AUROC ≥ 0.70, ECE ≤ 0.10, ECE(probe) < ECE(token_prob).

C2: layer-swept probe_v_continuous on H_2^L predicting c (primary). Auxiliary probe_v_binary for AUROC secondary. Baselines: random-dir null, shuffled-label null, paraphrase (P1, P2) on 500-dev. Metric: Spearman ρ primary, binarized AUROC secondary. Expected: ρ ≥ 0.5, AUROC ≥ 0.70, Δρ across paraphrase ≤ 0.1.

C3a: cos(v_c^L, v_v^L) per layer with bootstrap CIs. Primary at matched best-AUROC layers; secondary at nearest-shared-layer. Null: cos with random unit direction. Test: |cos| ≤ 0.3 gated on C1, C2.

C3b: 3α × 3 directions × 500 = 4500 passes. Log BOTH internal probe readout AND emitted output. Test: |Δ_v_other|/|Δ_random| ≤ 1.5 on BOTH measurements, in BOTH cross directions.

C3c: bin (probe_c_calibrated, verbalized_conf) × 4 cells, report accuracy per cell. Test: accuracy(low probe, high verbal) < accuracy(low probe, low verbal), paired-sample stat test.

## Must-Run Ablations (6)
1. Random-direction probe null
2. Shuffled-label probe null
3. Paraphrase robustness (500-sample dev)
4. Per-layer |cos| trajectory + bootstrap CIs
5. Matched-magnitude random-direction steering control
6. Bootstrap CIs on headline metrics

## Optional (appendix)
Token-position ablation, calibration-method comparison, question-only lower bound, post-emission upper bound, 6-α robustness slice.

=== END REVISED PROPOSAL ===

Please:
- Re-score the same 7 dimensions and overall
- State whether the Problem Anchor is preserved or drifted
- State whether the dominant contribution is now sharper or still too broad
- State whether the method is simpler or still overbuilt
- State whether the frontier leverage is now appropriate
- Focus new critiques on missing mechanism, weak training signal, weak integration point, pseudo-novelty, or unnecessary complexity
- Use the same verdict rule: READY only if overall score ≥ 9 and no blocking issue remains
- Same output format: 7 scores, overall score, verdict, drift warning, simplification opportunities, modernization opportunities, remaining action items.
