[Round 3 re-evaluation — llm-chat is stateless; prior context is included below]

## Prior Round Review (verbatim summary from round 2)

- Previous overall score: **8.5/10** (up from 7.1 in round 1)
- Previous verdict: **REVISE** (need ≥ 9 for READY)
- Previous drift warning: **NONE**
- Previous dimension scores: PF 8.8, MS 9.0, CQ 8.1, FL 8.4, F 8.8, VF 8.4, VR 8.0.
- Previous BLOCKING items:
  1. Tighten interpretive framing — support only "linearly separable and weakly aligned" + "limited causal coupling under tested prompts", not full "readout failure".
  2. Weak-training-signal contingency for C2 (verbalized confidence may cluster near 100).
  3. Two-context caveat + robustness variant (C3 cos is same-model same-layer across two elicitation contexts, not one shared task state).
- Previous IMPORTANT items:
  4. Absolute-effect companion criterion for C3b when ratio denominator is tiny.
  5. Justify last-input-token position.
  6. Explicit parsing/normalization for confidence numbers across P0/P1/P2.
  7. Frame 100-sample manual audit as label-quality verification.
- Previous simplifications requested: collapse "matched best-AUROC" to one primary layer; de-emphasize binary/continuous C2-probe seam; keep C3c as secondary.
- Previous modernizations requested: report full effect-size distributions with bootstrap CIs, lean into paired-sample framing, preempt representation-space objections.

I revised the proposal based on that feedback. Same three fixed claims (C1, C2, C3). No claim modified. All reviewer suggestions accepted at the proposal level.

Key changes in round 3:
1. Added a **"Scope of the Mechanism Claim" paragraph** — the anchor's "knowledge deficit vs. readout failure" is now stated as the *motivating dichotomy*, and the *asserted evidence-supported conclusion* is bounded to "linear separability with low geometric alignment and limited causal cross-coupling in this model + dataset + prompts + hook".
2. Added **Stage 1.5 Verbalized-Confidence Variance Diagnostic** with an explicit decision rule: if std(c) ≥ 15 AND share_at_max ≤ 0.7 → continuous-probe primary; else → ordinal 4-bin probe primary with binary probe at 30th percentile for v_v.
3. Added **explicit two-context Scope Statement** + a **single-pass robustness variant** (unified prompt with question + confidence in ONE forward pass; report cos alongside two-pass primary).
4. **Primary reporting layer L\*** = argmax_L of the mean of normalized AUROC_c(L) and normalized AUROC_v_binary(L). ONE headline cos number at L*, per-layer trajectories as robustness only.
5. **Absolute-effect companion criterion** for C3b: passes iff EITHER ratio criterion holds when random Δ ≥ 0.5σ, OR absolute-effect |Δ_v_other| ≤ 0.5σ when ratio unstable.
6. **Full effect-size distributions with bootstrap CIs** as primary evidence; threshold-pass table is derived summary.
7. Confidence-number parsing spec + last-input-token rationale + audit framing all added.

Please re-score. Same verdict rule: READY only if overall ≥ 9 and no blocking issue remains.

=== REVISED PROPOSAL (round 3) ===

# Research Proposal (round 3): Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence in Llama-3.1-8B-Instruct

## Problem Anchor (verbatim from round 0)

- Bottom-line problem, must-solve bottleneck, non-goals, constraints, success condition — UNCHANGED across rounds.

## Scope of the Mechanism Claim (NEW round 3 §)

The task-anchor's motivating dichotomy is: is RLHF-induced overconfidence a *knowledge deficit* (calibration is broken internally) or a *readout failure* (calibration is intact but the verbalization channel does not surface it)? This project does NOT claim to resolve that dichotomy in full generality. What the proposed evidence can support is (i) that a linear direction in Llama-3.1-8B-Instruct's residual stream at a canonical hook location predicts gold correctness on TriviaQA (C1), (ii) that another such direction linearly predicts pre-emission verbalized confidence on the same model + dataset (C2), and (iii) that these two directions are geometrically weakly aligned in that specific model + dataset + prompt distribution + hook, and are causally weakly coupled under matched-magnitude activation steering restricted to those prompts (C3). Any further mechanism-level interpretation ("this proves readout failure") is expressly out of scope and left to follow-up work.

## Method Thesis

On Llama-3.1-8B-Instruct evaluated on TriviaQA, train two independent linear probes on paired residual-stream hidden states extracted at ONE canonical hook location (`outputs.hidden_states[L]`, post-block residual add) — one predicting gold correctness of the model's free-form answer, one predicting the numeric confidence the model will subsequently verbalize — and measure their per-layer AUROC / Spearman ρ / ECE with bootstrap CIs, their pairwise cosine similarity at a canonical single primary reporting layer L\*, and their causal separability under matched-magnitude activation steering evaluated on BOTH internal probe readouts AND emitted output, producing the first direct measurement of the geometric alignment and causal cross-coupling between gold calibration and verbalized confidence.

## Contribution Focus
- Dominant: matched-pair, canonical-hook, same-representation-space characterization of the angle and causal cross-coupling between v_c and v_v.
- Supporting (clearly secondary): dissociation-when-disagree accuracy analysis.
- Non-contributions: no new probe algorithm, no SAE, no formation tracing, no fine-tuning, no cross-dataset angle generalization.

## Proposed Method

### Canonical Hook (fixed)
`outputs.hidden_states[L]` (residual-stream post-block-out at layer L). Same hook applies to forward passes 1, 2, and the single-pass unified-prompt robustness variant.

### Token Position Rationale (NEW round 3 §)
Last-input-token = final pre-emission state, i.e., the state used to produce the next token. Marks-Tegmark 2023 and the majority of the truthfulness-probing literature use this convention, making cosine numbers directly comparable across the literature. Applied identically to pass 1 (last question token before answer generation) and pass 2 (last token of "...Confidence:" before number emission).

### Complexity Budget
- Frozen: Llama-3.1-8B-Instruct.
- New trainable: probes only. Primary probe_c_binary (correctness). Primary probe_v_continuous OR probe_v_ordinal (chosen by Stage-1.5 variance diagnostic). Auxiliary probe_v_binary (produces unit vector v_v for C3 comparability with v_c).
- Rejected: nonlinear probes, SAE, multi-token-position primary, PCA multi-D primary.

### Stage 1.5 Verbalized-Confidence Variance Diagnostic (NEW round 3)
Before probe training, compute over the 10k-sample training set:
- mean(c), std(c), entropy(c), quantiles(c), share(c = max_bin), share(c = min_bin), effective sample spread.
- Decision rule (pre-registered):
  - If `std(c) ≥ 15` AND `share(c at max_bin) ≤ 0.7` → probe_v_continuous is primary, probe_v_binary(median) is secondary + source of v_v.
  - Else → probe_v_ordinal (4 bins {0-25, 26-50, 51-75, 76-100}, ordinal cross-entropy) is primary, probe_v_binary(30th_percentile) is source of v_v (guarantees ≥30% minority class).
- The chosen path + the diagnostic values are reported in the paper alongside C2 evidence.

### Primary Reporting Layer L\* (NEW round 3)
`L* = argmax_L [ AUROC_c(L) / max_L' AUROC_c(L') + AUROC_v_binary(L) / max_L' AUROC_v_binary(L') ]`.
All headline cosine + steering results reported at L*. Per-layer trajectories are secondary robustness evidence.

### Two-Elicitation-Context Scope Statement (NEW round 3)
v_c is extracted from forward pass 1 (question only, before answer generation). v_v is extracted from forward pass 2 (question + answer + confidence-elicitation, before number emission). C3's cos is a well-defined geometric quantity between two unit vectors in the same 4096-dim residual space at layer L, but the two vectors are learned from activations produced under two different textual contexts. The claim C3a is therefore about "same-model, same-layer residual geometry across two elicitation contexts", not about a single shared task state.

### Single-Pass Robustness Variant (NEW round 3)
An additional forward pass with a unified prompt that concatenates the question, a temporary answer slot, and the confidence elicitation into ONE input, all before the model emits anything: `"Q: {question}\nAnswer with your best guess then confidence (0-100).\nA:"` — hidden state at last-input-token used to train probe_c_single (against y_c derived from the model's subsequently-emitted answer) and probe_v_single (against c from the subsequently-emitted confidence). Report cos(v_c_single^L*, v_v_single^L*) alongside the two-pass primary. This directly answers the "different textual contexts" objection.

### Steering (C3b)
- Grid: α ∈ {−1σ, 0, +1σ} × direction ∈ {v_c^L*, v_v^L*, random_unit^L*} × 500 held-out samples = 4500 forward passes.
- Metrics per condition: (i) internal probe readout Δ at L*, (ii) emitted output Δ (confidence number for v_v steering; correctness rate change for v_c steering).
- Pass criterion: EITHER (i) `|Δ_v_other| / |Δ_random_direction| ≤ 1.5` when `|Δ_random| ≥ 0.5σ_probe_readout` OR (ii) `|Δ_v_other| ≤ 0.5σ_probe_readout` when the ratio is undefined/unstable. Reported on BOTH internal readout AND emitted output.
- Perplexity safety: if steered generations show > 3× baseline token-level perplexity, halve α.

### Confidence Number Parsing (NEW spec)
Extract the first integer in [0, 100] (or first decimal in [0.0, 1.0] × 100 for P1, or mapped Likert bin for P2) from the model's output up to 8 tokens after "Confidence:" / "Probability:". Unparseable samples marked and excluded from C2; unparseable-rate reported as an integrity metric (target < 10%; if ≥ 10% → switch to fallback prompt).

### Reporting Convention (NEW modernization)
All headline metrics reported as (mean, 95% bootstrap CI over 1000 resamples of the probe training split). Cross-steering Δ reported as (mean, 95% bootstrap CI over the 500 held-out samples) + full distribution (violin plot). Threshold-pass table is a derived summary of the primary quantitative results.

### Training Plan
- Data: 10k TriviaQA rc.web, drop long-answer, 6k/2k/2k split.
- Stage 1: forward passes 1, 2, and single-pass robustness. Cache hidden states at last-input-token position, all 32 layers, ~4 GB total. ~2h GPU.
- Stage 1.5: variance diagnostic on c. Minutes.
- Stage 2: 32 layers × 3 primary probes + 3 single-pass probes = ~200 probe fits. CPU minutes.
- Stage 3: cos + bootstrap CIs. Minutes.
- Stage 4: cross-steering, 4500 primary passes (~45 min).
- Stage 5: paraphrase robustness on 500-sample dev × 2 alt prompts (~30 min).
- Stage 6: verify swaps ~2h.
- Total: ~6h main + ~2h verify = ~8h, buffer 2h under 10h HARD.

## Claim-Driven Validation Sketch

### C1: Gold correctness linearly accessible in H_1^L
- probe_c_binary on H_1^L → y_c.
- Baselines: token-prob, P(True), random-dir null, shuffled-label null.
- Metric: AUROC + ECE (isotonic-calibrated), bootstrap CIs.
- Expected: AUROC ≥ 0.70 at L*, ECE ≤ 0.10, ECE(probe) < ECE(token_prob).

### C2: Verbalized confidence linearly accessible in H_2^L (pre-emission)
- Primary probe (path chosen by Stage-1.5 diagnostic): probe_v_continuous OR probe_v_ordinal.
- Baselines: random-dir null, shuffled-label null, paraphrase (P1, P2) on 500-dev.
- Metric: Spearman ρ or ordinal accuracy (path-dependent) + binarized AUROC secondary, bootstrap CIs.
- Expected: Spearman ρ ≥ 0.5 (continuous path) or ordinal accuracy well above chance (ordinal path) at L*; binarized AUROC ≥ 0.70; Δ across paraphrase ≤ 0.1.

### C3a: Geometric near-orthogonality
- Primary: cos(v_c^{L*}, v_v^{L*}) with bootstrap 95% CI.
- Secondary: per-layer trajectory; single-pass variant cos(v_c_single^{L*}, v_v_single^{L*}).
- Baseline: cos(v_c^L*, random_unit) null.
- Test: |cos_primary| ≤ 0.3 gated on C1 & C2 passing.

### C3b: Causal separability
- Primary: 3α × 3 directions × 500 samples = 4500 passes; ratio + absolute-effect criterion; measured on internal readout AND emitted output.
- Test: pass criterion above.

### C3c: Dissociation-when-disagree (SECONDARY supporting analysis)
- Bin (probe_c_calibrated, verbalized_conf), report accuracy per cell.
- Test: paired-sample stat test.

## Must-Run Ablations (unchanged from round 1 + single-pass variant added)
1. Random-direction probe null
2. Shuffled-label probe null
3. Paraphrase robustness (500 dev)
4. Per-layer |cos| trajectory + bootstrap CIs
5. Matched-magnitude random-direction steering control
6. Bootstrap CIs on all headline metrics
7. Single-pass unified-prompt robustness (v_c_single, v_v_single, cos_single) — NEW round 3
8. C2 variance diagnostic (Stage 1.5) — NEW round 3 (protocol, not compute-heavy)

## Optional (appendix)
Token-position ablation, calibration-method comparison, question-only lower bound, post-emission upper bound, 6-α robustness slice.

## Failure Modes and Diagnostics
Unchanged from round 1. Additional: if unparseable-rate ≥ 10%, switch to fallback P1 prompt. If Stage-1.5 diagnostic shows std(c) < 15 AND share_at_max > 0.7 → automatic ordinal-path switch.

## Compute & Timeline Estimate
- Total: ~6h main + ~2h verify + buffer = ~8h, within 10h HARD.
- Data cost: none new; ~1 person-hour for 100-sample label-quality verification.
- Timeline: 1 day implementation + 1 day main + 1 day verify + buffer.

=== END REVISED PROPOSAL ===

Please re-score all 7 dimensions. Verdict: READY only if overall ≥ 9 AND no blocking item remains.
