# Round 2 Refinement

## Problem Anchor (verbatim from round 0 — unchanged)

Three given claims about emotion-specific circuits in Llama-3.2-3B on SEV; matched-budget verification protocol; not proposing a new mechanism family; not tracing formation; no SAE; no fine-tuning.

## Anchor Check
- Preserved.
- All fixes are protocol precision, not claim changes.

## Simplicity Check
- Dominant contribution unchanged.
- Rank fusion simplified to Stage A = shortlist, Stage B = ranker (as reviewer suggested).
- Judge audit compressed to a QA note in the plan (subsystem framing removed).
- Claim 2e (neuron overlap < head overlap) explicitly demoted to *secondary* predicate.

## Changes Made

### 1. Stage-B causal-ranking metric fully specified (CRITICAL 1)
- **Reviewer said**: `Δ(target-emo score)` is under-specified — determines who enters `C_e`.
- **Action**: Stage B ranks each shortlisted component by the **change in the log-probability of the target-emotion continuation prefix** under enhancement of that single component at fixed `α = α2`, aggregated by mean over a fixed val subset. Concretely:
  - Input: paired `(event_stem, target_emotion e)`. The judge / classifier is **not** used at Stage B (avoids circularity with Claim 3's judge and keeps Stage B cheap and internal).
  - Fixed prompt template: `"{event_stem}"` followed by a fixed continuation prefix `"I feel {emotion_word_e}"` (natural-language emotion word for e, e.g., "joy" / "anger" / …). The scored quantity is `log P(prefix | event_stem)` under the model with the component enhanced.
  - Score per component `c`: `s_c = mean_{val} [ log P(prefix_e | event_stem; enhance c at α2) − log P(prefix_e | event_stem) ]`, where val = 30 event stems per emotion drawn from the val fold.
  - Ranking: rank all shortlisted components (heads + neurons separately) by `s_c` descending. Top-`k_h` heads and top-`k_n` neurons enter `C_e`.
- **Rationale**: The **prefix-logprob** score is an internal, judge-free, deterministic, cheap-per-item metric. It measures the *causal* contribution of the component to the model's own tendency to continue emotionally. Aggregation is a plain mean.
- **Impact**: Removes the largest residual DoF; makes Stage B fully reproducible.

### 2. Arm C direction construction locked pre-eval-split (CRITICAL 2)
- **Reviewer said**: matched-budget locks tuning but not direction construction — single-direction steering critically depends on direction source and injection semantics.
- **Action**: Freeze Arm C's direction construction **before any eval-split activation is read**:
  - **Direction source**: `d_e = mean_diff` of *residual-stream* activations at the **last event-stem token** (same token position used for `d_{e,L}` in Location) between positive-e prompts (event + emotion-tag suffix for e) and negative prompts (event + emotion-tag suffix for e′ ≠ e, uniformly sampled over the 5 off-target emotions, one per pair). Averaged over the train fold event stems (no eval leakage).
  - **Injection site**: **residual stream** at the chosen layer L (from the val grid), immediately after that layer's output projection, added at **every token position after the event stem** during autoregressive generation (CAA convention). No per-head, per-position schedule.
  - **Injection semantics**: additive with scalar `α_C` (from the val grid).
  - **Tuning grid** on val: layer L ∈ top-3 layers by paired-contrast probe AUC on the train fold (same shortlist used in Stage A of the circuit locator — matched across arms); `α_C` ∈ {0.5, 1.0, 2.0}.
- **Rationale**: This is the canonical RepE/CAA recipe with all pieces frozen pre-eval-split. The comparison to Arm A is now genuinely apples-to-apples: same direction-extraction protocol, same layer shortlist, same activation-space injection semantics — Arm A adds *component-level sparsity + specificity*, everything else matched.
- **Impact**: Arm C fairness fully addressed.

### 3. Off-target mean pool for ablation clarified (Important)
- **Action**: For each event stem `s` and each target emotion `e`, the off-target mean pool for mean-substitution is the average per-neuron / per-head activation over the FIVE other emotion variants of the SAME stem `s` (SEV guarantees six variants per event by construction — see the sev.json inspection step). If for some reason a stem is missing a variant (data-integrity check), that stem is dropped from the ablation batch and logged; no imputation.
- **Rationale**: SEV's design makes this well-defined; the mean over 5 off-target variants of the same stem is a natural "generic activation on this event" reference.
- **Impact**: Ablation operator is now fully specified.

### 4. `k_h, k_n` selection is explicitly ONE choice for all six emotions (Important)
- **Action**: `k_h` and `k_n` are selected ONCE globally, from `{24, 48, 96} × {2000, 4000, 8000}` on val, by maximizing **macro-average target prefix log-prob gain across all six emotions** at fixed `α = α2`. The *same* `(k_h*, k_n*)` are then used for all six emotions in Claim 1's Jaccard reporting and Claim 2's causal / stability tests. No per-emotion tuning of `k`.
- **Rationale**: Prevents the "different k per emotion → hidden researcher DoF" concern.
- **Impact**: reproducibility + fairness.

### 5. Explicit success rubric for Claim 2 (Important)
- **Action**: Claim 2 has FIVE predicates (a-e in the round-1 revision); classification:
  - Claim 2 is **fully supported** iff all of (a) causal-sign + dose-response, (b) raw-off-target specificity, (c) targeted-`C_{e'}` specificity, (d) scenario stability are supported by directional-sign predicates with paired-bootstrap 95% CIs excluding 0 / crossing null-CI upper edge; predicate (e) (neuron overlap < head overlap) is **secondary** and does not gate full support.
  - Claim 2 is **partial** iff (a) + at least one of (b, c) hold, but (d) fails.
  - Claim 2 is **causal-only** iff (a) holds but (b) and (c) both fail.
  - Claim 2 is **not supported** iff (a) fails (no causal sign or no dose-response).
- **Rationale**: makes mixed-outcome interpretation deterministic, closes reviewer's concern about fragmented reads.

### 6. Rank fusion removed (Simplification)
- **Action**: Stage A produces a shortlist (top-3 layers by AUC; within those layers, top-5% neurons and top-20% heads by cheap alignment / probe score). Stage B is the **sole ranker** on the shortlist. `C_e` = top-`k_h` heads + top-`k_n` neurons by Stage-B `s_c`.
- **Impact**: Cleaner story; one causal ranker; no need to defend a rank-fusion rule.

### 7. Claim 2e demoted to secondary (Simplification)
- **Action**: Claim 2e is labeled **secondary** in FINAL_PROPOSAL.md and EXPERIMENT_PLAN.md. It is *reported* with a bootstrap CI on the difference; it does *not* gate Claim 2 support (per rubric above).

### 8. Judge audit compressed (Simplification)
- **Action**: The judge-reliability section in FINAL_PROPOSAL.md is now a single-paragraph QA check: 60 gold items, agreement ≥ 0.75 gate, 10% judge-swap, fallback → SEV-emotion classifier. No parallel "audit subsystem" framing.

### 9. Length threshold clarified as reporting-only (Nice-to-have)
- **Action**: 15% length-mismatch triggers a *secondary* length-matched analysis; the primary finding uses uncapped length. This is a *reporting* threshold, not a decision threshold.

### 10. Judge calibration reported (Nice-to-have)
- **Action**: On the 60-item gold subset, report the per-emotion **confusion matrix** (not just aggregate agreement), so any systematic per-emotion bias is visible.

## Revised Proposal — SEE FINAL_PROPOSAL.md

The finalized proposal is written to `refine-logs/FINAL_PROPOSAL.md` (containing every change from Round 0 through Round 2). This refinement document is the audit trail; the FINAL_PROPOSAL is the clean deliverable.
