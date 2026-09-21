# Round 2 Refinement

## Problem Anchor (verbatim from round 0)

See round-0 and round-1 files. Unchanged.

## Anchor Check

Original bottleneck (matched-pair geometric+causal characterization) still central. Reviewer's blocking items #1 and #3 tighten the **narrative around** the anchor, not the anchor itself. Item #2 (C2 signal contingency) is a robustness fix, not a claim change. No drift.

## Simplicity Check

- Dominant contribution: matched-pair, canonical-hook, same-space characterization + causal cross-steering. Unchanged.
- Components removed or merged: report cosine + steering at ONE primary layer L*, not two (matched-best-AUROC + nearest-shared → one primary rule "argmax mean-normalized AUROC_c + AUROC_v").
- Reviewer suggestions rejected as complexity: none.

## Changes Made

### 1. Add "Scope of the Mechanism Claim" paragraph (fixes framing overreach)

- **Reviewer said**: "Do not overinterpret result as 'readout failure'; support only 'linearly separable and weakly aligned' + 'limited causal coupling under tested prompts'."
- **Action**: add explicit Scope paragraph to the proposal that (a) states the anchor's evocative framing ("knowledge deficit vs. readout failure") as the *motivating dichotomy*, not the *asserted conclusion*, and (b) restricts the *asserted evidence-supported conclusion* to "linear separability with low geometric alignment and limited causal cross-coupling in this model, on this dataset, under this canonical hook, on the tested prompts". The three CLAIMS (C1, C2, C3) themselves — the operational measurable predicates — remain unchanged.
- **Impact**: paper is defensible against reviewer skepticism; downstream conclusions honestly bounded.

### 2. Add C2 signal-strength diagnostic + contingency (fixes weak-signal risk)

- **Reviewer said**: "If confidence values bunch near 100, Spearman may be unstable / binary median may be semantically thin. Contingency for low-variance targets."
- **Action**:
  - Add Stage 1.5 "**Verbalized-confidence variance diagnostic**" — before probe training, report the empirical distribution of c: mean, std, entropy, quantiles, share-at-100, share-at-0.
  - **Decision rule**:
    - If std(c) ≥ 15 AND share_at_max ≤ 0.7 → proceed with `probe_v_continuous` (linear regression) as primary + `probe_v_binary` at median as secondary.
    - Else (low variance) → switch primary probe to `probe_v_ordinal`: a 4-bin ordinal probe on {0-25, 26-50, 51-75, 76-100} bins trained with ordinal cross-entropy; extract v_v as the L2-normalized weight vector of a binary probe at the *empirical* median (which under a heavy-100-skew scenario is closer to 90 or 100, so binarize instead at the 30th percentile to guarantee ≥30% minority class); primary C2 metric becomes ordinal probe accuracy + Spearman ρ over ordinal bins.
- **Impact**: C2 has a data-driven contingency, not a footnote.

### 3. Explicit two-context caveat + single-pass robustness variant (fixes C3 identifiability tightness)

- **Reviewer said**: "State C3 is about same-model, same-layer residual geometry across two elicitation contexts, not a single shared task state."
- **Action**:
  - Add "Scope Statement" paragraph in the proposal.
  - Add **single-pass robustness variant** as a new ablation (moved from "no such thing" to must-run): a *unified prompt* containing both the question AND the confidence elicitation in ONE forward pass. From that single hidden state at layer L, train BOTH probe_c_single and probe_v_single. Report `cos(v_c_single^L, v_v_single^L)` alongside the two-pass primary. If both give consistent |cos| ≤ 0.3, the C3a claim is robust to the two-context concern. If they diverge (e.g., single-pass gives cos ≈ 0.5), we report *both* faithfully — the two-pass measurement is the primary (it's what the paired-sample design provides cleanly), and the single-pass measurement is a robustness diagnostic.
- **Impact**: reviewer's biggest remaining methodological concern is neutralized by construction.

### 4. Collapse "matched best-AUROC" logic to a single primary reporting layer

- **Reviewer said**: "Pick one primary rule and make the other explicitly robustness-only."
- **Action**: PRIMARY REPORTING LAYER: `L* = argmax_L [ AUROC_c(L)/max_L AUROC_c(L) + AUROC_v_binary(L)/max_L AUROC_v_binary(L) ]` — the single layer L that maximizes the mean of the two normalized AUROCs. Report cos(v_c^{L*}, v_v^{L*}) as C3a primary. Steering (C3b) executed at layer L* using v_c^{L*} and v_v^{L*}. Per-layer trajectories reported as robustness only.
- **Impact**: no ambiguity in primary numbers; the paper has ONE headline cos number.

### 5. Add absolute-effect companion criterion to C3b ratio

- **Reviewer said**: "|Δ_v_other|/|Δ_random| ≤ 1.5 can be unstable with tiny random denominator."
- **Action**: C3b passes iff EITHER (i) `|Δ_v_other| / |Δ_random| ≤ 1.5` when `|Δ_random| ≥ 0.5σ_probe_readout` OR (ii) `|Δ_v_other| ≤ 0.5σ_probe_readout` (absolute-effect small when the ratio is undefined/unstable). Reported on BOTH internal readout AND emitted output.
- **Impact**: ratio criterion robust when random effects are near-zero.

### 6. Include effect-size distributions and bootstrap CIs primary, not thresholds primary

- **Reviewer said**: "Reviewers want continuous evidence: distributions of cosine, steering deltas, bootstrap CIs, and variance across samples/layers—not just threshold passes."
- **Action**: report per-layer AUROC_c, AUROC_v, Spearman ρ, and |cos| with 95% bootstrap CIs (1000 bootstrap resamples on the training set). For steering, report the full distribution of Δ across the 500 held-out samples per condition (mean ± bootstrap 95% CI + violin/box across samples). Threshold-pass table is a *derived* summary, not the primary evidence.
- **Impact**: paper is quantitatively rigorous rather than pass/fail.

### 7. Preempt representation-space objections in the writeup

- **Action**: dedicated methodological paragraph in the proposal explaining WHY the canonical hook + last-input-token position is the right choice. Explanation: (a) it is the state the model uses to produce the next token, so it is the maximally-informative pre-emission representation; (b) it is the convention shared with Marks-Tegmark 2023 and the majority of subsequent truthfulness-probing work, so cosine numbers are directly comparable across the literature; (c) using the same hook + same-position choice on both forward passes is what makes cos a well-defined geometric quantity.

### 8. Minor items accepted
- Confidence-number parsing spec: extract the first integer in [0, 100] (or the first decimal in [0.0, 1.0] × 100 for P1, or the mapped Likert bin for P2) from the model's output up to 8 tokens after "Confidence:"; if no valid number, mark the sample as "unparseable" and exclude from C2 (report unparseable rate as an integrity metric).
- 100-sample audit rephrased as "label-quality verification" — not exhaustive adjudication.
- Dissociation-when-disagree analysis (C3c) explicitly labeled "secondary supporting analysis."

## Revised Proposal (Round 2 → Round 3 candidate)

See `refine-logs/FINAL_PROPOSAL.md` for the clean version. Full inline duplication omitted here to keep this round file focused on *changes*.
