# Round 2 Review — Parsed Summary

**Overall score**: 8.5/10 (up from 7.1)
**Verdict**: REVISE (need ≥9 for READY)
**Drift Warning**: NONE

## Dimension scores

| Dim | Round 1 | Round 2 |
|-----|---------|---------|
| Problem Fidelity | 8 | 8.8 |
| Method Specificity | 8 | 9.0 |
| Contribution Quality | 7 | 8.1 |
| Frontier Leverage | 8 | 8.4 |
| Feasibility | 6 | 8.8 |
| Validation Focus | 6 | 8.4 |
| Venue Readiness | 6 | 8.0 |

## Priority action items

### Blocking (must fix for READY)

1. **Tighten interpretive framing.** What the linear-probe + steering evidence supports is: "correctness and verbalized confidence are linearly separable and weakly aligned in the chosen residual space" + "local steering suggests limited causal coupling between them under the tested prompts." Full "readout failure" is beyond what the evidence supports. **Fix**: add an explicit "Scope of the Mechanism Claim" paragraph and rewrite hedge-language in Method Thesis / Contribution Focus to match the evidence ceiling. The C1/C2/C3 claim predicates themselves are UNCHANGED (they're operational tests, not interpretive claims); only the *narrative framing* around them is tightened.

2. **Weak-training-signal contingency for C2.** If verbalized confidence values cluster near 100 (which the anchor itself predicts!), Spearman ρ may be unstable and median-binarized labels may be semantically thin. **Fix**: (a) as an *early diagnostic* before training probes, report the empirical distribution of c across the 10k samples — target-variance, entropy, effective sample spread; (b) if the distribution is too narrow to support probe_v_continuous, switch primary to a discrete probe on a fixed-bin scheme ({0-25, 26-50, 51-75, 76-100}) with ordinal cross-entropy; (c) report this diagnostic as part of the C2 evidence chain, not as a fallback that hides in a footnote.

3. **Two-context caveat.** State explicitly that the C3a claim is about "same-model, same-layer residual geometry across two elicitation contexts" — not a single shared task state. **Fix**: add a "Scope Statement" that clarifies the two-context nature of the measurement, and add a *within-context* control experiment where we also compute cos(v_c^{L}, v_v^{L}) using activations from a *single unified prompt* that contains both the question and the confidence elicitation in one pass (a robustness variant), reporting both the two-pass primary and the single-pass secondary.

### Important (non-blocking)

4. Cross-steering ratio criterion `|Δ_v_other| / |Δ_random| ≤ 1.5` can be unstable if the random denominator is tiny. **Fix**: add a companion absolute-effect criterion (e.g., |Δ_v_other| in probe-readout σ units) that fires when the ratio is undefined.

5. Justify last-input-token position: "it is the final pre-emission state most directly poised to determine the next token."

6. State parsing/normalization for confidence numbers across P0/P1/P2 explicitly.

7. Frame 100-sample manual audit as label-quality verification, not exhaustive adjudication.

### Simplifications (accepted)

- **Collapse "matched best-AUROC" logic**: pick ONE primary rule — "cosine + steering reported at the single shared layer L* maximizing (mean-normalized AUROC_c(L) + mean-normalized AUROC_v(L))"; per-layer trajectories as robustness only.
- **De-emphasize the binary/continuous C2-probe seam**: present the binary probe as an "operational extractor for the steering vector v_v" (a technical device), and the continuous probe as the C2 primary evidence — one narrative sentence to prevent target-switching perception.
- **Keep C3c (dissociation-when-disagree) clearly secondary**: label it a supporting analysis, not a headline claim.

### Modernization (accepted)

- Report full distributions of cos and steering deltas + bootstrap CIs across samples/layers, not just threshold passes.
- Lean into paired-sample framing throughout the writing.
- Preempt representation-space objections in the proposal itself.

## Raw response

See `round-2-review-raw.md` for the full verbatim response.
