# Round 1 Review — Parsed Summary

**Overall score**: 7.1/10
**Verdict**: REVISE
**Drift Warning**: NONE

## Dimension scores

| Dim | Score |
|-----|-------|
| Problem Fidelity | 8/10 |
| Method Specificity | 8/10 |
| Contribution Quality | 7/10 |
| Frontier Leverage | 8/10 |
| Feasibility | 6/10 |
| Validation Focus | 6/10 |
| Venue Readiness | 6/10 |

## Priority action items

### CRITICAL 1 (Feasibility) — trim the compute-heavy plan
- Reduce steering grid from 6 α values × 3 directions × 2k = 36k passes → 3 α values × 3 directions × subset = ~12-18k passes
- Move token-position ablation, calibration-method comparison, question-only lower bound, post-emission upper bound to *optional* (not must-run)
- Paraphrase robustness on 500-example dev slice (not full test)
- Explicitly cache answer strings + confidence prompts to avoid duplicate generation

### CRITICAL 2 (Validation Focus / C3 identification) — fix canonical hook
- Define ONE canonical residual-stream hook location (e.g., residual-stream post-block-out at layer L) and report cosine primarily *at that same hook location* on both forward-pass distributions
- If best-C1 and best-C2 layers differ, ALSO report cosine at the nearest shared layer (this is a control, not a replacement)
- For C3b, measure NOT ONLY probe readout change but ALSO the actually emitted confidence number (for v_v-steering) — an outcome-level test on the observable, not just an internal-readout test. Where compute is limiting, report only for the smaller α grid but still on the observable.

### IMPORTANT 1 (Venue Readiness) — sharpen the C3 narrative
- Frame as "matched-pair, same-representation-space characterization" with one canonical hook
- Move broader diagnostics (token-position, calibration variants, question-only baseline) to appendix / optional

### Simplifications
1. Collapse C2 primary to continuous linear regressor (Spearman ρ as primary metric), report binarized AUROC as derived-secondary. Direction vector v_v extracted from the *binarized* logistic regression to keep it unit-length and directly comparable with v_c (both are unit vectors from binary probes).
2. Bootstrap CIs on per-layer probe scores + cos values.

### Non-drift note
Reviewer explicitly says drift = NONE. Claims C1, C2, C3 still faithful. Adjustments are all method-level, not claim-level.

## Concrete method-level changes for Round 1 refinement
1. **Canonical hook**: pin the residual-stream hook to "block output after the final residual-add of block L" (equivalent to Marks-Tegmark's convention; the same hook must apply across forward passes 1 and 2).
2. **Steering grid**: {-1σ, 0, +1σ} × {v_c, v_v, random} × 500 held-out test samples = 4500 forward passes (~1h with vllm-batched).
3. **Cross-steering outcome metric for v_v**: emit and parse the actual confidence number under steering, not just probe readout.
4. **Must-run ablations pruned**: keep (a) random-direction probe null, (b) shuffled-label probe null, (c) paraphrase robustness (500 samples), (d) per-layer trajectory of |cos|, (e) matched-magnitude random-direction steering control, (f) bootstrap CIs. Drop token-position ablation, calibration-method comparison, question-only baseline, post-emission upper bound → optional appendix if time.
5. **C2 primary probe**: linear regression on continuous c (Spearman ρ primary metric); derived binarized AUROC.

## Raw response

<details>
<summary>Round 1 raw response (verbatim)</summary>

See `round-1-review-raw.md` for the full verbatim response.

</details>
