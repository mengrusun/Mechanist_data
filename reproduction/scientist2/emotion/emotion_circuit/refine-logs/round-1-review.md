# Round 1 Review

**Date**: 2026-07-13
**Reviewer**: external LLM (gpt-5.4) via dmxapi
**Overall Score**: 7 / 10
**Verdict**: REVISE
**Drift Warning**: NONE

## Parsed Scores

| Dim | Score |
|---|---|
| Problem Fidelity | 8 |
| Method Specificity | 6 |
| Contribution Quality | 8 |
| Frontier Leverage | 7 |
| Feasibility | 6 |
| Validation Focus | 6 |
| Venue Readiness | 7 |
| **Overall** | **7** |

## Action items ranked

### CRITICAL — Method Specificity (dim 2, score 6)

Pin down the exact interfaces:

1. **Component selection**: fix a single score-per-family; single aggregation rule (top-k globally after within-layer z-scoring); fix `k_h, k_n` by val sweep from a small discrete grid shared across emotions (not post-hoc per emotion); explicit layer cap or explicit unrestricted-with-global-cardinality-cap.
2. **Ablation**: one primary operator only — **mean-substitution from matched off-target / control examples** (not zeroing). Zero-ablation only as a robustness appendix.
3. **Enhancement**: one primary operator only — **additive activation injection on selected components with scalar α**.
4. **Judge**: forced-choice 6-way classification with fixed rubric; hide the target label from the judge input; pre-register ONE primary endpoint (either unconditional predicted-emotion accuracy OR "does continuation express target emotion?" binary).
5. **Matched budget**: defined as *number of validation hyperparameter evaluations*. Each arm gets exactly `N` val trials per emotion. If circuit tunes `{k_h, k_n, α}`, baseline C must get an equally sized `{layer, α, position}` search or, alternatively, add a restricted circuit variant matched to baseline complexity.

### CRITICAL — Validation Focus (dim 6, score 6)

1. **Add a within-family size-matched targeted control**: ablate/enhance `C_{e'}` while evaluating on emotion `e`, for `e' ≠ e`. This is stronger than random-set null and directly tests specificity.
2. **Fix generation**: same decoding params across all arms (greedy or T=0), report average output length per arm; if materially different, add length-matched secondary analysis or cap output length uniformly.
3. **Normalize overlap for Claim 2c**: enforce equal set sizes across emotions within family (heads / neurons), or supplement Jaccard with permutation / size-matched null.
4. **Primary endpoint for Claim 3** must be ONE pre-registered scalar (not a mix of label and confidence).

### IMPORTANT — Feasibility (dim 5, score 6)

1. **Two-stage locator**: Stage A cheap filter (alignment/probe score) → shortlist top 1-2 layers per emotion + candidate components; Stage B causal scoring only on the shortlist. Do NOT run leave-one-out at full 28 × 8192 neuron scale.
2. **Reduce resampling**: 3 data resamples OR 3 seeds, not both, unless the selection is stochastic (deterministic mean-diff extraction ⇒ seeds unnecessary).
3. **Restrict dose-response**: 3 strengths at ONE finalized recipe, not multiple variants.
4. **Verify swap**: reduced scope — only Claim 3 primary comparison on the Qwen model, not the full ladder.
5. **Fix decoding**: state exact number of generated continuations per arm; temperature 0 fixed.

### Simplification Opportunities

1. Delete seed resampling (`C_e` extraction is deterministic mean-diff).
2. Merge judge-swap ablation + fallback into one reliability section with one primary gate + one backup scorer.
3. Restrict Claim 2 interventions to ONE ablation + ONE enhancement operator; kill "OR" language.

### Modernization Opportunities

1. Add optional secondary metric: LLM-judge pairwise comparative scoring (same stem + target emotion, which continuation better expresses target?) — keep the primary metric single.
2. Judge calibrated-uncertainty flag: low-confidence items go to fallback classifier adjudication on a small subset.
3. Require `/mechanism-skills` to output a single fixed scoring family *before any eval split is touched* — preserves adaptivity without opening researcher degrees of freedom.

## Raw response

<details>
<summary>Round 1 raw response (gpt-5.4)</summary>

See `round-1-review.raw.md` (saved separately).

</details>
