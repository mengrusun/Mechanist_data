# Verify Plan — Claim C4

## Claim C4: steering-vector control offers more distinct (rate, accuracy) operating points than prompt/TI

**Main experiment (from /auto-experiment)**
- Method: Tuning & Editing — CAA steering (α ∈ {-2,-1,+1,+2}×σ_proj·u at L*(b)) vs. NL-prompt (suppress/amplify) vs. Thinking-Intervention token-insertion (suppress/amplify)
- Dataset: 500-task custom reasoning benchmark (60-task subset, n=60)
- Model: DeepSeek-R1-Distill-Llama-8B
- Metric: n_distinct_operating_points per controller family (ε_r=0.05, ε_a=0.01)
- Main-experiment verdict: not-supported (positive-partial for uncertainty: 4 > 2 = 2; preservation misses 3pt floor by 1pt)

## Variants

| # | Dimension | Swap | Replaces | Justification | Source |
|---|-----------|------|----------|---------------|--------|
| 1 | model | DeepSeek-R1-Distill-Qwen-14B | DeepSeek-R1-Distill-Llama-8B | Cross-backbone, cross-size model swap within the R1-Distill family. Tests whether the "more operating points" claim generalizes beyond the Llama-8B architecture. 14B is closer in scale to 8B than 1.5B, making this the most informative cross-backbone test (not a scaling artefact). The Qwen backbone uses a different tokenizer and attention mechanism, providing a genuine architectural stress test. | NOTICE section: DeepSeek-R1-Distill-Qwen-14B listed as primary model-swap candidate |

## Success Criterion
The variant PASSES iff: for expressing_uncertainty on the swapped model, n_distinct_operating_points(steering) > n_distinct_operating_points(prompt) AND n_distinct_operating_points(steering) > n_distinct_operating_points(TI). This is the same primary predicate as C4's main experiment. The secondary predicates (matched-rate accuracy, preservation) are tracked but the binary pass/fail is on the primary predicate.

Since the main experiment is not-supported (positive-partial — the primary predicate passes for uncertainty but the full cross-behaviour claim fails), a variant that also shows n_distinct(steering) > n_distinct(prompt) for uncertainty would be CONSISTENT with the main experiment (both confirm the partial positive for uncertainty). A variant that fails this would be INCONSISTENT.

## Steps for variant (model-swap-qwen-14b):
1. Download DeepSeek-R1-Distill-Qwen-14B to $MODEL_DIR if absent
2. Re-run M1 on the new model (activation capture + direction extraction + L* selection for expressing_uncertainty only) — this extracts the uncertainty direction from the 14B model's activations on the SAME auxiliary corpus
3. Re-run M4 for expressing_uncertainty only (8 controllers × 60 tasks) using the 14B model and its extracted direction
4. Compute n_distinct_operating_points and preservation at α_op for the 14B model
5. Judge whether primary predicate passes (n_distinct(steering) > 2)
