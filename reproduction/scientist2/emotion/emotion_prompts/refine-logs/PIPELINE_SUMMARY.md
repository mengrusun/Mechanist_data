# Pipeline Summary

**Problem**: Characterize the actual behavior of static basic-emotion prefixes as an input-dependent signal on Qwen3-14B, and locate the internal component that carries the emotional frame and causally mediates the observed per-item accuracy shifts.
**Final Method Thesis**: Treat the 26 prompt conditions (6 emotions × 2 intensities × 2 wording sources + neutral + filler) as a controlled grid; measure per-item Δaccuracy across GSM8K (math), SocialIQA (social), MedQA (factual) on Qwen3-14B; locate the frame-carrying residual-stream direction via probing on cached M2 activations; causally verify via patching + dose-response steering with matched-length filler controls; train EmotionRL as a Directional-Stimulus-Prompting-style policy over the discrete 13-way action space and compare against the fixed-emotion argmax on held-out GSM8K.
**Final Verdict**: READY
**Date**: 2026-07-13

## Final Deliverables

- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Refinement report: `refine-logs/REFINEMENT_REPORT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Idea report: `idea-stage/IDEA_REPORT.md`
- Landscape: `idea-stage/LANDSCAPE.md`
- Retrieval dump: `idea-stage/RESEARCH_LIT.md`

## Contribution Snapshot

- **Dominant contribution**: the first paper to place emotional-prefix effects (i) in a FormatSpread-style noise-floor comparison across a math/social/factual task-family grid with the *distribution* per emotion×intensity×wording-source reported, (ii) with a causally-localized mechanism on the same model & task via residual-stream probing + activation patching + dose-response steering with matched-length filler control, and (iii) with a fair head-to-head against a learned adaptive policy (EmotionRL) over the same discrete action space.
- **Optional supporting contribution**: SAE / vocab-projection decoding of the located frame direction (Unit Interpretation) — added only if M5 produces a clean shortlist.
- **Explicitly rejected complexity**: weight-space editing / task-vector tuning; training-time formation tracing / influence functions; free-form policy for EmotionRL; cross-model comparison at this stage.

## Must-Prove Claims

- **C1**: Static emotional prefixes cause only small, input-dependent accuracy shifts (mean|Δ| ≤ format-perturbation p90 noise floor, sign-consistency ∈ [0.4, 0.6] per prefix).
- **C2**: Effects are larger on socially grounded tasks than on math/factual QA (spread_social ≥ 2× spread_math; strict ordering in ≥ 2/3 comparisons).
- **C3a** / **C3b**: No single emotion consistently wins (argmax flips across families); no monotone intensity→gain (≥ 3/6 emotions fail monotonicity CI on GSM8K).
- **C4**: Adaptive EmotionRL beats both neutral and the fixed-emotion argmax on held-out GSM8K, lower CI > 0 across 3 seeds.
- **CM**: Some low-rank residual-stream direction on Qwen3-14B carries the emotional frame identity (Location, M5) and causally modulates GSM8K per-item accuracy (Intervention, M6), with matched-length filler control null and off-target near-null.

## First Runs to Launch

1. **M1** — Build the 26-condition prompt corpus via dmxapi `gpt-5.4` (no GPU; produces `data/prefixes/prefixes.json`).
2. **M2 + M2b** — Launch the 26 GSM8K conditions and 8 format-perturbation conditions on Qwen3-14B in parallel across GPUs {1,2,3,5,6} (~1.4 GPU-h combined; wall ≈ 20 min).
3. **M3** — Launch the 52 (task × condition) SocialIQA + MedQA runs (~1.1 GPU-h; wall ≈ 15 min).

## Main Risks

- **Risk**: M6's need for intra-forward residual-stream hooks forces a framework switch from vLLM to `transformers` for that milestone only.
  **Mitigation**: flagged in REFINEMENT_REPORT.md; M6 is small (0.6 GPU-h), so the switch cost is minimal.
- **Risk**: LLM-generated (`gpt-5.4`) prefix wording may not be length-matched on first call.
  **Mitigation**: M1 uses a 3-candidate strategy and picks the closest length match.
- **Risk**: Qwen3-14B reasoning-mode (`<think>`) could drift the CoT decoding.
  **Mitigation**: `thinking=False` is set at inference time in the runner; logged in each run's metadata.
- **Risk**: M7a reward-table build is the compute-heaviest single milestone (~2.2 GPU-h).
  **Mitigation**: chunked into 5 × 400-item shards; can be re-launched partial.

## Next Action

- Proceed to `/mechanism-skills` (Workflow 1.25) to bind concrete families to M5 (Location) and M6 (Causal Intervention).
- Then `/auto-experiment` (Workflow 1.5) to implement and deploy.
