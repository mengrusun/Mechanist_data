# REFINEMENT REPORT

**Behavior-source**: given (from task.md — behavior/claim wording never altered)
**Mechanism**: discovery
**Date**: 2026-07-13
**Iteration count**: 1 (internal — external LLM review is `/auto-verify`'s job for a `given` behavior)

## Problem Anchor (frozen)

Characterize the actual behavior of static basic-emotion prefixes as an input-dependent signal on Qwen3-14B, and locate the internal component in Qwen3-14B that carries the emotional frame and causally mediates the observed per-item accuracy shifts. Anchor never moves through refinement.

## What was refined

The four claims C1–C4 were **not** altered — they are the task.md verbatim capture. Refinement acted only on the *testing method*:

1. **Sample-size resolution.** C1's "small, input-dependent" required an operational threshold: chose the M2b format-perturbation p90 as the noise-floor threshold; chose n=500 items per condition to give macro-accuracy ±4.4 pp CI at p=0.5.
2. **Task-family companion resolution.** C2 required a socially grounded and factual companion; picked SocialIQA (social) + MedQA (factual) from task.md's verify-stage candidate list because they are already blessed by task.md's own resource list and cleanly match the "social inference" and "factual QA" definitions in the claim.
3. **Confound control resolution.** Added a length-matched non-emotional filler prefix (condition #26) as the specificity control for both the behavioral claims (C1) and the mechanism claim (M6 filler-control gate).
4. **Mechanism altitude discipline.** Rejected pinning any specific layer/head at claim time; specified only that the mechanism claim is at the *kind-of-component* level (some low-rank residual-stream direction and/or small head set), leaving M5 to *discover* the specific identities.
5. **EmotionRL policy architecture resolution.** Fixed the Directional Stimulus Prompting architecture (arXiv:2302.11520) with a 13-way discrete action space, Llama-3.2-1B backbone (with bert-base fallback), SFT-then-RL against a cached Qwen3-14B reward table so the RL step does not require additional heavy forward passes.
6. **Budget-conscious sequencing.** M2 caches activations in-flight so M5 requires no extra Qwen3-14B forward passes. M7 caches Qwen3-14B rewards so RL fine-tune is CPU/1-GPU cheap. Total sums to ~6.5 GPU-h, respecting the 10 GPU-h envelope with headroom.

## Rejected complexity (rationale)

Same as `mechanism_strategy.rejected` in FINAL_PROPOSAL.md. All four rejections are motivated by (a) matching the task.md scope, (b) fitting the 10 GPU-h budget on a 14B model, or (c) staying at the correct mechanism altitude.

## Remaining risks

- **M6 hook framework switch**: M2/M3 use vLLM for throughput; M6 requires intra-forward residual-stream write access which vLLM does not expose, so M6 uses `transformers` + custom hooks. This is a framework switch, not a plan risk — flagged for the implement stage.
- **LLM-generated wording length control**: dmxapi `gpt-5.4` may not produce length-matched (±10 tokens) variants on first call; M1 uses a 3-candidate-per-cell strategy and picks the closest length match.
- **Qwen3-14B reasoning-mode default**: Qwen3 has an optional `<think>...</think>` mode; the runner disables it (`thinking=False` at inference) for GSM8K to keep decoding deterministic; the choice is logged in each run's metadata.

## Verdict

**READY** — hand off to `/mechanism-skills` (Workflow 1.25) for M5/M6 family binding, then `/auto-experiment` (Workflow 1.5).
